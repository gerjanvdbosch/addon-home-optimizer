import logging
import math
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from domain.time import local_day_start, to_local_time
from domain.types import (
    Config,
    SeriesPoint,
    State,
)
from features.building import BuildingLumpedIdentifier, BuildingThermalIdentifier
from features.dataset import DatasetBuilder, DatasetDefinition, DatasetLoader
from features.solar import (
    PREDICT_STEP_MINUTES,
    SolarBiasIdentifier,
    nowcast_solar,
    predict_solar,
    predict_solar_band,
)
from infrastructure.repositories import ConfigRepository, StateRepository

logger = logging.getLogger(__name__)

# How far past the present the zone forecast reaches. Open-Meteo's own
# horizon on this installation is about 38 hours, so asking for two days takes
# whatever it has without ever being the binding limit.
FORECAST_HORIZON = timedelta(days=2)


class StateManager:
    def __init__(
        self,
        loader: DatasetLoader,
        state_repository: StateRepository,
        config_repository: ConfigRepository,
        models_path: Path,
        latitude: float,
        longitude: float,
    ):
        self.loader = loader
        self.state_repository = state_repository
        self.config_repository = config_repository
        self.models_path = models_path
        self.latitude = latitude
        self.longitude = longitude

    def load(self) -> State:
        return self.state_repository.load()

    def update(self) -> None:
        now = datetime.now(timezone.utc)

        # Local midnight, not UTC midnight: the dashboard shows the local day,
        # and UTC midnight would drop its first hours east of Greenwich.
        start = local_day_start(now).astimezone(timezone.utc)

        config = self.config_repository.load()

        # Loaded from the previous local midnight and trimmed afterwards: sensors
        # filled with their previous value need a reading from before midnight,
        # or the day's first interval stays empty (seen as a 00:15 start).
        df = self.loader.load(
            self._dataset(config),
            local_day_start(now, days=-1).astimezone(timezone.utc),
            now,
        )
        df = df[df["time"] >= start]

        state = self._map(df, self.load(), config=config)

        self._predict_solar(state, now)
        self._simulate_climate(state, config, now)

        self.state_repository.save(state)

    def _simulate_climate(self, state: State, config: Config, now: datetime) -> None:
        """Reconstructs the zone temperature the calibrated building model
        implies, so the dashboard can show it against the measurement.

        Draws the filter's running state estimate rather than a sequence of
        fixed-horizon rollouts. The filter corrects every step, so its output
        is continuous instead of jumping back to the measurement every horizon
        - those jumps are the model's own forecast error, which belongs in
        validate()'s metrics, not in a line that is supposed to be read as a
        temperature.

        That needs the two-node structure, even though the single-node one
        currently scores better: only this one HAS a thermal mass, and its
        temperature is the interesting quantity here because nothing measures
        it. The air estimate alone would just retrace the sensors.

        Loads its own window because the building model needs inputs the state
        dataset does not carry - irradiance components, shutter positions.

        Leaves an existing curve untouched whenever the model is not calibrated
        yet or the window has no gap-free stretch long enough to simulate,
        rather than replacing a real curve with an empty one.
        """

        identifier = BuildingThermalIdentifier(self.latitude, self.longitude)
        identifier.load(self.models_path)

        if identifier.model is None:
            return

        dataset = identifier.dataset(config)

        warmup = timedelta(hours=identifier.MASS_WARMUP_HOURS)
        start = local_day_start(now, days=-1).astimezone(timezone.utc) - warmup

        try:
            loaded = self.loader.load(dataset, start, now)
            estimated = identifier.estimate(loaded)
            measured = identifier.prepare(loaded).set_index("time")["T_air"]
        except ValueError:
            return

        # The air estimate is not stored: with the process noise set from the
        # model's own unreliability the filter gain is essentially one, so it
        # reproduces the measurement exactly and a second identical line says
        # nothing. How far the model's own forecast drifts belongs in
        # validate()'s metrics, not in a line read as a temperature.
        state.measurements.climate.zone_temperature = self._series_points(measured)

        # The mass line runs straight on into the forecast: the filter's last
        # estimate is what the rollout starts from, so there is no seam.
        mass = estimated["mass"]

        forecasts = self._forecast_zone(dataset, start, now)

        if forecasts is not None:
            two_node, lumped = forecasts
            mass = pd.concat([mass, two_node["mass"].iloc[1:]])
            state.predictions.zone_forecast_two_node = self._series_points(
                two_node["air"]
            )
            state.predictions.zone_forecast_lumped = self._series_points(lumped["air"])

        state.predictions.thermal_mass = self._series_points(mass)

    def _forecast_zone(
        self,
        dataset: DatasetDefinition,
        start: datetime,
        now: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Both structures rolled forward, or None if either cannot be.

        Loaded past the present so the weather forecast's own rows come along -
        that is where the irradiance and outdoor temperature a forecast needs
        actually live.
        """

        loaded = self.loader.load(dataset, start, now + FORECAST_HORIZON)

        state = self.load()
        baseload = pd.Series(
            {point.time: point.value for point in state.predictions.baseload}
        )

        results = []

        for cls in (BuildingThermalIdentifier, BuildingLumpedIdentifier):
            identifier = cls(self.latitude, self.longitude)
            identifier.load(self.models_path)

            if identifier.model is None:
                return None

            identifier.dataset(self.config_repository.load())

            try:
                results.append(
                    identifier.forecast(
                        loaded, now, baseload if not baseload.empty else None
                    )
                )
            except ValueError as error:
                # Normal before the weather sensor's horizon reaches past now,
                # or right after a restart with no measurements yet. Logged
                # rather than swallowed: a silent None here once hid a plain
                # coding mistake for a whole run.
                logger.warning("No zone forecast from %s: %s", identifier.name, error)
                return None

        return results[0], results[1]

    def _predict_solar(self, state: State, now: datetime) -> None:
        """Builds state.predictions.solar and its calibrated p10/p90 band from
        the live Solcast forecast curves and whatever SolarBiasIdentifier last
        calibrated, so the MPC optimizer and the dashboard chart always see a
        forecast as current as the state refresh itself. Leaves existing
        predictions untouched when there's no Solcast curve or no calibrated
        model yet, rather than clobbering a stale-but-real prediction with an
        empty one.
        """

        forecast = state.forecast.solcast

        if not forecast.p50:
            return

        identifier = SolarBiasIdentifier(self.latitude, self.longitude)
        identifier.load(self.models_path)

        if identifier.model is None:
            return

        step = timedelta(minutes=PREDICT_STEP_MINUTES)
        nowcast = self._current_quarter_nowcast(state.measurements.solar, now)

        def from_now(series: pd.Series) -> pd.Series:
            # Only quarter hours that haven't ended. The running one - the step
            # the optimizer acts on - takes the measurement-based estimate where
            # there is one, in all three solar scenarios alike: its spread over
            # the next few minutes is not calibrated.
            series = series[series.index > now - step]

            if nowcast is not None and nowcast[0] in series.index:
                series = series.copy()
                series[nowcast[0]] = nowcast[1]

            return series

        p50 = predict_solar(
            identifier.model,
            self._future_series(forecast.p50, now),
            self.latitude,
            self.longitude,
        )
        state.predictions.solar = self._series_points(from_now(p50))

        if forecast.p10 and forecast.p90:
            p10, p90 = predict_solar_band(
                identifier.model,
                self._future_series(forecast.p10, now),
                self._future_series(forecast.p90, now),
                now,
            )
            state.predictions.solar_p10 = self._series_points(from_now(p10))
            state.predictions.solar_p90 = self._series_points(from_now(p90))
        else:
            state.predictions.solar_p10 = []
            state.predictions.solar_p90 = []

    @staticmethod
    def _future_series(points: list[SeriesPoint], now: datetime) -> pd.Series:
        series = pd.Series(
            [point.value for point in points],
            index=pd.DatetimeIndex([point.time for point in points]),
        )

        # From the period running at `now`: Solcast times are period starts, so
        # that period's value is the forecast for right now - starting after
        # `now` instead left the optimizer's plan beginning up to 30 minutes
        # ahead.
        running = series.index[series.index <= now]

        return series[series.index >= running.max()] if len(running) else series

    @staticmethod
    def _latest_measurement(
        measured: list[SeriesPoint], now: datetime
    ) -> tuple[pd.Timestamp, float, pd.Timestamp] | None:
        """(start of the quarter hour running at `now`, the latest measured
        quarter-hour mean, the middle of the time that mean covers), or None
        without a measurement from this or the previous quarter hour - e.g. the
        PV sensor stops reporting at night."""

        if not measured:
            return None

        step = timedelta(minutes=PREDICT_STEP_MINUTES)
        quarter = pd.Timestamp(now).floor(f"{PREDICT_STEP_MINUTES}min")
        latest_start = pd.Timestamp(measured[-1].time)

        if not quarter - step <= latest_start <= quarter:
            return None

        # A running quarter hour's mean covers its start up to now.
        latest_end = min(latest_start + step, pd.Timestamp(now))

        return (
            quarter,
            float(measured[-1].value),
            latest_start + (latest_end - latest_start) / 2,
        )

    def _current_quarter_nowcast(
        self, measured: list[SeriesPoint], now: datetime
    ) -> tuple[pd.Timestamp, float] | None:
        """(start of the quarter hour running at `now`, PV output expected over
        it) from the latest measurement (see nowcast_solar), or None."""

        latest = self._latest_measurement(measured, now)

        if latest is None:
            return None

        quarter, measured_w, measured_time = latest
        value = nowcast_solar(
            measured_w,
            measured_time,
            quarter + timedelta(minutes=PREDICT_STEP_MINUTES) / 2,
            self.latitude,
            self.longitude,
        )

        return None if value is None else (quarter, value)

    def baseload_forecast(
        self, state: State, times: list[datetime], now: datetime
    ) -> list[float]:
        """Baseload forecast aligned to `times`, 0.0 where there is none (see
        align_predictions). The quarter hour running at `now` - the step the
        optimizer acts on - takes the measurement so far instead: load that is on
        right now is real. On real data, for the rest of that quarter hour over
        28 days, that cut the error from 105.6 to 95.0 W. It does expect more load
        that then isn't drawn (21.9 -> 47.1 W, an appliance run ending within the
        quarter hour), but that only delays heating until the next replan minutes
        later, while missing load that is on (83.8 -> 47.9 W) starts a run that
        imports from the grid for at least its minimum runtime. The lower of
        forecast and measurement was tried first: least phantom load (8.9 W), but
        the most missed load (92.5 W) - the costlier error.
        """

        forecast = self.align_predictions(
            state.predictions.baseload, times, default=math.nan
        )
        latest = self._latest_measurement(state.measurements.baseload, now)

        if latest is not None and times and pd.Timestamp(times[0]) == latest[0]:
            forecast[0] = latest[1]

        return [0.0 if math.isnan(value) else value for value in forecast]

    @staticmethod
    def _series_points(series: pd.Series) -> list[SeriesPoint]:
        return [
            SeriesPoint(time=pd.Timestamp(t).to_pydatetime(), value=float(v))
            for t, v in series.items()
        ]

    def update_prediction(self, name: str, series: pd.Series) -> None:
        state = self.load()

        points = [
            SeriesPoint(time=pd.to_datetime(str(time)), value=float(value))
            for time, value in series.items()
        ]

        if hasattr(state.predictions, name):
            setattr(state.predictions, name, points)
        else:
            raise ValueError(f"Unknown prediction: '{name}'")

        self.state_repository.save(state)

    def update_schedule(
        self,
        schedule: Sequence[int],
        temperatures: Sequence[float],
        power_w: Sequence[float],
        times: list[datetime],
    ) -> None:
        state = self.load()

        state.schedule.heat_pump.power = [
            SeriesPoint(time=t, value=float(on) * float(power))
            for t, on, power in zip(times, schedule, power_w, strict=False)
        ]

        state.schedule.heat_pump.boiler.temperatures = [
            SeriesPoint(time=t, value=float(val))
            for t, val in zip(times, temperatures, strict=False)
        ]

        self.state_repository.save(state)

    def _map(
        self,
        df: pd.DataFrame,
        existing: State | None = None,
        config: Config | None = None,
    ) -> State:
        state = State(updated=datetime.now(timezone.utc))

        if existing is not None:
            state.predictions = existing.predictions
            state.schedule = existing.schedule

        state.measurements.solar = self._parse_series(df, "pv_production")
        state.measurements.baseload = self._parse_series(df, "baseload")
        state.measurements.heat_pump.state = self._parse_series(df, "heat_pump_state")
        state.measurements.heat_pump.power = self._parse_series(df, "heat_pump_power")
        state.measurements.heat_pump.compressor_frequency = self._parse_series(
            df, "heat_pump_compressor_frequency"
        )
        state.measurements.heat_pump.boiler.top_temperature = self._parse_series(
            df, "boiler_top_temperature"
        )
        state.measurements.heat_pump.boiler.bottom_temperature = self._parse_series(
            df, "boiler_bottom_temperature"
        )
        state.measurements.heat_pump.boiler.ambient_temperature = self._parse_series(
            df, "boiler_ambient_temperature"
        )
        state.measurements.climate.temperature = self._parse_series(
            df, "climate_temperature"
        )
        state.measurements.climate.setpoint = self._parse_series(df, "climate_setpoint")

        for forecast_source in ["solcast", "open_meteo"]:
            source_obj = getattr(state.forecast, forecast_source)
            for attr, _ in source_obj.items():
                setattr(source_obj, attr, self._parse_series(df, attr))

        if config is not None:
            times = sorted(df["time"].dropna().unique())

            if times:
                state.schedule.heat_pump.boiler.target_temperature = (
                    self.resolve_schedule(
                        config.heat_pump.boiler.target_temperature, times
                    )
                )
                state.schedule.climate.target_temperature = self.resolve_schedule(
                    config.climate.target_temperature, times
                )

        return state

    def _parse_series(self, df: pd.DataFrame, column: str) -> list[SeriesPoint]:
        if column not in df.columns:
            return []

        return [
            SeriesPoint(
                time=row["time"],
                value=row[column],
            )
            for _, row in df[["time", column]].dropna().iterrows()
        ]

    def align_predictions(
        self,
        points: list[SeriesPoint],
        times: list[datetime],
        default: float = 0.0,
    ) -> list[float]:
        """Looks up a forecast's own points by exact timestamp against `times`
        (typically another forecast's horizon, e.g. solar's), falling back to
        `default` wherever no matching point exists - the forecast may not have
        been run, may cover a different horizon, or may not exist at all for
        this installation. `default=0.0` means "assume none of whatever this
        forecast predicts", the same assumption implicitly made before a given
        forecast existed at all, not a claim that the true value is zero.
        """

        by_time = {point.time: point.value for point in points}

        return [by_time.get(t, default) for t in times]

    def resolve_schedule(
        self,
        target: float | list[tuple[time, float]],
        times: list[datetime],
    ) -> list[SeriesPoint]:
        if isinstance(target, (float, int)):
            return [SeriesPoint(time=t, value=float(target)) for t in times]

        schedule = sorted(target)

        def get_val(t: time) -> float:
            past = [val for st, val in schedule if st <= t]
            return past[-1] if past else schedule[-1][1]

        return [
            SeriesPoint(time=t, value=get_val(to_local_time(t).time())) for t in times
        ]

    def _dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .attribute_series(
                "solcast",
                config.forecast.solcast,
                attributes=["p10", "p50", "p90"],
            )
            .attribute_series(
                "open_meteo",
                config.forecast.open_meteo,
                attributes=["temperature"],
            )
            .timeseries(
                "pv_production",
                config.solar,
                aggregation="mean",
                interval="15m",
            )
            .timeseries(
                "baseload",
                config.baseload,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "heat_pump_state",
                config.heat_pump.state,
                interval="15m",
                aggregation="first",
                fill="previous",
            )
            # Distinguishes the compressor from the resistive booster within a
            # run: the operating state stays "SWW" while the booster finishes
            # the tank, but the compressor is off at 0 Hz. Only reported on
            # change - including the change to 0 Hz - so the previous reading
            # is the current one (see BoilerThermalIdentifier.dataset for the
            # two fills that were tried and rejected here).
            .timeseries(
                "heat_pump_compressor_frequency",
                config.heat_pump.compressor_frequency,
                interval="15m",
                aggregation="last",
                fill="previous",
            )
            .timeseries(
                "heat_pump_power",
                config.heat_pump.power,
                interval="15m",
                aggregation="mean",
                fill=0,
            )
            .timeseries(
                "climate_temperature",
                config.climate.temperature,
                interval="5m",
                fill="previous",
                target_interval="15min",
            )
            .timeseries(
                "climate_setpoint",
                config.climate.setpoint,
                aggregation="mean",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "boiler_top_temperature",
                config.heat_pump.boiler.top_temperature,
                aggregation="mean",
                interval="5m",
                fill="previous",
                target_interval="15min",
                # A temperature is a state, not a flow: planning must start from
                # the most recent reading, and the quarter's average lags it
                # while the tank is heating. A plan made minutes after a run
                # ended started from a tank 1.5-2 K colder than it really was
                # and added a second run for the shortfall that followed.
                target_resample="last",
            )
            .timeseries(
                "boiler_bottom_temperature",
                config.heat_pump.boiler.bottom_temperature,
                aggregation="mean",
                interval="5m",
                fill="previous",
                target_interval="15min",
                # See boiler_top_temperature.
                target_resample="last",
            )
            .timeseries(
                "boiler_ambient_temperature",
                config.heat_pump.boiler.ambient_temperature,
                aggregation="mean",
                interval="5m",
                fill="previous",
                target_interval="15min",
                # See boiler_top_temperature.
                target_resample="last",
            )
            .build()
        )
