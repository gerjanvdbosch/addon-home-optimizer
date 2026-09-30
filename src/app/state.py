import logging
import math
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from domain.config import Config
from domain.dataset import DatasetDefinition
from domain.physics import tank_stratification_k
from domain.state import SeriesPoint, State
from domain.time import local_day_start, to_local_time
from features.dataset import DatasetBuilder, DatasetLoader
from features.solar import (
    NOWCAST_SPREAD_WINDOW,
    PREDICT_STEP_MINUTES,
    SolarBiasIdentifier,
    nowcast_solar,
    nowcast_weight,
    predict_solar,
    predict_solar_band,
)
from infrastructure.repositories import ConfigRepository, StateRepository

logger = logging.getLogger(__name__)


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

        recent_solar = self.loader.load(
            DatasetBuilder()
            .timeseries("pv_production", config.solar, aggregation="mean")
            .build(),
            now - NOWCAST_SPREAD_WINDOW,
            now,
        )
        self._predict_solar(state, now, recent_solar.set_index("time")["pv_production"])

        self.state_repository.save(state)

    def _predict_solar(
        self, state: State, now: datetime, recent_solar: pd.Series
    ) -> None:
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

        p50 = predict_solar(
            identifier.model,
            self._future_series(forecast.p50, now),
            self.latitude,
            self.longitude,
        )
        p10 = p90 = None

        if forecast.p10 and forecast.p90:
            p10, p90 = predict_solar_band(
                identifier.model,
                self._future_series(forecast.p10, now),
                self._future_series(forecast.p90, now),
                now,
            )

        # Without a calibrated band there is no forecast variance to weigh the
        # measurement against, so it stands alone.
        weight = 1.0

        if (
            nowcast is not None
            and p10 is not None
            and p90 is not None
            and nowcast[0] in p10.index
            and nowcast[0] in p90.index
        ):
            weight = nowcast_weight(recent_solar, p10[nowcast[0]], p90[nowcast[0]])

        # The running quarter hour - the step the optimizer acts on - combines
        # the measurement-based estimate with the forecast (see
        # nowcast_weight). The measurement moves the whole band, not its width:
        # it says where the output is now, not how much the sky will still
        # change within the quarter hour. Blending each scenario with the one
        # measurement instead shrank the band by (1 - weight): on 24 days the
        # calibration had not seen, 73% of the rest of the quarter hour fell
        # outside p10-p90 (median width 49 W) - the plan took a run's first
        # step as all but certain sun. Shifted, 34% did (220 W), and the p10/p90
        # pinball loss fell from 79.9 to 64.4 W. A run's expected grid import
        # over its minimum runtime was already close to what it really drew,
        # and stayed so (within 0.01 kWh either way).
        shift = 0.0

        if nowcast is not None and nowcast[0] in p50.index:
            shift = weight * (nowcast[1] - p50[nowcast[0]])

        def from_now(series: pd.Series) -> pd.Series:
            # Only quarter hours that haven't ended.
            series = series[series.index > now - step]

            if shift and nowcast[0] in series.index:
                series = series.copy()
                series[nowcast[0]] = max(series[nowcast[0]] + shift, 0.0)

            return series

        state.predictions.solar = self._series_points(from_now(p50))

        if p10 is not None and p90 is not None:
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
        heat_w: Sequence[float] = (),
    ) -> None:
        state = self.load()

        state.schedule.heat_pump.power = [
            SeriesPoint(time=t, value=float(on) * float(power))
            for t, on, power in zip(times, schedule, power_w, strict=False)
        ]
        state.schedule.heat_pump.heat = [
            SeriesPoint(time=t, value=float(heat))
            for t, heat in zip(times, heat_w, strict=False)
        ]

        state.schedule.heat_pump.boiler.temperatures = [
            SeriesPoint(time=t, value=float(val))
            for t, val in zip(times, temperatures, strict=False)
        ]

        self.state_repository.save(state)

    def update_supply_setpoint(self, point: SeriesPoint | None) -> None:
        state = self.load()
        state.schedule.building.supply_setpoint = point
        self.state_repository.save(state)

    def update_zone(self, temperature: pd.Series, thermal_mass: pd.Series) -> None:
        """The zone as the optimization job estimates it (see
        Optimization._zone): its measured temperature, and the
        thermal mass no sensor measures."""

        state = self.load()

        state.measurements.building.zone_temperature = self._series_points(temperature)
        state.predictions.thermal_mass = self._series_points(thermal_mass)

        self.state_repository.save(state)

    def update_building_schedule(
        self,
        heat_w: Sequence[float],
        temperatures: Sequence[float],
        times: list[datetime],
        supply_c: Sequence[float] = (),
        power_w: Sequence[float] = (),
    ) -> None:
        """The zone's part of the plan, not acted on yet. Its heat and power go
        into the heat pump's plan beside the tank's (see update_schedule, which
        rewrote those this same run): one plan, one machine serving both."""

        state = self.load()
        heat_pump = state.schedule.heat_pump

        def plus(points: list[SeriesPoint], values: Sequence[float]) -> list:
            extra = {t: float(v) for t, v in zip(times, values, strict=False)}
            return [
                SeriesPoint(time=p.time, value=p.value + extra.get(p.time, 0.0))
                for p in points
            ]

        heat_pump.power = plus(heat_pump.power, power_w)
        heat_pump.heat = plus(heat_pump.heat, [abs(h) for h in heat_w])
        state.schedule.building.temperatures = [
            SeriesPoint(time=t, value=float(value))
            for t, value in zip(times, temperatures, strict=False)
        ]
        # NaN where the floor is not run; left out, so no line is drawn there.
        state.schedule.building.supply = [
            SeriesPoint(time=t, value=float(value))
            for t, value in zip(times, supply_c, strict=False)
            if not math.isnan(value)
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
            # Measured, but by the optimization job, which averages the rooms
            # as the building model does (see update_zone).
            state.measurements.building.zone_temperature = (
                existing.measurements.building.zone_temperature
            )

        state.measurements.solar = self._parse_series(df, "pv_production")
        state.measurements.baseload = self._parse_series(df, "baseload")
        state.measurements.heat_pump.state = self._parse_series(df, "heat_pump_state")
        state.measurements.heat_pump.return_temperature = self._parse_series(
            df, "heat_pump_return_temperature"
        )
        state.measurements.heat_pump.flow = self._parse_series(df, "heat_pump_flow")
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

        if config is not None and {
            "heat_pump_state",
            "boiler_top_temperature",
            "boiler_bottom_temperature",
        } <= set(df.columns):
            # Mixed where a quarter hour began on DHW (heat_pump_state is each
            # quarter's first): a run shorter than a quarter hour that begins
            # and ends inside one is not seen, and its tank read as at rest.
            df = df.assign(
                boiler_stratification=tank_stratification_k(
                    pd.to_numeric(df["boiler_top_temperature"], errors="coerce"),
                    pd.to_numeric(df["boiler_bottom_temperature"], errors="coerce"),
                    df["heat_pump_state"] == config.heat_pump.states.dhw,
                )
            )
            state.measurements.heat_pump.boiler.stratification = self._parse_series(
                df, "boiler_stratification"
            )
        state.measurements.building.temperature = self._parse_series(
            df, "thermostat_temperature"
        )
        state.measurements.building.setpoint = self._parse_series(
            df, "thermostat_setpoint"
        )
        state.measurements.heat_pump.mode = self._parse_series(df, "heat_pump_mode")
        dew_points = [
            column for column in df.columns if column.startswith("dew_point_")
        ]

        if dew_points:
            state.measurements.building.dew_point = self._parse_series(
                df.assign(dew_point=df[dew_points].max(axis=1, skipna=True)),
                "dew_point",
            )

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
                state.schedule.building.target_temperature = self.resolve_schedule(
                    config.building.target_temperature, times
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
        builder = (
            DatasetBuilder()
            .attribute_series(
                "solcast",
                config.forecast.solcast,
                attributes=["p10", "p50", "p90"],
            )
            .attribute_series(
                "open_meteo",
                config.forecast.open_meteo,
                # Humidity for the indoor dew point's forecast (see
                # features.dew_point), either as stored.
                attributes=["temperature", "dew_point", "relative_humidity"],
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
            # What the floor sends back and how fast, which a cooling run's
            # supply setpoint follows (see Optimization.zone_setpoint_c). The
            # return a state, so its last reading; flow rate-like, stopping
            # when the pump does.
            .timeseries(
                "heat_pump_return_temperature",
                config.heat_pump.return_temperature,
                interval="15m",
                aggregation="last",
                fill="previous",
            )
            .timeseries(
                "heat_pump_flow",
                config.heat_pump.flow,
                interval="15m",
                aggregation="mean",
                fill=0,
            )
            .timeseries(
                "thermostat_temperature",
                config.building.thermostat.temperature,
                aggregation="last",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "thermostat_setpoint",
                config.building.thermostat.setpoint,
                aggregation="last",
                interval="15m",
                fill="previous",
            )
            .timeseries(
                "boiler_top_temperature",
                config.heat_pump.boiler.top_temperature,
                aggregation="last",
                interval="15m",
                fill="previous",
                # A temperature is a state, not a flow: planning must start from
                # the most recent reading, and the quarter's average lags it
                # while the tank is heating. This tank climbs about 0.6 K a
                # minute on a run - it read 47.50 at 10:54:27 while the state
                # said 46.75, exactly the mean of 10:50-10:55, so a plan made
                # minutes after a run ended started from a tank 1.5-2 K colder
                # than it really was and added a second run for the shortfall.
                #
                # Which is why nothing here averages, at either level. Given
                # that, the bucket width no longer matters: last-of-lasts is
                # the same reading whatever it is grouped by first (checked
                # against minute buckets on irregular readings, every minute of
                # a twelve hour window, identical throughout), so this asks for
                # the quarter hours it actually needs rather than fifteen times
                # the rows.
            )
            .timeseries(
                "boiler_bottom_temperature",
                config.heat_pump.boiler.bottom_temperature,
                aggregation="last",
                interval="15m",
                fill="previous",
                # See boiler_top_temperature.
            )
            .timeseries(
                "boiler_ambient_temperature",
                config.heat_pump.boiler.ambient_temperature,
                aggregation="last",
                interval="15m",
                fill="previous",
                # See boiler_top_temperature - a state, so the last reading.
            )
        )

        # Both states, so the last reading: a mode only reports when it is
        # changed, and a dew point changes slowly.
        if config.heat_pump.mode is not None:
            builder = builder.timeseries(
                "heat_pump_mode",
                config.heat_pump.mode,
                aggregation="last",
                interval="15m",
                fill="previous",
            )

        for i, sensor in enumerate(config.building.dew_points):
            builder = builder.timeseries(
                f"dew_point_{i}",
                sensor,
                aggregation="last",
                interval="15m",
                fill="previous",
            )

        return builder.build()
