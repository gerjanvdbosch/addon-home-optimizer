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
from features.dataset import DatasetBuilder, DatasetDefinition, DatasetLoader
from features.solar import (
    PREDICT_STEP_MINUTES,
    SolarBiasIdentifier,
    nowcast_solar,
    predict_solar,
    predict_solar_band,
)
from infrastructure.repositories import ConfigRepository, StateRepository


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

        self.state_repository.save(state)

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

    def _current_quarter_nowcast(
        self, measured: list[SeriesPoint], now: datetime
    ) -> tuple[pd.Timestamp, float] | None:
        """(start of the quarter hour running at `now`, PV output expected over
        it) from the latest measured quarter-hour mean (see nowcast_solar), or
        None without a measurement from this or the previous quarter hour - e.g.
        the PV sensor stops reporting at night."""

        if not measured:
            return None

        step = timedelta(minutes=PREDICT_STEP_MINUTES)
        quarter = pd.Timestamp(now).floor(f"{PREDICT_STEP_MINUTES}min")
        latest_start = pd.Timestamp(measured[-1].time)

        if not quarter - step <= latest_start <= quarter:
            return None

        # A running quarter hour's mean covers its start up to now.
        latest_end = min(latest_start + step, pd.Timestamp(now))
        value = nowcast_solar(
            measured[-1].value,
            latest_start + (latest_end - latest_start) / 2,
            quarter + step / 2,
            self.latitude,
            self.longitude,
        )

        return None if value is None else (quarter, value)

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
            )
            .timeseries(
                "boiler_bottom_temperature",
                config.heat_pump.boiler.bottom_temperature,
                aggregation="mean",
                interval="5m",
                fill="previous",
                target_interval="15min",
            )
            .timeseries(
                "boiler_ambient_temperature",
                config.heat_pump.boiler.ambient_temperature,
                aggregation="mean",
                interval="5m",
                fill="previous",
                target_interval="15min",
            )
            .build()
        )
