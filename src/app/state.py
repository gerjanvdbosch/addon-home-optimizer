from datetime import datetime, time, timezone
from typing import Sequence

import pandas as pd

from domain.time import to_local_time
from domain.types import (
    Config,
    SeriesPoint,
    State,
)
from features.dataset import DatasetBuilder, DatasetDefinition, DatasetLoader
from infrastructure.repositories import ConfigRepository, StateRepository


class StateManager:
    def __init__(
        self,
        loader: DatasetLoader,
        state_repository: StateRepository,
        config_repository: ConfigRepository,
    ):
        self.loader = loader
        self.state_repository = state_repository
        self.config_repository = config_repository

    def load(self) -> State:
        return self.state_repository.load()

    def update(self) -> None:
        now = datetime.now(timezone.utc)

        start = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        config = self.config_repository.load()

        df = self.loader.load(
            self._dataset(config),
            start,
            now,
        )

        state = self._map(df, self.load(), config=config)

        self.state_repository.save(state)

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
