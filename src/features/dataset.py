import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, TypeVar

import pandas as pd
from pandas._typing import MergeHow
from pydantic import BaseModel

from domain.time import parse_datetime
from domain.types import (
    SensorAttributesReference,
    SensorReference,
)
from infrastructure.influx import (
    Aggregation,
    FillMethod,
    InfluxDatabase,
    InfluxSensorResolver,
)


class DataDefinition(BaseModel):
    name: str
    time_attribute: str = "time"
    target_interval: str | None = None
    target_closed: Literal["right", "left"] | None = None
    target_label: Literal["right", "left"] | None = None
    target_resample: str | None = None
    target_shift: bool | list[str] = False


class TimeSeriesDefinition(DataDefinition):
    sensor: SensorReference
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod | int | float = "none"


class AttributeSeriesDefinition(DataDefinition):
    sensor: SensorAttributesReference
    attributes: list[str]


class AttributeTimeSeriesDefinition(AttributeSeriesDefinition):
    aggregation: Aggregation | None = None
    interval: str = "1min"
    fill: FillMethod | int | float = "none"


@dataclass(frozen=True)
class JoinDefinition:
    left: str
    right: str
    left_on: tuple[str, ...]
    right_on: tuple[str, ...]
    how: MergeHow = "left"


class DatasetDefinition(BaseModel):
    definitions: list[DataDefinition] = []
    joins: list[JoinDefinition]


D = TypeVar("D", bound=DataDefinition, contravariant=True)


class DataLoader(Protocol[D]):
    def supports(self, definition: DataDefinition) -> bool: ...

    def load(self, definition: D, start: datetime, end: datetime) -> pd.DataFrame: ...


def resample_dataframe(
    df: pd.DataFrame,
    time_col: str,
    interval: str | None,
    closed: Literal["right", "left"] | None = None,
    label: Literal["right", "left"] | None = None,
    resample_method: str | None = None,
    shift: bool | list[str] = False,
) -> pd.DataFrame:
    if interval is None or df.empty:
        return df

    val_cols = [col for col in df.columns if col != time_col]
    if not val_cols:
        return df[[time_col]]

    df_sorted = df.sort_values(time_col)

    resampler = df_sorted.set_index(time_col)[val_cols].resample(
        interval,
        label=label,
        closed=closed,
    )

    method = resample_method or "mean"
    resampled = getattr(resampler, method)().reset_index()

    if shift is True:
        resampled[time_col] = resampled[time_col] - pd.to_timedelta(interval)
    elif isinstance(shift, (list, tuple, set)):
        for shift_col in shift:
            if shift_col in resampled.columns:
                resampled[shift_col] = resampled[shift_col].shift(-1)

    return resampled


class DatasetLoader:
    def __init__(self, loaders: list[DataLoader]):
        self.loaders = loaders

    def load(
        self,
        dataset: DatasetDefinition,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        frames: dict[str, pd.DataFrame] = {}

        for definition in dataset.definitions:
            loader = self._find(definition)

            df = loader.load(
                definition=definition,
                start=start,
                end=end,
            )

            for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
                df[col] = df[col].dt.as_unit("ns")

            frames[definition.name] = df

        return self._merge(
            frames=frames,
            joins=dataset.joins,
        )

    def _find(self, definition: DataDefinition) -> DataLoader:
        for loader in self.loaders:
            if loader.supports(definition):
                return loader

        raise ValueError(f"No loader found for sensor: {definition.name}")

    def _merge(
        self,
        frames: dict[str, pd.DataFrame],
        joins: list[JoinDefinition],
    ) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()

        for name, frame in frames.items():
            if "time" not in frame.columns:
                raise ValueError(
                    f"Dataset '{name}' has no 'time' column. "
                    f"Columns: {frame.columns.tolist()}"
                )

        if not joins:
            frames_iter = iter(frames.values())

            result = next(frames_iter).copy()

            for frame in frames_iter:
                result = result.merge(
                    frame,
                    on="time",
                    how="outer",
                )

            return result.reset_index(drop=True)

        result = frames[joins[0].left].copy()
        used = {joins[0].left}

        for join in joins:
            if join.left not in used:
                raise ValueError(
                    f"Cannot join '{join.left}': it is not part of the current dataset"
                )

            if len(join.left_on) != len(join.right_on):
                raise ValueError(
                    "left_on and right_on must contain the same number of columns"
                )

            right = frames[join.right].copy()

            result = result.merge(
                right,
                left_on=list(join.left_on),
                right_on=list(join.right_on),
                how=join.how,
                suffixes=("", f"_{join.right}"),
            )

            for left_key, right_key in zip(
                join.left_on,
                join.right_on,
                strict=True,
            ):
                if left_key != right_key:
                    result = result.drop(
                        columns=f"{right_key}_{join.right}",
                        errors="ignore",
                    )

            used.add(join.right)

        return result.reset_index(drop=True)


class DatasetBuilder:
    def __init__(self):
        self._definitions: list[DataDefinition] = []
        self._joins: list[JoinDefinition] = []

    def timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation | None = None,
        interval: str = "1m",
        fill: FillMethod | int | float = "none",
        target_interval: str | None = None,
        target_closed: Literal["right", "left"] | None = None,
        target_label: Literal["right", "left"] | None = None,
        target_resample: str | None = None,
        target_shift: bool | list[str] = False,
    ) -> "DatasetBuilder":
        """
        Load a regular time series.

        Each row represents a value at a specific point in time. The `time`
        column represents the timestamp at which the value applies.

        Example:
            time   | value
            -------|------
            10:00  | 300
            10:15  | 400
            10:30  | 500
        """

        self._definitions.append(
            TimeSeriesDefinition(
                name=name,
                sensor=sensor,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
                target_interval=target_interval,
                target_closed=target_closed,
                target_label=target_label,
                target_resample=target_resample,
                target_shift=target_shift,
            )
        )

        return self

    def attribute_series(
        self,
        name: str,
        sensor: SensorAttributesReference,
        attributes: list,
        target_interval: str | None = None,
        target_closed: Literal["right", "left"] | None = None,
        target_label: Literal["right", "left"] | None = None,
        target_resample: str | None = None,
        target_shift: bool | list[str] = False,
    ) -> "DatasetBuilder":
        """
        Load a single attribute series.

        Each row represents a value at a specific point in time. The `time`
        column represents the timestamp at which the value applies.

        Example:
            time   | p50
            -------|----
            10:00  | 300
            10:30  | 400
            11:00  | 500
        """

        self._definitions.append(
            AttributeSeriesDefinition(
                name=name,
                sensor=sensor,
                attributes=attributes,
                target_interval=target_interval,
                target_closed=target_closed,
                target_label=target_label,
                target_resample=target_resample,
                target_shift=target_shift,
            )
        )
        return self

    def attribute_timeseries(
        self,
        name: str,
        sensor: SensorAttributesReference,
        attributes: list,
        aggregation: Aggregation | None = None,
        interval: str = "1m",
        fill: FillMethod | int | float = "none",
        target_interval: str | None = None,
        target_closed: Literal["right", "left"] | None = None,
        target_label: Literal["right", "left"] | None = None,
        target_resample: str | None = None,
        target_shift: bool | list[str] = False,
    ) -> "DatasetBuilder":
        """
        Load a time series of attribute snapshots.

        Each row represents a forecast or snapshot created at `time` for a
        specific `target_time`. The `time` column indicates when the snapshot
        was created, while `target_time` indicates when the value applies.

        Example:
            time  | target_time  | p50 | p90
            ------|--------------|-----|----
            09:30 | 10:00        | 300 | 350
            10:00 | 10:30        | 400 | 460
            10:00 | 11:00        | 500 | 580
            10:00 | 11:30        | 320 | 400
            10:30 | 11:00        | 450 | 490
        """

        self._definitions.append(
            AttributeTimeSeriesDefinition(
                name=name,
                sensor=sensor,
                attributes=attributes,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
                target_interval=target_interval,
                target_closed=target_closed,
                target_label=target_label,
                target_resample=target_resample,
                target_shift=target_shift,
            )
        )

        return self

    def join(
        self,
        left: str,
        right: str,
        on: tuple[str, ...] | None = None,
        left_on: tuple[str, ...] | None = None,
        right_on: tuple[str, ...] | None = None,
        how: MergeHow = "left",
    ) -> "DatasetBuilder":
        if on is not None:
            left_on = right_on = on

        if left_on is None or right_on is None:
            raise ValueError(
                "Either 'on' or both 'left_on' and 'right_on' must be provided"
            )

        if len(left_on) != len(right_on):
            raise ValueError("left_on and right_on must have the same length")

        self._joins.append(
            JoinDefinition(
                left=left,
                right=right,
                left_on=left_on,
                right_on=right_on,
                how=how,
            )
        )

        return self

    def build(self) -> DatasetDefinition:
        names = {definition.name for definition in self._definitions}

        for join in self._joins:
            if join.left not in names:
                raise ValueError(f"Unknown join source: {join.left}")

            if join.right not in names:
                raise ValueError(f"Unknown join source: {join.right}")

        return DatasetDefinition(
            definitions=self._definitions,
            joins=self._joins,
        )


class TimeSeriesLoader(DataLoader):
    def __init__(self, influx: InfluxDatabase, resolver: InfluxSensorResolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition) -> bool:
        return type(definition) is TimeSeriesDefinition

    def load(
        self, definition: TimeSeriesDefinition, start: datetime, end: datetime
    ) -> pd.DataFrame:
        sensor = self.resolver.resolve(definition.sensor)

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
            aggregation=definition.aggregation,
            interval=definition.interval,
            fill=definition.fill,
        )

        if not points and definition.fill == "previous":
            last_point = self.influx.find(
                measurement=sensor.measurement,
                entity_id=sensor.entity_id,
                field=sensor.field,
            )
            if last_point and parse_datetime(last_point["time"]) < start:
                points = [{"time": start.isoformat(), "value": last_point["value"]}]

        rows = [
            {
                "time": parse_datetime(point["time"]),
                definition.name: point["value"],
            }
            for point in points
            if point["value"] is not None
        ]

        if not rows:
            return pd.DataFrame(
                {
                    "time": pd.to_datetime([], utc=True),
                    definition.name: pd.Series(),
                }
            )

        df = pd.DataFrame(rows)

        return resample_dataframe(
            df=df,
            time_col="time",
            interval=definition.target_interval,
            closed=definition.target_closed,
            label=definition.target_label,
            resample_method=definition.target_resample,
            shift=definition.target_shift,
        )


class AttributeSeriesLoader(DataLoader):
    def __init__(
        self,
        influx: InfluxDatabase,
        resolver: InfluxSensorResolver,
    ):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition: Any) -> bool:
        return type(definition) is AttributeSeriesDefinition

    def load(
        self,
        definition: AttributeSeriesDefinition,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        sensors = self.resolver.resolve_attributes(definition.sensor)

        time_sensor = self.resolver.resolve(
            SensorReference(
                entity_id=definition.sensor.entity_id,
                attribute=definition.time_attribute,
            )
        )

        time_point = self.influx.find(
            measurement=time_sensor.measurement,
            entity_id=time_sensor.entity_id,
            field=time_sensor.field,
        )

        if time_point is None or time_point.get("value") is None:
            return pd.DataFrame(
                columns=["time", *definition.attributes],
            )

        times = ast.literal_eval(str(time_point["value"]))

        frame = pd.DataFrame(
            {
                "time": [parse_datetime(str(value)) for value in times],
            }
        )

        for name in definition.attributes:
            sensor = sensors.get(name)

            if sensor is None:
                continue

            point = self.influx.find(
                measurement=sensor.measurement,
                entity_id=sensor.entity_id,
                field=sensor.field,
            )

            if point is None or point.get("value") is None:
                continue

            values = ast.literal_eval(str(point["value"]))

            if len(values) != len(times):
                raise ValueError(
                    f"Attribute '{name}' has {len(values)} values, "
                    f"expected {len(times)}"
                )

            frame[name] = [
                float(value) if value is not None else None for value in values
            ]

        if definition.target_interval is not None and not frame.empty:
            frame = frame.set_index("time")

            resampler = frame.resample(
                definition.target_interval,
                label=definition.target_label,
                closed=definition.target_closed,
            )

            method = definition.target_resample or "mean"
            frame = getattr(resampler, method)()

            if definition.target_shift is True:
                frame.index = frame.index - pd.to_timedelta(definition.target_interval)
            elif isinstance(definition.target_shift, (list, tuple, set)):
                for shift_col in definition.target_shift:
                    if shift_col in frame.columns:
                        frame[shift_col] = frame[shift_col].shift(-1)

            frame = frame.reset_index()

        return frame


class AttributeTimeSeriesLoader(DataLoader):
    def __init__(
        self,
        influx: InfluxDatabase,
        resolver: InfluxSensorResolver,
    ):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition: Any) -> bool:
        return type(definition) is AttributeTimeSeriesDefinition

    def load(
        self,
        definition: AttributeTimeSeriesDefinition,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        sensors = self.resolver.resolve_attributes(definition.sensor)

        time_sensor = self.resolver.resolve(
            SensorReference(
                entity_id=definition.sensor.entity_id,
                attribute=definition.time_attribute,
            )
        )

        points = self.influx.find_series(
            measurement=time_sensor.measurement,
            entity_id=time_sensor.entity_id,
            field=time_sensor.field,
            start=start,
            end=end,
            aggregation=definition.aggregation,
            interval=definition.interval,
            fill=definition.fill,
        )

        snapshots: dict[datetime, dict[str, list]] = defaultdict(dict)

        for point in points:
            value = point.get("value")

            if value is None:
                continue

            snapshot_time = parse_datetime(point["time"])
            values = ast.literal_eval(str(value))

            snapshots[snapshot_time][definition.time_attribute] = values

        for name, sensor in sensors.items():
            points = self.influx.find_series(
                measurement=sensor.measurement,
                entity_id=sensor.entity_id,
                field=sensor.field,
                start=start,
                end=end,
                aggregation=definition.aggregation,
                interval=definition.interval,
                fill=definition.fill,
            )

            for point in points:
                value = point.get("value")

                if value is None:
                    continue

                snapshot_time = parse_datetime(point["time"])
                values = ast.literal_eval(str(value))

                snapshots[snapshot_time][name] = values

        frames: list[pd.DataFrame] = []

        for snapshot_time, values in snapshots.items():
            target_times = values.get(definition.time_attribute)

            if target_times is None:
                continue

            frame = pd.DataFrame(
                {
                    "time": snapshot_time,
                    "target_time": [
                        parse_datetime(str(value)) for value in target_times
                    ],
                }
            )

            for name in definition.attributes:
                attribute_values = values.get(name)

                if attribute_values is None:
                    continue

                if len(attribute_values) != len(target_times):
                    raise ValueError(
                        f"Attribute '{name}' has "
                        f"{len(attribute_values)} values, expected "
                        f"{len(target_times)} for forecast "
                        f"{snapshot_time}"
                    )

                frame[name] = [
                    float(value) if value is not None else None
                    for value in attribute_values
                ]

            frames.append(frame)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "time",
                    "target_time",
                    *definition.attributes,
                ]
            )

        df = pd.concat(frames, ignore_index=True)

        if definition.target_interval is None or df.empty:
            return df

        resampled_frames: list[pd.DataFrame] = []

        for snapshot_time, group in df.groupby("time", sort=False):
            resampled = resample_dataframe(
                df=group.drop(columns=["time"]),
                time_col="target_time",
                interval=definition.target_interval,
                closed=definition.target_closed,
                label=definition.target_label,
                resample_method=definition.target_resample,
                shift=definition.target_shift,
            )
            resampled.insert(0, "time", snapshot_time)
            resampled_frames.append(resampled)

        return pd.concat(resampled_frames, ignore_index=True)
