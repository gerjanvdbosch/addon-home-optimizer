"""Stating what data a model needs, and loading it.

The builder assembles a DatasetDefinition; the loader hands each definition to
whichever DataLoader supports it and joins the results as the definition says.
Which loader that is - and therefore which database is behind it - is not this
module's business: it holds a list of DataLoader, the protocol in
domain/dataset.py.
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from pandas._typing import MergeHow

from domain.dataset import (
    AttributeSeriesDefinition,
    AttributeTimeSeriesDefinition,
    DataDefinition,
    DataLoader,
    DatasetDefinition,
    JoinDefinition,
    TimeSeriesDefinition,
)
from domain.sensors import (
    Aggregation,
    FillMethod,
    SensorAttributesReference,
    SensorReference,
)


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
