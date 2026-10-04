"""InfluxDB adapters: they fetch a definition's data and reshape it.

One per definition type. Most of what they do is translation rather than
fetching, because this source stores things in its own shapes - a Home
Assistant forecast attribute arrives as a Python literal in a string field,
and a history of such attributes as loose rows that have to be regrouped into
snapshots. That knowledge belongs to the source, which is why it lives here
and not with the models: they say what they need in domain/dataset.py terms
and never see any of this.
"""

import ast
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from domain.dataset import (
    AttributeSeriesDefinition,
    AttributeTimeSeriesDefinition,
    DataLoader,
    TimeSeriesDefinition,
)
from domain.sensors import InfluxSensor, SensorReference
from domain.time import parse_datetime
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver

logger = logging.getLogger(__name__)


def resolve_attributes(
    resolver: InfluxSensorResolver,
    definition: AttributeSeriesDefinition | AttributeTimeSeriesDefinition,
) -> dict[str, InfluxSensor]:
    """The attributes a definition asks for, as stored. One the entity has
    never published is left out, with a warning, rather than failing the whole
    load: its column is then simply absent, as it is from any snapshot that
    lacks it - so a newly configured attribute (say Open-Meteo's dew point)
    does not break every model until its first publication."""

    attributes = dict(definition.sensor.attributes.items())
    sensors = {}

    for name in definition.attributes:
        try:
            sensors[name] = resolver.resolve(
                SensorReference(
                    entity_id=definition.sensor.entity_id,
                    attribute=attributes[name],
                )
            )
        except ValueError as error:
            logger.warning("Attribute %s left out: %s", name, error)

    return sensors


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


def time_weighted_means(
    points: list[dict[str, Any]], start: datetime, end: datetime, interval: str
) -> list[dict[str, Any]]:
    """The mean over time per interval from `start` to `end`, each reading
    holding until the next (see Aggregation's "time_mean"); the running
    interval up to `end`. None before the first reading. No limit on how long a
    reading holds: a sensor that does not change is not stored again, so a
    long silence is a value that held, as Home Assistant's own history draws
    it."""

    readings = pd.Series(
        pd.to_numeric([p["value"] for p in points], errors="coerce"),
        index=pd.DatetimeIndex([parse_datetime(p["time"]) for p in points]),
    ).dropna()
    readings = readings[~readings.index.duplicated(keep="last")].sort_index()

    if readings.empty:
        return []

    step = pd.Timedelta(interval)
    edges = pd.date_range(pd.Timestamp(start).floor(step), end, freq=step)
    edges = edges.append(pd.DatetimeIndex([pd.Timestamp(end)])).unique()
    edge_s = (edges - edges[0]).total_seconds().to_numpy()
    at = (readings.index - edges[0]).total_seconds().to_numpy()
    values = readings.to_numpy(dtype=float)

    # The signal's integral from the first reading: at each reading, and from
    # there at each edge. Time before the first reading has no value.
    integral = np.concatenate([[0.0], np.cumsum(values[:-1] * np.diff(at))])
    held = np.maximum(np.searchsorted(at, edge_s, side="right") - 1, 0)
    at_edges = np.where(
        edge_s >= at[0], integral[held] + values[held] * (edge_s - at[held]), 0.0
    )
    covered = np.diff(np.maximum(edge_s, at[0]))
    means = np.diff(at_edges) / np.where(covered > 0, covered, np.nan)

    return [
        {"time": time.isoformat(), "value": None if np.isnan(mean) else mean}
        for time, mean in zip(edges[:-1], means, strict=True)
    ]


class TimeSeriesLoader(DataLoader):
    """The series of one sensor over time."""

    def __init__(self, influx: InfluxDatabase, resolver: InfluxSensorResolver):
        self.influx = influx
        self.resolver = resolver

    def supports(self, definition) -> bool:
        return type(definition) is TimeSeriesDefinition

    def load(
        self, definition: TimeSeriesDefinition, start: datetime, end: datetime
    ) -> pd.DataFrame:
        sensor = self.resolver.resolve(definition.sensor)
        time_mean = definition.aggregation == "time_mean"

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
            aggregation=None if time_mean else definition.aggregation,
            interval=definition.interval,
            fill=definition.fill,
        )

        if time_mean:
            # The reading in force at the start holds into the window.
            before = self.influx.find(
                measurement=sensor.measurement,
                entity_id=sensor.entity_id,
                field=sensor.field,
                before=start,
            )
            points = time_weighted_means(
                ([before] if before else []) + points, start, end, definition.interval
            )

        # fill(previous) only carries readings within the window: the buckets
        # before its first one stay empty. A state sensor is stored only when it
        # changes, so there the reading in force at the start holds - a shutter
        # shut for months read as missing, and so as fully open.
        if (
            definition.fill == "previous"
            and not time_mean
            and (not points or points[0]["value"] is None)
        ):
            before = self.influx.find(
                measurement=sensor.measurement,
                entity_id=sensor.entity_id,
                field=sensor.field,
                before=start,
            )

            if before is not None:
                # No reading in the window at all: every bucket, as fill(previous)
                # gives once there is one.
                if not points:
                    step = pd.Timedelta(definition.interval)
                    points = [
                        {"time": time.isoformat(), "value": None}
                        for time in pd.date_range(
                            pd.Timestamp(start).floor(step),
                            end,
                            freq=step,
                            inclusive="left",
                        )
                    ]

                for point in points:
                    if point["value"] is not None:
                        break

                    point["value"] = before["value"]

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
    """The attributes an entity publishes right now - one forecast, as it
    stands. All of a forecast's attributes share a single reading, so they
    share its timestamp.
    """

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
        # Only the attributes asked for: each is a query of its own, and the
        # state asks for a few of Open-Meteo's.
        sensors = resolve_attributes(self.resolver, definition)

        time_sensor = self.resolver.resolve(
            SensorReference(
                entity_id=definition.sensor.entity_id,
                attribute=definition.time_attribute,
            )
        )

        published = self._published(definition, sensors, time_sensor)
        frame = self._frame(definition, published)

        if frame.empty:
            return frame

        # A forecast covering whole UTC days starts at 00:00 UTC, while a
        # request starting at local midnight asks for the hours before that
        # (confirmed on real data: Open-Meteo left the first two hours of the
        # local day empty). Only a forecast published before those hours covers
        # them, so the newest of those fills them in; where both cover a time,
        # the newer forecast wins.
        if frame["time"].iloc[0] > start:
            earlier = self._frame(
                definition,
                self._published(
                    definition, sensors, time_sensor, start, frame["time"].iloc[0]
                ),
            )

            if not earlier.empty:
                frame = (
                    pd.concat([earlier, frame], ignore_index=True)
                    .drop_duplicates(subset="time", keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                )

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

    def _published(
        self,
        definition: AttributeSeriesDefinition,
        sensors: dict[str, InfluxSensor],
        time_sensor: InfluxSensor,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, list]:
        """One published forecast as {attribute: values}: the newest one there
        is, or the newest published within [start, end). All of a forecast's
        attributes are stored in the same reading, so they share its time."""

        published: dict[str, list] = {}

        for name, sensor in {definition.time_attribute: time_sensor, **sensors}.items():
            if start is None or end is None:
                point = self.influx.find(
                    measurement=sensor.measurement,
                    entity_id=sensor.entity_id,
                    field=sensor.field,
                )
            else:
                points = self.influx.find_series(
                    measurement=sensor.measurement,
                    entity_id=sensor.entity_id,
                    field=sensor.field,
                    start=start,
                    end=end,
                )
                point = points[-1] if points else None

            if point is None or point.get("value") is None:
                continue

            published[name] = ast.literal_eval(str(point["value"]))

        return published

    def _frame(
        self,
        definition: AttributeSeriesDefinition,
        published: dict[str, list],
    ) -> pd.DataFrame:
        times = published.get(definition.time_attribute)

        if times is None:
            return pd.DataFrame(columns=["time", *definition.attributes])

        frame = pd.DataFrame({"time": [parse_datetime(str(value)) for value in times]})

        for name in definition.attributes:
            values = published.get(name)

            if values is None:
                continue

            if len(values) != len(times):
                raise ValueError(
                    f"Attribute '{name}' has {len(values)} values, "
                    f"expected {len(times)}"
                )

            frame[name] = [
                float(value) if value is not None else None for value in values
            ]

        return frame


class AttributeTimeSeriesLoader(DataLoader):
    """Every forecast an entity has published, not just the current one: the
    history of its attributes, regrouped into one snapshot per publication.
    That is what lets a model be scored against what was known at the time.
    """

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
        # Only the attributes asked for, as above: resolving all of them made
        # one Open-Meteo attribute missing from the database fail every load.
        sensors = resolve_attributes(self.resolver, definition)

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
