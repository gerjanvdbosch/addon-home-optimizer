from datetime import UTC, datetime, timedelta

import pytest

from domain.dataset import AttributeSeriesDefinition, TimeSeriesDefinition
from domain.sensors import InfluxSensor, SensorAttributesReference, SensorReference
from infrastructure.loaders import (
    AttributeSeriesLoader,
    TimeSeriesLoader,
    time_weighted_means,
)

# A forecast covering whole UTC days, published twice: yesterday's covers the
# hours before 00:00 UTC that the local day already asks for.
YESTERDAY = {
    "time": [datetime(2026, 9, 17, 22 + i, tzinfo=UTC).isoformat() for i in range(2)],
    "temperature": [14.8, 14.4],
}
TODAY = {
    "time": [datetime(2026, 9, 18, i, tzinfo=UTC).isoformat() for i in range(3)],
    "temperature": [13.8, 13.2, 12.9],
}
# StateManager.update() loads from the previous local midnight, so yesterday's
# forecast is within the window it asks for.
LOAD_START = datetime(2026, 9, 16, 22, tzinfo=UTC)
LOCAL_MIDNIGHT = datetime(2026, 9, 17, 22, tzinfo=UTC)


class _Influx:
    """The forecast sensor, published yesterday at 21:00 UTC and today at 00:00."""

    def find(self, measurement, entity_id, field, before=None):
        return {"value": str(TODAY[field])}

    def find_series(self, measurement, entity_id, field, start, end, **kwargs):
        published = datetime(2026, 9, 17, 21, tzinfo=UTC)

        if not start <= published < end:
            return []

        return [{"time": published.isoformat(), "value": str(YESTERDAY[field])}]


class _Resolver:
    def resolve(self, sensor):
        return InfluxSensor(
            measurement="°C", entity_id="forecast", field=sensor.attribute
        )


def _load(influx) -> list[tuple[datetime, float]]:
    definition = AttributeSeriesDefinition(
        name="open_meteo",
        sensor=SensorAttributesReference(
            entity_id="forecast", attributes={"temperature": "temperature"}
        ),
        attributes=["temperature"],
    )

    frame = AttributeSeriesLoader(influx, _Resolver()).load(
        definition, LOAD_START, datetime(2026, 9, 19, tzinfo=UTC)
    )

    return list(zip(frame["time"], frame["temperature"], strict=True))


def test_the_forecast_before_it_covers_the_hours_the_newest_one_misses():
    """A forecast on UTC days starts at 00:00 UTC, so the hours between local
    midnight and that come from the forecast published before it."""

    assert _load(_Influx()) == [
        (LOCAL_MIDNIGHT, 14.8),
        (datetime(2026, 9, 17, 23, tzinfo=UTC), 14.4),
        (datetime(2026, 9, 18, 0, tzinfo=UTC), 13.8),
        (datetime(2026, 9, 18, 1, tzinfo=UTC), 13.2),
        (datetime(2026, 9, 18, 2, tzinfo=UTC), 12.9),
    ]


def test_the_newest_forecast_wins_where_both_cover_a_time():
    class _Overlapping(_Influx):
        def find_series(self, measurement, entity_id, field, start, end, **kwargs):
            published = datetime(2026, 9, 17, 21, tzinfo=UTC)
            overlapping = {
                "time": YESTERDAY["time"] + TODAY["time"][:1],
                "temperature": YESTERDAY["temperature"] + [99.0],
            }

            return [{"time": published.isoformat(), "value": str(overlapping[field])}]

    assert _load(_Overlapping())[2] == (datetime(2026, 9, 18, 0, tzinfo=UTC), 13.8)


def test_without_an_earlier_forecast_only_the_newest_one_is_used():
    class _Only(_Influx):
        def find_series(self, measurement, entity_id, field, start, end, **kwargs):
            return []

    assert _load(_Only())[0] == (datetime(2026, 9, 18, 0, tzinfo=UTC), 13.8)


def test_a_reading_counts_for_as_long_as_it_held():
    """A run at 2400 W stopping one minute into a quarter, its last readings
    many, the 0 after it one: a plain mean of the quarter's readings gives
    1600 W, the time it held 160 W. The reading in force at the start holds
    into the window; the running quarter counts up to the end."""

    start = datetime(2026, 10, 2, 11, 30, tzinfo=UTC)
    readings = [(-5.0, 2400.0), (14.0, 2400.0), (14.5, 2400.0), (16.0, 0.0)]
    points = [
        {"time": (start + timedelta(minutes=m)).isoformat(), "value": v}
        for m, v in readings
    ]

    means = time_weighted_means(points, start, start + timedelta(minutes=20), "15m")

    assert [p["time"] for p in means] == [
        start.isoformat(),
        (start + timedelta(minutes=15)).isoformat(),
    ]
    assert means[0]["value"] == pytest.approx(2400.0)
    assert means[1]["value"] == pytest.approx(2400.0 * 1 / 5)


def test_nothing_is_known_before_the_first_reading():
    start = datetime(2026, 10, 2, 11, 30, tzinfo=UTC)
    points = [{"time": (start + timedelta(minutes=20)).isoformat(), "value": 100.0}]

    means = time_weighted_means(points, start, start + timedelta(minutes=30), "15m")

    assert [p["value"] for p in means] == [None, pytest.approx(100.0)]
    assert time_weighted_means([], start, start + timedelta(minutes=30), "15m") == []


def test_a_state_held_from_before_the_window_fills_its_empty_start():
    """fill(previous) leaves the buckets before the window's first reading
    empty; a state stored only on change still holds its earlier value there."""

    start = datetime(2026, 10, 3, tzinfo=UTC)
    times = [(start + timedelta(minutes=15 * i)).isoformat() for i in range(4)]

    class _Shutter:
        def find(self, measurement, entity_id, field, before=None):
            assert before == start
            return {"time": (start - timedelta(days=60)).isoformat(), "value": 18.0}

        def find_series(self, measurement, entity_id, field, start, end, **kwargs):
            return [
                {"time": t, "value": v}
                for t, v in zip(times, [None, None, 75.0, 75.0], strict=True)
            ]

    frame = TimeSeriesLoader(_Shutter(), _Resolver()).load(
        TimeSeriesDefinition(
            name="shutter",
            sensor=SensorReference(entity_id="cover.x", attribute="current_position"),
            interval="15m",
            aggregation="last",
            fill="previous",
        ),
        start,
        start + timedelta(hours=1),
    )

    assert list(frame["shutter"]) == [18.0, 18.0, 75.0, 75.0]


def test_a_state_unchanged_through_the_window_holds_in_every_bucket():
    """A shutter that did not move in the window has no reading in it at all."""

    start = datetime(2026, 10, 3, tzinfo=UTC)

    class _IdleShutter:
        def find(self, measurement, entity_id, field, before=None):
            return {"time": (start - timedelta(days=60)).isoformat(), "value": 0.0}

        def find_series(self, *args, **kwargs):
            return []

    frame = TimeSeriesLoader(_IdleShutter(), _Resolver()).load(
        TimeSeriesDefinition(
            name="shutter",
            sensor=SensorReference(entity_id="cover.x", attribute="current_position"),
            interval="15m",
            aggregation="last",
            fill="previous",
        ),
        start,
        start + timedelta(hours=1),
    )

    assert list(frame["shutter"]) == [0.0] * 4
    assert frame["time"].iloc[-1] == start + timedelta(minutes=45)
