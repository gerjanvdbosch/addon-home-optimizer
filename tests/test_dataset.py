from datetime import UTC, datetime

from domain.dataset import AttributeSeriesDefinition
from domain.sensors import InfluxSensor, SensorAttributesReference
from infrastructure.loaders import AttributeSeriesLoader

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

    def find(self, measurement, entity_id, field):
        return {"value": str(TODAY[field])}

    def find_series(self, measurement, entity_id, field, start, end, **kwargs):
        published = datetime(2026, 9, 17, 21, tzinfo=UTC)

        if not start <= published < end:
            return []

        return [{"time": published.isoformat(), "value": str(YESTERDAY[field])}]


class _Resolver:
    def resolve(self, sensor):
        return InfluxSensor(measurement="°C", entity_id="forecast", field="time")

    def resolve_attributes(self, sensor):
        return {
            name: InfluxSensor(measurement="°C", entity_id="forecast", field=name)
            for name in sensor.attributes
        }


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
