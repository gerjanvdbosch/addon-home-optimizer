from collections.abc import Iterable
from datetime import datetime
from typing import Any, cast
from urllib.parse import quote_plus

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from domain.config import Settings
from domain.sensors import (
    Aggregation,
    FillMethod,
    InfluxSensor,
    SensorReference,
)


class InfluxDatabase:
    def __init__(self, settings: Settings):
        self.client = InfluxDBClient(
            host=settings.influx_host,
            port=settings.influx_port,
            username=settings.influx_username,
            password=settings.influx_password,
            database=settings.influx_database,
        )

    # The most query one request carries, as it stands in the URL (characters,
    # URL-encoded): the client sends the statements there, and a proxy before
    # InfluxDB commonly takes no request line past 8 KiB - InfluxDB itself
    # takes a megabyte. Room left for the rest of the line and the database.
    MAX_ENCODED_QUERY_CHARS = 7000

    def query(self, query: str) -> ResultSet:
        return cast(ResultSet, self.client.query(query))

    def query_many(self, queries: list[str]) -> list[ResultSet]:
        """Several statements in as few requests as fit, their results in the
        same order: each request is a round trip to Home Assistant, and a
        sensor's resolution alone took two or three of them."""

        separator = ";\n"
        results: list[ResultSet] = []
        batch: list[str] = []
        size = 0

        def send() -> None:
            answer = self.client.query(separator.join(batch))
            results.extend(answer if isinstance(answer, list) else [answer])
            batch.clear()

        for query in queries:
            encoded = len(quote_plus(separator + query))

            if batch and size + encoded > self.MAX_ENCODED_QUERY_CHARS:
                send()
                size = 0

            batch.append(query)
            size += encoded

        if batch:
            send()

        if len(results) != len(queries):
            raise RuntimeError(
                f"{len(queries)} statements returned {len(results)} results."
            )

        return results

    def find(
        self,
        measurement: str,
        entity_id: str,
        field: str,
        before: datetime | None = None,
    ) -> dict[str, Any] | None:
        """The latest reading, or the latest before `before`."""

        return first_point(
            self.query(self.find_query(measurement, entity_id, field, before))
        )

    def find_series(
        self,
        measurement: str,
        entity_id: str,
        field: str,
        start: datetime,
        end: datetime,
        interval: str | None = None,
        aggregation: Aggregation | None = None,
        fill: FillMethod | int | float = "none",
    ) -> list[dict[str, Any]]:
        return list(
            self.query(
                self.series_query(
                    measurement,
                    entity_id,
                    field,
                    start,
                    end,
                    interval,
                    aggregation,
                    fill,
                )
            ).get_points()
        )

    @staticmethod
    def find_query(
        measurement: str,
        entity_id: str,
        field: str,
        before: datetime | None = None,
    ) -> str:
        until = f"AND time < '{before.isoformat()}'" if before else ""

        return f"""
        SELECT "{field}" AS value
        FROM "{measurement}"
        WHERE "entity_id" = '{entity_id}' {until}
        ORDER BY time DESC
        LIMIT 1
        """

    @staticmethod
    def series_query(
        measurement: str,
        entity_id: str,
        field: str,
        start: datetime,
        end: datetime,
        interval: str | None = None,
        aggregation: Aggregation | None = None,
        fill: FillMethod | int | float = "none",
    ) -> str:
        if interval and aggregation:
            select = f'{aggregation}("{field}")'
        else:
            select = f'"{field}"'

        query = f"""
        SELECT {select} AS value
        FROM "{measurement}"
        WHERE
            "entity_id" = '{entity_id}'
            AND time >= '{start.isoformat()}'
            AND time < '{end.isoformat()}'
        """

        if interval and aggregation:
            query += f"""
        GROUP BY time({interval}) fill({fill})
        """

        return query


def first_point(result: ResultSet) -> dict[str, Any] | None:
    points = list(result.get_points())

    return points[0] if points else None


class InfluxSensorResolver:
    def __init__(self, db: InfluxDatabase):
        self.db = db
        self.cache: dict[str, InfluxSensor] = {}
        self.schema: list[InfluxSensor] = []
        self.schema_loaded = False

    def load_schema(self) -> None:
        if self.schema_loaded:
            return

        # One query for every measurement's fields: a job runs in a fresh
        # process, so this runs once per job, and asking per measurement cost
        # a query for each of this database's ~90.
        for (name, _), fields in self.db.query("SHOW FIELD KEYS").items():
            for field in fields:
                self.schema.append(
                    InfluxSensor(
                        measurement=name,
                        entity_id="",
                        field=field["fieldKey"],
                        value_type=field["fieldType"],
                    )
                )

        self.schema_loaded = True

    def resolve(self, sensor: SensorReference) -> InfluxSensor:
        self.resolve_many([sensor])
        cache_key = self._cache_key(sensor)

        if cache_key not in self.cache:
            raise ValueError(f"Sensor not found: {cache_key}")

        return self.cache[cache_key]

    def resolve_many(self, sensors: Iterable[SensorReference]) -> None:
        """Find where each sensor is stored, into the cache: in two requests
        for all of them, the measurements each entity has series in and then
        whether one holds the field - tried in the same order as one at a
        time, so the same one is found. A sensor not found is left out."""

        self.load_schema()

        pending = {
            self._cache_key(sensor): sensor
            for sensor in sensors
            if self._cache_key(sensor) not in self.cache
        }

        if not pending:
            return

        entity_ids = list(
            dict.fromkeys(self._entity_id(sensor) for sensor in pending.values())
        )
        # Only the measurements an entity has series in, from the index. Trying
        # every measurement with a matching field instead took one query each -
        # ~70 for a plain "value" sensor.
        measurements = {
            entity_id: {point["name"] for point in result.get_points()}
            for entity_id, result in zip(
                entity_ids,
                self.db.query_many(
                    [
                        f"SHOW MEASUREMENTS WHERE \"entity_id\" = '{entity_id}'"
                        for entity_id in entity_ids
                    ]
                ),
                strict=True,
            )
        }

        candidates: list[tuple[str, InfluxSensor]] = []

        for cache_key, sensor in pending.items():
            entity_id = self._entity_id(sensor)

            for field_name in self._candidate_fields(sensor.attribute):
                for stored in self.schema:
                    if (
                        stored.field == field_name
                        and stored.measurement in measurements[entity_id]
                    ):
                        candidates.append(
                            (
                                cache_key,
                                InfluxSensor(
                                    measurement=stored.measurement,
                                    entity_id=entity_id,
                                    field=field_name,
                                    value_type=stored.value_type,
                                ),
                            )
                        )

        holds = self.db.query_many(
            [
                f"""SELECT "{candidate.field}" FROM "{candidate.measurement}" """
                f"""WHERE "entity_id" = '{candidate.entity_id}' LIMIT 1"""
                for _, candidate in candidates
            ]
        )

        for (cache_key, candidate), result in zip(candidates, holds, strict=True):
            if cache_key not in self.cache and list(result.get_points()):
                self.cache[cache_key] = candidate

    @staticmethod
    def _entity_id(sensor: SensorReference) -> str:
        return sensor.entity_id.split(".", 1)[-1]

    def _cache_key(self, sensor: SensorReference) -> str:
        return f"{self._entity_id(sensor)}.{sensor.attribute}"

    def _candidate_fields(self, attribute: str | None) -> list[str]:
        if attribute is None:
            return ["value", "state"]

        return [f"{attribute}_str", attribute]
