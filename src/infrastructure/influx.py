from datetime import datetime
from typing import Any, Iterable, Literal, Protocol, cast

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from domain.types import (
    InfluxSensor,
    SensorAttributesReference,
    SensorReference,
    Settings,
)

Aggregation = Literal[
    "mean",
    "count",
    "last",
    "first",
    "min",
    "max",
    "sum",
    "median",
    "spread",
    "stddev",
]

FillMethod = Literal[
    "none",
    "null",
    "previous",
    "linear",
]


class AttributeDefinition(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


class InfluxDatabase:
    def __init__(self, settings: Settings):
        self.client = InfluxDBClient(
            host=settings.influx_host,
            port=settings.influx_port,
            username=settings.influx_username,
            password=settings.influx_password,
            database=settings.influx_database,
        )

    def query(self, query: str) -> ResultSet:
        return cast(ResultSet, self.client.query(query))

    def find(
        self,
        measurement: str,
        entity_id: str,
        field: str,
    ) -> dict[str, Any] | None:
        query = f"""
        SELECT "{field}" AS value
        FROM "{measurement}"
        WHERE "entity_id" = '{entity_id}'
        ORDER BY time DESC
        LIMIT 1
        """

        result = self.query(query)

        points = list(result.get_points())
        if not points:
            return None

        return points[0]

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

        result = self.query(query)

        return list(result.get_points())


class InfluxSensorResolver:
    def __init__(self, db: InfluxDatabase):
        self.db = db
        self.cache: dict[str, InfluxSensor] = {}
        self.schema: list[InfluxSensor] = []
        self.schema_loaded = False

    def load_schema(self) -> None:
        if self.schema_loaded:
            return

        measurements = self.db.query("SHOW MEASUREMENTS")

        for measurement in measurements.get_points():
            name = measurement["name"]

            fields = self.db.query(f'SHOW FIELD KEYS FROM "{name}"')

            for field in fields.get_points():
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
        return self._resolve(
            entity_id=sensor.entity_id,
            attribute=sensor.attribute,
        )

    def resolve_attributes(
        self,
        sensor: "SensorAttributesReference[AttributeDefinition]",
    ) -> dict[str, InfluxSensor]:
        return {
            name: self._resolve(
                entity_id=sensor.entity_id,
                attribute=attribute,
            )
            for name, attribute in sensor.attributes.items()
        }

    def _resolve(
        self,
        entity_id: str,
        attribute: str | None,
    ) -> InfluxSensor:
        self.load_schema()

        entity_id = entity_id.split(".", 1)[-1]
        cache_key = f"{entity_id}.{attribute}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        candidates = self._candidate_fields(attribute)

        for field_name in candidates:
            for influx_sensor in self.schema:
                if influx_sensor.field != field_name:
                    continue

                query = f"""
                SELECT "{field_name}"
                FROM "{influx_sensor.measurement}"
                WHERE "entity_id" = '{entity_id}'
                LIMIT 1
                """

                if list(self.db.query(query).get_points()):
                    resolved = InfluxSensor(
                        measurement=influx_sensor.measurement,
                        entity_id=entity_id,
                        field=field_name,
                        value_type=influx_sensor.value_type,
                    )

                    self.cache[cache_key] = resolved

                    return resolved

        raise ValueError(f"Sensor not found: {entity_id}.{attribute}")

    def _candidate_fields(self, attribute: str | None) -> list[str]:
        if attribute is None:
            return ["value", "state"]

        return [f"{attribute}_str", attribute]
