"""How a measurement is addressed, and how it is put on a grid.

A sensor reference names a Home Assistant entity (and optionally one of its
attributes); the aggregation and fill vocabulary says how its samples become
one value per model step. That is the model's question, not the database's -
a rate that stops being reported is zero, a state that stops being reported
still holds - so it is stated here and the InfluxDB adapter reads it.
"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator


class InfluxSensor(BaseModel):
    measurement: str
    entity_id: str
    field: str
    value_type: str | None = None


# The vocabulary a dataset definition is written in: how raw samples are
# collapsed onto a grid step, and what an absent sample means there. Stated
# here rather than in the InfluxDB adapter because it is the model's question,
# not the database's - a rate that stops being reported is zero, a state that
# stops being reported still holds (see the dataset definitions in features/).
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


class SensorReference(BaseModel):
    entity_id: str = Field()
    attribute: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attribute": None,
            }

        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attribute": value[1],
            }

        return value


T = TypeVar("T")


class SensorAttributesReference(BaseModel, Generic[T]):
    entity_id: str = Field()
    attributes: T

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attributes": {},
            }

        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attributes": value[1],
            }

        return value
