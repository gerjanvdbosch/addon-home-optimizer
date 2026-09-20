"""What data a model needs, and who can fetch it.

A dataset definition says which sensors a model wants, on which grid, and how
their frames line up - a statement about the model, not about any database.
The DataLoader protocol is the seam: features ask for a definition to be
loaded, infrastructure knows how to get it out of InfluxDB.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeVar

import pandas as pd
from pandas._typing import MergeHow
from pydantic import BaseModel

from domain.sensors import (
    Aggregation,
    FillMethod,
    SensorAttributesReference,
    SensorReference,
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
