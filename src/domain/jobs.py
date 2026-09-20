"""What the worker can be asked to do, and the payload of each request."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from domain.config import Config
from domain.models import ForecasterType, IdentificationType


class JobType(str, Enum):
    CONFIG = "config"
    UPDATE = "update"
    FIT = "fit"
    PREDICT = "predict"
    TUNE = "tune"
    BACKTEST = "backtest"
    CALIBRATE = "calibrate"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"


JsonType = dict[str, Any] | list[Any]


class UpdateConfig(BaseModel): ...


class FitConfig(BaseModel):
    target: ForecasterType | None = Field(default=None)
    days: int = Field(default=90)


class PredictConfig(BaseModel):
    target: ForecasterType | None = Field(default=None)
    steps: int = Field(default=192)


class BacktestConfig(BaseModel):
    target: ForecasterType
    days: int = Field(default=90)
    # Same horizon as PredictConfig.steps, so backtest and tune measure the
    # forecast the optimizer actually receives, not a shorter, easier one.
    steps: int = Field(default=192)


class TuneConfig(BacktestConfig):
    trails: int = Field(default=10)


class CalibrateConfig(BaseModel):
    target: IdentificationType | None = Field(default=None)
    days: int = Field(default=90)


class ValidateConfig(BaseModel):
    target: IdentificationType | None = Field(default=None)
    days: int = Field(default=90)
    # Anchors the `days`-long window to this instant instead of "now" - so
    # repeated validations (e.g. checking a model change, or the solar bias
    # identifier's own multi-window pooling) evaluate the exact same
    # historical period rather than one that drifts forward every time the
    # job runs, which otherwise makes two runs' numbers incomparable
    # regardless of what actually changed between them.
    end: datetime | None = Field(default=None)


class OptimizeConfig(BaseModel):
    # MPC horizon in 15-minute steps (same convention as PredictConfig.steps) -
    # fixed here so the optimizer always plans over this many steps, rather
    # than incidentally following however many points the last solar
    # prediction happened to produce.
    steps: int = Field(default=192)


@dataclass
class Job:
    type: JobType
    config: (
        Config
        | UpdateConfig
        | FitConfig
        | PredictConfig
        | TuneConfig
        | BacktestConfig
        | CalibrateConfig
        | ValidateConfig
        | OptimizeConfig
    )
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
