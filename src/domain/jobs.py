"""What the worker can be asked to do, and the payload of each request."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from domain.config import Config
from domain.models import IdentificationType


class JobType(str, Enum):
    CONFIG = "config"
    UPDATE = "update"
    CALIBRATE = "calibrate"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"


class WorkerState(str, Enum):
    """Where a submitted job is: waiting its turn, or running. A job leaves
    the worker's list once it has ended, however it ended."""

    QUEUED = "queued"
    RUNNING = "running"


JsonType = dict[str, Any] | list[Any]


class UpdateConfig(BaseModel): ...


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
    # The most the MPC horizon may span, in 15-minute steps; it ends at the
    # end of tomorrow (see
    # Optimization.run).
    steps: int = Field(default=192)
    # Also log the plan beside alternatives that finish its first run earlier
    # (see Optimization.explain_dhw_plan) - a few more solves, so off unless
    # asked for.
    explain: bool = Field(default=False)
    # Plan the zone cooled (True) or heated (False) whatever mode the heat pump
    # is set to, to see a mode's plan before switching the heat pump over.
    # Only this run: the next one follows the heat pump again. None follows it
    # now (see HeatPumpConfig.mode).
    cooling: bool | None = Field(default=None)


@dataclass
class Job:
    type: JobType
    config: Config | UpdateConfig | CalibrateConfig | ValidateConfig | OptimizeConfig
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
