import logging
from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from joblib import dump, load

from domain.types import Config
from features.dataset import DatasetDefinition

logger = logging.getLogger(__name__)

SystemModel = TypeVar("SystemModel")


class SystemIdentifier(Generic[SystemModel]):
    def __init__(self) -> None:
        self.model: SystemModel | None = None
        # Remembered so a subclass can locate sibling model files saved alongside
        # its own (see BoilerThermalIdentifier's use of the tap-demand forecaster).
        self.models_path: Path | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def label(self) -> str: ...

    @property
    @abstractmethod
    def unit(self) -> str: ...

    @abstractmethod
    def calibrate(self, df: pd.DataFrame) -> SystemModel: ...

    @abstractmethod
    def validate(
        self, df: pd.DataFrame, horizon_hours: float = 2.0
    ) -> dict[str, float]: ...

    @abstractmethod
    def dataset(self, config: Config) -> DatasetDefinition: ...

    def get_model(self, dt_hours: float = 0.25) -> SystemModel:
        if self.model is None:
            raise RuntimeError(f"Model {self.name} not calibrated")

        if hasattr(self.model, "model_copy"):
            return self.model.model_copy(update={"dt_hours": dt_hours})

        return self.model

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError(f"Model {self.name} not calibrated")

        path.mkdir(parents=True, exist_ok=True)

        target_file = path / f"{self.name}.joblib"

        dump(self.model, target_file)

    def load(self, path: Path) -> None:
        self.models_path = path

        target_file = path / f"{self.name}.joblib"

        if not target_file.exists():
            logger.warning(f"Model {self.name} not calibrated")
            return

        self.model = load(target_file)

    @staticmethod
    def _parameter_std_errors(fit_result) -> np.ndarray:
        """Standard errors from a scipy.optimize.least_squares result's own
        Jacobian - shared by any subclass identifying parameters this way
        (see BoilerThermalIdentifier and HeatPumpCOPIdentifier), so a
        parameter pinned at a bound or otherwise poorly determined by the
        data can be reported rather than presented as a precise value.
        """

        degrees_of_freedom = max(len(fit_result.fun) - len(fit_result.x), 1)
        residual_variance = float(np.sum(fit_result.fun**2) / degrees_of_freedom)

        try:
            covariance = residual_variance * np.linalg.inv(
                fit_result.jac.T @ fit_result.jac
            )
            return np.sqrt(np.diag(covariance))
        except np.linalg.LinAlgError:
            return np.full(len(fit_result.x), np.nan)
