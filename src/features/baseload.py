from typing import Any

import numpy as np
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer

from domain.types import Config, ForecasterType
from features.dataset import DatasetBuilder, DatasetDefinition
from features.forecasters import SkforecastForecaster


class BaseloadForecaster(SkforecastForecaster):
    @property
    def name(self) -> ForecasterType:
        return "baseload"

    @property
    def label(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "P_baseload"

    @property
    def exog_columns(self) -> list[str]:
        return []

    def create(self, **overrides: Any):
        return ForecasterRecursive(
            forecaster_id=overrides.pop("forecaster_id", self.name),
            estimator=overrides.pop(
                "estimator",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.03,
                    max_depth=7,
                    max_iter=120,
                    min_samples_leaf=5,
                    l2_regularization=5.0,
                    random_state=42,
                ),
            ),
            lags=overrides.pop("lags", [1, 2, 3, 4, 95, 96, 97, 671, 672, 673]),
            calendar_features=overrides.pop(
                "calendar_features",
                CalendarFeatures(
                    features=["hour", "day_of_week", "weekend"], encoding="onehot"
                ),
            ),
            window_features=overrides.pop(
                "window_features",
                RollingFeatures(
                    stats=["mean", "mean", "min", "min"],
                    window_sizes=[4, 96, 4, 96],
                ),
            ),
            transformer_y=FunctionTransformer(func=np.sqrt, inverse_func=np.square),
            **overrides,
        )

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.02,
                0.06,
                log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "max_iter": trial.suggest_int(
                "max_iter",
                80,
                160,
                step=40,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                3,
                12,
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1.0, 15.0, log=True
            ),
        }

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_baseload",
                config.baseload,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .build()
        )
