from typing import Any

import numpy as np
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures
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
                # Median, not mean: baseload is a ~160 W base with short appliance
                # spikes up to ~3 kW whose timing cannot be forecast. A mean
                # forecast smears them over every hour as load that mostly never
                # comes (on real data ~86 W on average, ~280 W predicted at night
                # against ~156 W actual), which would make the optimizer
                # underestimate solar surplus. The median cut that to ~10 W and
                # the error by 28% (walk-forward, last 28 days). The sqrt/square
                # transformer below leaves the median unchanged.
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.03,
                    max_depth=7,
                    max_iter=120,
                    min_samples_leaf=5,
                    l2_regularization=5.0,
                    random_state=42,
                ),
            ),
            # Only same-time-yesterday and same-time-last-week lags, no recent
            # (1-4 step) lags or rolling windows. Forecasting is recursive, so
            # beyond the first hour those recent inputs are the model's own
            # predictions; in training, though, the previous quarter hour almost
            # fully explains the next one, so the model learned to follow it
            # instead of the daily rhythm. The forecast then collapsed to a flat
            # line (within-day spread 23 W vs 240 W actual) that missed the
            # dinner load present on most days. Without them (walk-forward, last
            # 28 days, 24h folds): MAE 105 -> 101 W, 17-18h median 157 -> 247 W
            # (actual 254 W), phantom load 10 -> 19 W, and no loss even 0-2h
            # ahead (45 -> 44 W).
            lags=overrides.pop("lags", [95, 96, 97, 671, 672, 673]),
            calendar_features=overrides.pop(
                "calendar_features",
                CalendarFeatures(
                    features=["hour", "day_of_week", "weekend"], encoding="onehot"
                ),
            ),
            window_features=overrides.pop("window_features", None),
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
