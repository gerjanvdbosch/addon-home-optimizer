import logging
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from joblib import dump, load
from optuna import Study, Trial, create_study
from skforecast.base import ForecasterBase
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)
from skforecast.utils import load_forecaster, save_forecaster
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from domain.config import Config
from domain.dataset import DatasetDefinition
from domain.jobs import ForecasterType
from domain.state import BacktestPoint, BacktestResult
from domain.time import to_local_time


class Forecaster(Protocol):
    @property
    def name(self) -> ForecasterType: ...

    @property
    def target_column(self) -> str: ...

    @property
    def exog_columns(self) -> list[str]: ...

    @property
    def label(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def dataset(self, config: Config) -> DatasetDefinition: ...

    def fit(self, df: pd.DataFrame): ...

    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.Series: ...

    def predict_band(
        self, df: pd.DataFrame, steps: int = 48
    ) -> tuple[pd.Series, pd.Series] | None: ...

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult: ...

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path, study_storage: str | None = None) -> None: ...


def _best_params(name: str, storage: str) -> dict[str, Any]:
    """Best hyperparameters from this forecaster's Optuna study (see tune()), or
    {} while it has not been tuned."""

    if not Path(storage.removeprefix("sqlite:///")).exists():
        return {}

    study = create_study(
        study_name=name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )

    if not study.trials or study.best_trial is None:
        return {}

    logging.info("Load best params %s", study.best_params)

    return study.best_params


class SkforecastForecaster(Forecaster):
    # The lower and upper quantile predict_band() gives, None for none.
    band_quantiles: tuple[float, float] | None = None

    def __init__(self):
        self.forecaster = self.create()
        self.best_params: dict[str, Any] = {}

    @abstractmethod
    def create(self, **overrides: Any) -> ForecasterBase: ...

    def arguments(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame | None]:
        y = df[self.target_column]
        exog = df[self.exog_columns] if self.exog_columns else None

        return y, exog

    def predict_arguments(
        self,
        df: pd.DataFrame,
        steps: int = 48,
    ) -> pd.DataFrame | None:
        return df[self.exog_columns].iloc[:steps] if self.exog_columns else None

    @abstractmethod
    def search_space(self, trial: Trial) -> dict[str, Any]: ...

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().set_index("time").sort_index().asfreq("15min")

    def fit(self, df: pd.DataFrame):
        df = self.prepare(df)

        y, exog = self.arguments(df)

        # Rebuilt from code rather than refitting the loaded model, so a change to
        # the model itself (e.g. its loss) takes effect; tuned estimator settings
        # are reapplied on top.
        self.forecaster = self.create()
        estimator_params = self.forecaster.estimator.get_params()
        self.forecaster.estimator.set_params(
            **{k: v for k, v in self.best_params.items() if k in estimator_params}
        )

        self.forecaster.fit(y=y, exog=exog)

        cv = self.create_cv(df, 96)

        # The band comes from out-of-sample errors: a walk-forward over the last
        # 30% of this data, as backtest() does. In-sample errors of a boosted
        # model are too small to bound what it gets wrong ahead. History too
        # short to train on before that part leaves the model without a band.
        # Taken per local hour of the day, not per forecast level: appliances
        # follow the clock. Grouped by the level the median forecast took
        # instead (skforecast's binned residuals), the dinner peak's spread
        # carried into a quiet late evening - p90 932 W at 22 h against 393 W
        # measured - and the morning's washing into nothing (real data, 98
        # days walk-forward: per-hour coverage 4.8 points off its 10% on
        # average, 2.2 per hour of the day).
        if (
            self.band_quantiles is not None
            and cv.initial_train_size > self.forecaster.window_size
        ):
            _, result = backtesting_forecaster(
                n_jobs=1,
                metric="mean_absolute_error",
                forecaster=deepcopy(self.forecaster),
                cv=cv,
                y=y,
                exog=exog,
                show_progress=False,
            )
            error = y.loc[result.index] - result["pred"]
            # Kept on the model itself, so it is saved and loaded with it.
            self.forecaster.band_by_hour_ = (
                error.groupby(self._local_hours(error.index))
                .quantile(list(self.band_quantiles))
                .unstack()
            )

    @staticmethod
    def _local_hours(index: pd.Index) -> np.ndarray:
        return np.array([to_local_time(time).hour for time in index])

    def _last_window(self, df: pd.DataFrame) -> pd.Series | None:
        y = df[self.target_column].dropna()

        return y if not y.empty else None

    def predict(
        self,
        df: pd.DataFrame,
        steps: int = 48,
    ) -> pd.Series:
        df = self.prepare(df)

        return self.forecaster.predict(
            steps=steps,
            last_window=self._last_window(df),
            exog=self.predict_arguments(df=df, steps=steps),
        )

    def predict_band(
        self,
        df: pd.DataFrame,
        steps: int = 48,
    ) -> tuple[pd.Series, pd.Series] | None:
        """The band_quantiles of the forecast: the median and fit()'s
        out-of-sample error at each quantile for that hour of the day. Empty for
        a model fitted without them."""

        if self.band_quantiles is None:
            return None

        band_by_hour = getattr(self.forecaster, "band_by_hour_", None)

        if band_by_hour is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        median = self.predict(df, steps)
        error = band_by_hour.reindex(self._local_hours(median.index))
        error = error.fillna(band_by_hour.median())
        low, high = (error[q].to_numpy() for q in self.band_quantiles)

        return (median + low).clip(lower=0.0), median + high

    def backtest(
        self,
        df: pd.DataFrame,
        steps: int = 24,
    ) -> BacktestResult:
        df = self.prepare(df)

        y, exog = self.arguments(df)

        metric, result = backtesting_forecaster(
            n_jobs=1,
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps, 96),
            y=y,
            exog=exog,
        )

        return self.backtest_result(
            df=df,
            metric=metric,
            result=result,
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df)

        y, exog = self.arguments(df)

        # Searched from the model as defined in code, not the loaded one (see fit()).
        self.forecaster = self.create()

        result, study = bayesian_search_forecaster(
            n_jobs=1,
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps, False),
            y=y,
            exog=exog,
            search_space=self.search_space,
            n_trials=n_trials,
            random_state=42,
            return_best=True,
            kwargs_create_study={
                "study_name": self.name,
                "storage": study_storage,
                "load_if_exists": True,
                "direction": "minimize",
            },
        )

        return result, cast(Study, study)

    def create_cv(
        self,
        df: pd.DataFrame,
        steps: int,
        refit: bool | int = False,
    ) -> TimeSeriesFold:
        return TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(df) * 0.7),
            refit=refit,
            fixed_train_size=False,
        )

    def backtest_result(
        self,
        df: pd.DataFrame,
        metric: pd.DataFrame,
        result: pd.DataFrame,
    ) -> BacktestResult:
        result = result.copy()
        result["actual"] = df.loc[result.index, self.target_column]

        def make_point(label: str, column: str) -> BacktestPoint:
            points = (
                result[[column]]
                .rename(columns={column: "value"})
                .rename_axis("time")
                .reset_index()
                .to_dict("records")
            )

            return BacktestPoint(
                label=label,
                points=cast(list[dict[str, object]], points),
            )

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=float(metric["mean_absolute_error"].iloc[0]),
            rmse=float(np.sqrt(mean_squared_error(result["actual"], result["pred"]))),
            r2=float(r2_score(result["actual"], result["pred"])),
            points=[
                make_point("Actual", "actual"),
                make_point("Prediction", "pred"),
            ],
        )

    def save(self, path: Path) -> None:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_forecaster(
            self.forecaster,
            str(path / f"{self.name}.joblib"),
        )

    def load(self, path: Path, study_storage: str | None = None) -> None:
        if study_storage is not None:
            self.best_params = _best_params(self.name, study_storage)

        file_name = path / f"{self.name}.joblib"

        if not file_name.exists():
            return

        self.forecaster = cast(
            ForecasterBase,
            load_forecaster(str(file_name)),
        )


class SklearnForecaster(Forecaster):
    def __init__(self):
        self.forecaster = self.create()
        self.best_params: dict[str, Any] = {}

    @abstractmethod
    def create(self, **overrides: Any) -> HistGradientBoostingRegressor: ...

    def search_space(self, trial: Trial) -> dict[str, Any]:
        raise NotImplementedError()

    def arguments(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[self.exog_columns], df[self.target_column]

    def predict_arguments(self, df: pd.DataFrame, steps: int = 48) -> pd.DataFrame:
        return df[self.exog_columns].iloc[:steps]

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fit(self, df: pd.DataFrame):
        df = self.prepare(df)

        self.forecaster = self.create(**self.best_params)

        X, y = self.arguments(df)

        self.forecaster.fit(X, y)

    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.Series:
        df = self.prepare(df)

        X = self.predict_arguments(df=df, steps=steps)

        prediction = self.forecaster.predict(X)

        return self.predict_result(prediction, df.reindex(X.index))

    def predict_result(self, prediction: np.ndarray, df: pd.DataFrame) -> pd.Series:
        return pd.Series(prediction, index=df.index)

    def predict_band(
        self, df: pd.DataFrame, steps: int = 48
    ) -> tuple[pd.Series, pd.Series] | None:
        return None

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        raise NotImplementedError()

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]:
        raise NotImplementedError()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        dump(self.forecaster, path / f"{self.name}.joblib")

    def load(self, path: Path, study_storage: str | None = None) -> None:
        file_name = path / f"{self.name}.joblib"

        if not file_name.exists():
            return

        self.forecaster = load(file_name)

        if study_storage is not None:
            self.best_params = _best_params(self.name, study_storage)
