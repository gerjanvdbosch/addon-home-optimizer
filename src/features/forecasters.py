import logging
from abc import abstractmethod
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

from domain.types import BacktestPoint, BacktestResult, Config, ForecasterType
from features.dataset import DatasetDefinition


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


class SkforecastForecaster(Forecaster):
    def __init__(self):
        self.forecaster = self.create()

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

        self.forecaster.fit(y=y, exog=exog)

    def predict(
        self,
        df: pd.DataFrame,
        steps: int = 48,
    ) -> pd.Series:
        df = self.prepare(df)

        exog = self.predict_arguments(df=df, steps=steps)

        y = df[self.target_column].dropna()
        last_window = y if not y.empty else None

        return self.forecaster.predict(steps=steps, last_window=last_window, exog=exog)

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

    def _load_best_params(self, storage: str) -> bool:
        if not Path(storage.removeprefix("sqlite:///")).exists():
            return False

        study = create_study(
            study_name=self.name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
        )

        if not study.trials or study.best_trial is None:
            return False

        self.best_params = study.best_params

        logging.info("Load best params %s", study.best_params)

        return True

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        dump(self.forecaster, path / f"{self.name}.joblib")

    def load(self, path: Path, study_storage: str | None = None) -> None:
        file_name = path / f"{self.name}.joblib"

        if not file_name.exists():
            return

        self.forecaster = load(file_name)

        if study_storage is not None:
            self._load_best_params(storage=study_storage)
