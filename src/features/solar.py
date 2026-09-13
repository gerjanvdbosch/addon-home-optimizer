import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
from optuna import Study, Trial, create_study
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from tqdm import tqdm

from domain.time import to_local_time
from domain.types import (
    BacktestPoint,
    BacktestResult,
    Config,
    ForecasterType,
)
from features.dataset import DatasetBuilder, DatasetDefinition
from features.forecasters import SklearnForecaster

MIN_SOLAR_IRRADIANCE = 100.0
RETRAIN_INTERVAL_HOURS = 6
MAX_TRAIN_WINDOW_DAYS = 30
MAX_LEAD_TIME_HOURS = 3.0


class SolarForecaster(SklearnForecaster):
    @property
    def name(self) -> ForecasterType:
        return "solar"

    @property
    def label(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "P_solar"

    @property
    def exog_columns(self) -> list[str]:
        return [
            "p50",
            "p90",
            "spread_upper",
            "spread_lower",
            "solcast_skewness",
            "clear_sky_ratio",
            "global_tilted_irradiance",
            "direct_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "diffuse_fraction",
            "weather_discrepancy",
            "temperature",
            "wind_speed",
            "cloud_cover_low",
            "cloud_cover_mid",
            "solar_elevation",
            "lead_time_hours",
            "lag_30m_error",
            "lag_30m_trend",
            "lag_24h_mean",
        ]

    def _get_split_time(
        self, df: pd.DataFrame, test_ratio: float
    ) -> pd.Timestamp | None:
        if test_ratio <= 0.0 or test_ratio >= 1.0:
            return None

        unique_times = pd.Series(df["time"].unique()).sort_values()
        split_idx = int(len(unique_times) * (1.0 - test_ratio))
        return pd.Timestamp(unique_times.iloc[split_idx])

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.008, 0.05, log=True
            ),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 40, 120),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1.0, 200.0, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
        }

    def create(self, **overrides: Any) -> HistGradientBoostingRegressor:
        params: dict[str, Any] = dict(
            loss="absolute_error",
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=76,
            l2_regularization=22,
            max_depth=8,
            random_state=42,
            early_stopping=False,
        )
        params.update(overrides)

        return HistGradientBoostingRegressor(**params)

    def predict_arguments(self, df: pd.DataFrame, steps: int = 24) -> pd.DataFrame:
        now = datetime.now(UTC)

        future = (
            df[(df["target_time"] > now) & (df["time"] <= now)]
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .sort_values("target_time")
            .iloc[:steps]
        )

        return future[self.exog_columns]

    def arguments(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.dropna(subset=[self.target_column, *self.exog_columns]).copy()

        df = df[
            (df["p50"] >= MIN_SOLAR_IRRADIANCE)
            & (df["lead_time_hours"] >= 0.5)
            & (df["lead_time_hours"] <= MAX_LEAD_TIME_HOURS)
        ].copy()

        y_target = df[self.target_column] - df["p50"]

        return df[self.exog_columns], y_target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600.0

        df["spread_upper"] = (df["p90"] - df["p50"]).clip(lower=0)
        df["spread_lower"] = (df["p50"] - df["p10"]).clip(lower=0)

        df["hour"] = df["target_time"].dt.hour + df["target_time"].dt.minute / 60.0

        df["season_phase"] = np.cos(
            2 * np.pi * (df["target_time"].dt.dayofyear - 172) / 365.25
        )

        lat_rad = np.radians(52.0)
        doy = df["target_time"].dt.dayofyear
        dec_rad = np.radians(-23.45 * np.cos(np.radians(360.0 / 365.25 * (doy + 10))))
        hour_angle_rad = np.radians((df["hour"] - 12.0) * 15.0)
        sin_elevation = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(
            dec_rad
        ) * np.cos(hour_angle_rad)
        df["solar_elevation"] = np.degrees(
            np.arcsin(sin_elevation.clip(-1.0, 1.0))
        ).clip(lower=0.0)
        df["clear_sky_ratio"] = (df["p50"] / (df["p90"] + 50.0)).clip(0.0, 1.2)

        df["weather_discrepancy"] = df["p50"] - df["global_tilted_irradiance"]

        wind_clipped = df["wind_speed"].clip(lower=0.0)
        df["estimated_cell_temp"] = df["temperature"] + df[
            "global_tilted_irradiance"
        ] * np.exp(-3.56 - 0.075 * wind_clipped)
        df["temp_loss_factor"] = 1.0 - 0.004 * (df["estimated_cell_temp"] - 25.0)

        df["weather_discrepancy"] = df["p50"] - df["global_tilted_irradiance"]

        actuals = (
            df[["target_time", "time", "P_solar", "P_max", "P_std", "p50"]]
            .dropna(subset=["P_solar"])
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .set_index("target_time")
            .sort_index()
            .asfreq("30min")
        )

        lag_1 = actuals.shift(1)
        lag_2 = actuals.shift(2)
        lag_3 = actuals.shift(3)

        eps = 50.0
        solcast_error = actuals["P_solar"] - actuals["p50"]
        solcast_rel_error = (actuals["P_solar"] - actuals["p50"]) / (
            actuals["p50"] + eps
        )
        perf_ratio = actuals["P_solar"] / (actuals["p50"] + eps)

        volatility = (actuals["P_std"] / (actuals["P_solar"] + eps)).clip(0.0, 5.0)
        peak_ratio = (actuals["P_max"] / (actuals["P_solar"] + eps)).clip(1.0, 10.0)

        df["lag_30m_mean"] = df["time"].map(lag_1["P_solar"]).fillna(0.0)
        df["lag_30m_max"] = df["time"].map(lag_1["P_max"]).fillna(0.0)
        df["lag_30m_std"] = df["time"].map(lag_1["P_std"]).fillna(0.0)
        df["lag_30m_trend"] = (
            df["time"].map(lag_1["P_solar"] - lag_2["P_solar"]).fillna(0.0)
        )

        df["lag_30m_error"] = df["time"].map(solcast_error.shift(1)).fillna(0.0)
        df["lag_30m_rel_error"] = df["time"].map(solcast_rel_error.shift(1)).fillna(0.0)
        df["solcast_performance_ratio"] = (
            df["time"].map(perf_ratio.shift(1)).fillna(1.0)
        )

        df["solar_volatility"] = df["time"].map(volatility.shift(1)).fillna(0.0)
        df["peak_to_mean_ratio"] = df["time"].map(peak_ratio.shift(1)).fillna(1.0)

        df["lag_1h_trend"] = (
            df["time"].map(lag_1["P_solar"] - lag_3["P_solar"]).fillna(0.0)
        )

        rolling_2h_solar = actuals["P_solar"].shift(1).rolling(window=4, min_periods=1)
        rolling_2h_max = actuals["P_max"].shift(1).rolling(window=4, min_periods=1)
        rolling_2h_error = solcast_error.shift(1).rolling(window=4, min_periods=1)

        df["lag_2h_mean"] = df["time"].map(rolling_2h_solar.mean()).fillna(0.0)
        df["lag_2h_max"] = df["time"].map(rolling_2h_max.max()).fillna(0.0)
        df["lag_2h_std"] = df["time"].map(rolling_2h_solar.std()).fillna(0.0)
        df["lag_2h_error_mean"] = df["time"].map(rolling_2h_error.mean()).fillna(0.0)

        decay_rate = 0.5
        df["persisted_error_decayed"] = df["lag_30m_error"] * np.exp(
            -decay_rate * df["lead_time_hours"]
        )

        days_back = np.ceil(df["lead_time_hours"].clip(lower=0.5) / 24.0).astype(int)
        ref_past_target = df["target_time"] - pd.to_timedelta(days_back * 24, unit="h")
        df["lag_24h_mean"] = ref_past_target.map(actuals["P_solar"]).fillna(0.0)

        spread_up = (df["p90"] - df["p50"]).clip(lower=0)
        spread_down = (df["p50"] - df["p10"]).clip(lower=0)
        df["solcast_skewness"] = ((spread_up + 10.0) / (spread_down + 10.0)).clip(
            0.1, 10.0
        )

        total_rad = df["direct_radiation"] + df["diffuse_radiation"]
        df["diffuse_fraction"] = (df["diffuse_radiation"] / (total_rad + 10.0)).clip(
            0.0, 1.0
        )

        df = df.drop(columns=["P_max", "P_std"], errors="ignore")

        return df.sort_values(["time", "target_time"])

    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.Series:
        result: pd.Series = super().predict(df, steps=steps)

        if result.empty:
            return result

        return result.resample("15min").interpolate(method="time").clip(lower=0.0)

    def predict_result(self, prediction: np.ndarray, df: pd.DataFrame) -> pd.Series:
        p50 = df["p50"].to_numpy()

        dynamic_shrinkage = np.clip(
            0.15 + 0.03 * df["lead_time_hours"].to_numpy(), 0.15, 0.55
        )
        correction = dynamic_shrinkage * np.nan_to_num(prediction, nan=0.0)

        final_prediction = p50 + correction
        final_prediction = np.where(p50 < MIN_SOLAR_IRRADIANCE, p50, final_prediction)
        final_prediction = np.maximum(final_prediction, 0.0)

        return pd.Series(
            final_prediction,
            index=df["target_time"],
            name="pred",
        )

    def generate_walk_forward_folds(
        self,
        df: pd.DataFrame,
        steps: int,
        refit_hours: int = RETRAIN_INTERVAL_HOURS,
        max_train_days: int = MAX_TRAIN_WINDOW_DAYS,
    ) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame, bool]]:
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)
        time_index = pd.DatetimeIndex(df_sorted["time"])
        update_times = df_sorted["time"].unique()

        starts = time_index.searchsorted(update_times, side="left")
        ends = time_index.searchsorted(update_times, side="right")

        retrain_every = pd.Timedelta(hours=refit_hours)
        max_train_window = pd.Timedelta(days=max_train_days)

        last_trained_time = None
        train_df = pd.DataFrame()

        for update_time, start, end in zip(update_times, starts, ends, strict=True):
            update_time = pd.Timestamp(update_time)
            group = df_sorted.iloc[start:end]

            forecast = group[group["target_time"] > update_time].drop_duplicates(
                "target_time", keep="last"
            )
            test_df = forecast[forecast[self.target_column].notna()].iloc[:steps].copy()

            if test_df.empty:
                continue

            need_retrain = (
                last_trained_time is None
                or (update_time - last_trained_time) >= retrain_every
            )

            if need_retrain:
                window_start = update_time - max_train_window
                train_start_idx = time_index.searchsorted(window_start, side="left")
                train_slice = df_sorted.iloc[train_start_idx:start]

                train_mask = (
                    (train_slice["target_time"] < update_time)
                    & (train_slice["target_time"] > train_slice["time"])
                    & train_slice[self.target_column].notna()
                )
                candidate_train = train_slice[train_mask]

                if not candidate_train.empty:
                    train_df = candidate_train
                    last_trained_time = update_time

            yield update_time, train_df, test_df, need_retrain

    def backtest(
        self,
        df: pd.DataFrame,
        steps: int = 48,
        test_ratio: float = 0.2,
    ) -> BacktestResult:
        df = self.prepare(df).dropna(subset=[self.target_column, "p50"])
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        split_time = self._get_split_time(df_sorted, test_ratio)
        if split_time is not None:
            logging.info(
                "Backtest evalueert out-of-sample vanaf %s (laatste %.0f%% van de data)",
                split_time,
                test_ratio * 100,
            )

        def make_points(df: pd.DataFrame, value_col: str) -> list[dict]:
            records = df[["target_time", value_col]].to_dict("records")
            return [
                {
                    "time": pd.to_datetime(record["target_time"]).isoformat(),
                    "value": float(record[value_col]),
                }
                for record in records
            ]

        actual_slice = df_sorted
        if split_time is not None:
            actual_slice = df_sorted[df_sorted["target_time"] >= split_time]

        actual = (
            actual_slice[["target_time", "P_solar"]]
            .dropna(subset=["P_solar"])
            .drop_duplicates("target_time")
            .sort_values("target_time")
        )

        backtest_points = [
            BacktestPoint(
                label="Actual",
                group="Actual",
                color="white",
                points=make_points(actual, "P_solar"),
            )
        ]

        y_true_daylight: list[float] = []
        y_pred_daylight: list[float] = []
        y_base_daylight: list[float] = []

        baseline_errors_all: list[float] = []
        ml_errors_all: list[float] = []

        model = None
        last_X_train = None
        last_y_train = None

        windows = [
            (0.5, 2.0),
            (2.0, 4.0),
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 24.0),
        ]

        window_errors = {f"{start:g}-{end:g}h": [] for start, end in windows}
        window_baseline_errors = {f"{start:g}-{end:g}h": [] for start, end in windows}

        folds = self.generate_walk_forward_folds(df_sorted, steps=steps)
        total_updates = df_sorted["time"].nunique()
        progress = tqdm(folds, total=total_updates, desc="Solar backtest")

        for update_time, train_df, test_df, need_retrain in progress:
            if need_retrain and not train_df.empty:
                try:
                    X_train, y_train = self.arguments(train_df)
                    if not X_train.empty:
                        best_params = getattr(self, "best_params", {})
                        model = self.create(**best_params)
                        model.fit(X_train, y_train)

                        last_X_train = X_train
                        last_y_train = y_train
                except ValueError:
                    continue

            if split_time is not None and update_time < split_time:
                continue

            if model is None:
                continue

            test_clean = test_df.dropna(subset=self.exog_columns).copy()
            if test_clean.empty:
                continue

            error_pred = model.predict(test_clean[self.exog_columns])
            test_clean["pred"] = self.predict_result(error_pred, test_clean).to_numpy()

            err_base = (test_clean["P_solar"] - test_clean["p50"]).abs()
            err_ml = (test_clean["P_solar"] - test_clean["pred"]).abs()

            baseline_errors_all.extend(err_base.tolist())
            ml_errors_all.extend(err_ml.tolist())

            daylight_mask = test_clean["p50"] >= MIN_SOLAR_IRRADIANCE
            if daylight_mask.any():
                y_true_daylight.extend(
                    test_clean.loc[daylight_mask, "P_solar"].tolist()
                )
                y_pred_daylight.extend(test_clean.loc[daylight_mask, "pred"].tolist())
                y_base_daylight.extend(test_clean.loc[daylight_mask, "p50"].tolist())

            for start, end in windows:
                label = f"{start:g}-{end:g}h"
                mask = (
                    (test_clean["lead_time_hours"] >= start)
                    & (test_clean["lead_time_hours"] < end)
                    & daylight_mask
                )
                window_test = test_clean.loc[mask]

                if not window_test.empty:
                    window_baseline_errors[label].extend(
                        (window_test["P_solar"] - window_test["p50"]).abs().tolist()
                    )
                    window_errors[label].extend(
                        (window_test["P_solar"] - window_test["pred"]).abs().tolist()
                    )

            ts = pd.to_datetime(str(update_time))
            label_ts = to_local_time(ts.to_pydatetime()).strftime("%d-%m %H:%M")

            backtest_points.append(
                BacktestPoint(
                    label=f"ML {label_ts}",
                    group="ML",
                    points=make_points(test_clean.sort_values("target_time"), "pred"),
                )
            )
            backtest_points.append(
                BacktestPoint(
                    label=f"Update {label_ts}",
                    group="Solcast",
                    points=make_points(test_clean.sort_values("target_time"), "p50"),
                )
            )

        if y_true_daylight:
            y_true = np.array(y_true_daylight)
            y_pred = np.array(y_pred_daylight)
            y_base = np.array(y_base_daylight)

            day_base_mae = float(np.mean(np.abs(y_true - y_base)))
            day_ml_mae = float(np.mean(np.abs(y_true - y_pred)))
            day_imp_mae = (
                100 * (day_base_mae - day_ml_mae) / day_base_mae
                if day_base_mae
                else 0.0
            )

            day_base_rmse = float(np.sqrt(np.mean((y_true - y_base) ** 2)))
            day_ml_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            day_imp_rmse = (
                100 * (day_base_rmse - day_ml_rmse) / day_base_rmse
                if day_base_rmse
                else 0.0
            )

            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            day_base_r2 = (
                float(1.0 - (np.sum((y_true - y_base) ** 2) / ss_tot))
                if ss_tot > 0
                else 0.0
            )
            day_ml_r2 = (
                float(1.0 - (np.sum((y_true - y_pred) ** 2) / ss_tot))
                if ss_tot > 0
                else 0.0
            )
        else:
            day_base_mae, day_ml_mae, day_imp_mae = 0.0, 0.0, 0.0
            day_base_rmse, day_ml_rmse, day_imp_rmse = 0.0, 0.0, 0.0
            day_base_r2, day_ml_r2 = 0.0, 0.0

        all_base_mae = (
            float(np.mean(baseline_errors_all)) if baseline_errors_all else 0.0
        )
        all_ml_mae = float(np.mean(ml_errors_all)) if ml_errors_all else 0.0

        logging.info(
            "  MAE: baseline=%.2f W | ML=%.2f W | improvement=%+.1f%%",
            day_base_mae,
            day_ml_mae,
            day_imp_mae,
        )
        logging.info(
            "  RMSE: baseline=%.2f W | ML=%.2f W | improvement=%+.1f%%",
            day_base_rmse,
            day_ml_rmse,
            day_imp_rmse,
        )
        logging.info(
            "  R²:   baseline=%.4f   | ML=%.4f   | delta=%+.4f",
            day_base_r2,
            day_ml_r2,
            day_ml_r2 - day_base_r2,
        )
        logging.info(
            "Solar MAE (24/7 all hours): baseline=%.2f W, ML=%.2f W",
            all_base_mae,
            all_ml_mae,
        )

        logging.info("Solar MAE per lead-time window (Daylight):")

        for start, end in windows:
            label = f"{start:g}-{end:g}h"
            baseline_window = window_baseline_errors[label]
            ml_window = window_errors[label]

            if ml_window:
                base_w = float(np.mean(baseline_window))
                ml_w = float(np.mean(ml_window))
                imp_w = 100 * (base_w - ml_w) / base_w if base_w else 0.0
                logging.info(
                    "  %8s: baseline=%.2f W | ML=%.2f W | improvement=%+.1f%% | n=%d",
                    label,
                    base_w,
                    ml_w,
                    imp_w,
                    len(ml_window),
                )

        if model is not None and last_X_train is not None and not last_X_train.empty:
            result = cast(
                Bunch,
                permutation_importance(
                    model, last_X_train, last_y_train, n_repeats=5, random_state=42
                ),
            )

            top_features = last_X_train.columns[
                result.importances_mean.argsort()[::-1][:15]
            ]
            logging.info(f"Top 15 important features: {list(top_features)}")

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=day_ml_mae,
            rmse=day_ml_rmse,
            r2=day_ml_r2,
            points=backtest_points,
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 12,
        n_trials: int = 30,
        study_storage: str | Path | None = None,
        refit_hours: int = 24,
        test_ratio: float = 0.2,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df).dropna(subset=[self.target_column, "p50"])
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        split_time = self._get_split_time(df_sorted, test_ratio)
        if split_time is not None:
            logging.info(
                "Tune gebruikt data tot %s (eerste %.0f%% van de data)",
                split_time,
                (1.0 - test_ratio) * 100,
            )
            df_tune = df_sorted[df_sorted["time"] < split_time].copy()
        else:
            df_tune = df_sorted

        folds = list(
            self.generate_walk_forward_folds(
                df_tune,
                steps=steps,
                refit_hours=refit_hours,
                max_train_days=MAX_TRAIN_WINDOW_DAYS,
            )
        )

        def objective(trial: Trial) -> float:
            params = self.search_space(trial)
            ml_errors: list[float] = []
            model = None

            for _, train_df, test_df, need_retrain in folds:
                if need_retrain and not train_df.empty:
                    try:
                        X_train, y_train = self.arguments(train_df)
                        if not X_train.empty:
                            model = self.create(**params)
                            model.fit(X_train, y_train)
                    except ValueError:
                        pass

                if model is None:
                    continue

                test_clean = test_df.dropna(subset=self.exog_columns).copy()
                if test_clean.empty:
                    continue

                error_pred = model.predict(test_clean[self.exog_columns])
                pred = self.predict_result(error_pred, test_clean).to_numpy()

                daylight_mask = (test_clean["p50"] >= MIN_SOLAR_IRRADIANCE).to_numpy()
                if daylight_mask.any():
                    errs = np.abs(test_clean["P_solar"].to_numpy() - pred)[
                        daylight_mask
                    ]
                    ml_errors.extend(errs.tolist())

            return float(np.mean(ml_errors)) if ml_errors else 9999.0

        storage = str(study_storage) if study_storage else None
        study = create_study(
            study_name=self.name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.forecaster = self.create(**study.best_params)

        logging.info(
            "Tune finished - best Daylight MAE: %.2f | params: %s",
            study.best_value,
            study.best_params,
        )

        return study.trials_dataframe(), study

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_solar",
                config.solar,
                interval="30m",
                aggregation="mean",
                fill=0,
            )
            .timeseries(
                "P_max",
                config.solar,
                interval="30m",
                aggregation="max",
                fill=0,
            )
            .timeseries(
                "P_std",
                config.solar,
                interval="30m",
                aggregation="stddev",
                fill=0,
            )
            .attribute_timeseries(
                "solcast",
                config.forecast.solcast,
                attributes=["p10", "p50", "p90"],
                interval="30m",
                aggregation="last",
            )
            .attribute_timeseries(
                "open_meteo",
                config.forecast.open_meteo,
                attributes=[
                    "global_tilted_irradiance",
                    "direct_radiation",
                    "direct_normal_irradiance",
                    "diffuse_radiation",
                    "temperature",
                    "precipitation",
                    "wind_speed",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                ],
                interval="30m",
                aggregation="last",
                target_resample="mean",
                target_interval="30min",
                target_label="right",
                target_closed="right",
                target_shift=[
                    "global_tilted_irradiance",
                    "direct_radiation",
                    "direct_normal_irradiance",
                    "diffuse_radiation",
                    "precipitation",
                ],
            )
            .join(
                left="solcast",
                right="P_solar",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .join(
                left="solcast",
                right="P_max",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .join(
                left="solcast",
                right="P_std",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .join(
                left="solcast",
                right="open_meteo",
                on=("time", "target_time"),
                how="left",
            )
            .build()
        )
