from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer

from domain.types import Config, ForecasterType
from features.boiler import BoilerThermalIdentifier
from features.dataset import DatasetBuilder, DatasetDefinition
from features.forecasters import SkforecastForecaster


class TapForecaster(SkforecastForecaster):
    """Forecasts hot-water tap demand as a conservative (see
    TAP_FORECAST_QUANTILE) estimate of excess-heat-loss power (W), using the
    calibrated boiler thermal model's own residual diagnostic
    (BoilerThermalIdentifier.excess_loss_w) as the training target and presence
    as an exogenous feature.

    This does NOT predict a validated tap event or volume - no flow meter exists,
    and the underlying signal is the same unproven "candidate excess heat loss"
    used during calibration (see boiler.py). It is a forecast of that same
    residual quantity, useful as an exogenous input (e.g. for cleaning the
    boiler's own calibration data, or for MPC planning), not a claim of
    validated hot-water usage. It is also not an expected/average value - see
    TAP_FORECAST_QUANTILE for why an upper quantile is used instead.

    Further, the target itself is a mix of two effects, not purely tap draws:
    real data shows the excess-loss flag rate is far more sensitive to the
    tank's own starting temperature (51.8% hot vs. 19.2% cool) than to presence
    (39.0% vs. 34.7%) - see BoilerThermalIdentifier.excess_loss_w's docstring
    for the underlying finding (a temperature-dependent heat-transfer effect
    the constant-UA model cannot represent). This forecaster therefore predicts
    "excess loss," a quantity influenced by both tap draws and that modeling
    gap - not tap draws alone.
    """

    # excess_loss_w is extremely zero-inflated with a heavy tail (confirmed on
    # real data: 82% exactly 0, but up to ~7.8 kW when a draw occurs) and its
    # timing is uncertain. A squared-error point forecast is then optimized
    # towards a diluted "probability-of-a-draw x typical size" expected value,
    # not the size of an actual draw when one occurs - confirmed on real data,
    # where the model never predicted above ~200 W despite training on spikes
    # up to 7.8 kW. For sizing an MPC's heating margin, understating a real
    # heat sink is the costlier mistake (a missed temperature target) than
    # overstating one (a little extra, still solar-first heating), so this
    # forecasts a conservative upper quantile instead of the mean. Quantiles
    # are monotonic-transform-equivariant (unlike the mean), so this stays
    # valid together with the sqrt/square transformer below.
    TAP_FORECAST_QUANTILE = 0.85

    def __init__(self, models_path: Path) -> None:
        self.models_path = models_path
        super().__init__()

    @property
    def name(self) -> ForecasterType:
        return "tap"

    @property
    def label(self) -> str:
        return "Estimated tap draw"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "excess_loss_w"

    @property
    def exog_columns(self) -> list[str]:
        return ["present"]

    def create(self, **overrides: Any):
        return ForecasterRecursive(
            forecaster_id=overrides.pop("forecaster_id", self.name),
            estimator=overrides.pop(
                "estimator",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=self.TAP_FORECAST_QUANTILE,
                    learning_rate=0.03,
                    max_depth=7,
                    max_iter=120,
                    min_samples_leaf=5,
                    l2_regularization=5.0,
                    random_state=42,
                ),
            ),
            # No weekly lags (unlike BaseloadForecaster): Forecasting.predict()
            # only ever supplies 7 days of context (app/forecasting.py), and this
            # forecaster's own prepare() drops a few rows on top of that (the
            # first row has no dt_seconds, BoilerThermalIdentifier.prepare()
            # filters oversized gaps, excess_loss_w() drops the last row) - so a
            # lag near 7*96=672 is not reliably available. Physically, tap draws
            # are driven by daily human activity rhythm, already covered by the
            # daily lags below plus the hour/day_of_week/weekend calendar
            # features; a residual same-time-last-week effect on top of that
            # (real for whole-household baseload, e.g. work-from-home patterns)
            # is not a justified assumption for tap draws specifically.
            lags=overrides.pop("lags", [1, 2, 3, 4, 95, 96, 97]),
            calendar_features=overrides.pop(
                "calendar_features",
                CalendarFeatures(
                    features=["hour", "day_of_week", "weekend"], encoding="onehot"
                ),
            ),
            window_features=overrides.pop(
                "window_features",
                RollingFeatures(
                    stats=["mean", "mean", "max", "max"],
                    window_sizes=[4, 96, 4, 96],
                ),
            ),
            # excess_loss_w is non-negative and right-skewed (mostly near zero,
            # occasional spikes) - same distributional shape as baseload, same
            # variance-stabilizing transform.
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

    def predict(self, df: pd.DataFrame, steps: int = 48) -> pd.Series:
        prepared = self.prepare(df)

        y = prepared[self.target_column].dropna()
        last_window = y if not y.empty else None

        # No presence forecast exists (predicting future presence is a separate,
        # harder problem this forecaster does not attempt) - conservatively
        # assume "present" for the whole horizon, since that cannot understate a
        # possible draw the way assuming an empty house would. A real presence
        # forecast, if one is built later, would replace this block.
        future_index = pd.date_range(
            start=prepared.index[-1] + prepared.index.freq,
            periods=steps,
            freq=prepared.index.freq,
        )
        future_exog = pd.DataFrame({"present": 1.0}, index=future_index)

        return self.forecaster.predict(
            steps=steps, last_window=last_window, exog=future_exog
        )

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        identifier = BoilerThermalIdentifier()
        identifier.load(self.models_path)

        if identifier.model is None:
            raise RuntimeError(
                "The boiler thermal model must be calibrated (see "
                "BoilerThermalIdentifier) before the tap forecaster can be "
                "trained or used, since it trains on that model's own "
                "excess-heat-loss diagnostic."
            )

        # Reuses the exact same column names dataset() below requests, so
        # presence_columns is whatever presence_N columns are actually present -
        # no need to thread config through here.
        identifier.presence_columns = sorted(
            column for column in df.columns if column.startswith("presence_")
        )

        prepared = identifier.prepare(df)
        excess_loss = identifier.excess_loss_w(prepared)

        merged = excess_loss.merge(
            prepared[["time", "confirmed_away_settled"]], on="time", how="left"
        )

        # "Present" = NOT reliably known to be an empty house - the inverse of the
        # same conservative, whitelist-only confirmed_away_settled column already
        # computed by prepare() (only trusts an explicit away reading, sustained
        # for a settling margin - see boiler.py). Where no trackers are
        # configured, confirmed_away_settled is always False, so this defaults to
        # "present" everywhere - the safer assumption for not under-predicting
        # tap demand.
        merged["present"] = (~merged["confirmed_away_settled"]).astype(float)

        # A real draw during active heating is out of scope (confounded with
        # Q_in, same as the calibration diagnostics) - treated as a neutral zero
        # signal here rather than a gap, so the training series stays regular
        # instead of full of holes skforecast would need to fill.
        merged["excess_loss_w"] = merged["excess_loss_w"].fillna(0.0)

        merged = merged[["time", "excess_loss_w", "present"]]

        return super().prepare(merged)

    def dataset(self, config: Config) -> DatasetDefinition:
        builder = (
            DatasetBuilder()
            .timeseries(
                "T_ambient",
                config.heat_pump.boiler.ambient_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="15m",
                aggregation="last",
                fill="previous",
            )
        )

        for i, sensor in enumerate(config.presence):
            builder = builder.timeseries(
                f"presence_{i}",
                sensor,
                interval="15m",
                aggregation="last",
                fill="previous",
            )

        return builder.build()
