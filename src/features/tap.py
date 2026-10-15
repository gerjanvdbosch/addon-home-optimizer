from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.jobs import ForecasterType
from features.boiler import BoilerThermalIdentifier
from features.dataset import DatasetBuilder
from features.forecasters import SkforecastForecaster


class TapForecaster(SkforecastForecaster):
    """Forecasts hot-water tap demand as the expected excess-heat-loss power (W),
    using the
    calibrated boiler thermal model's own residual diagnostic
    (BoilerThermalIdentifier.excess_loss_w) as the training target and presence
    as an exogenous feature.

    This does NOT predict a validated tap event or volume - no flow meter exists,
    and the underlying signal is the same unproven "candidate excess heat loss"
    used during calibration (see boiler.py). It is a forecast of that same
    residual quantity, useful as an exogenous input for MPC planning, not a claim of
    validated hot-water usage.

    Further, the target itself is a mix of two effects, not purely tap draws:
    real data shows the excess-loss flag rate is far more sensitive to the
    tank's own starting temperature (51.8% hot vs. 19.2% cool) than to presence
    (39.0% vs. 34.7%) - see BoilerThermalIdentifier.excess_loss_w's docstring
    for the underlying finding (a temperature-dependent heat-transfer effect
    the constant-UA model cannot represent). This forecaster therefore predicts
    "excess loss," a quantity influenced by both tap draws and that modeling
    gap - not tap draws alone.
    """

    # The planning model is linear, so its expected tank temperature follows from
    # the expected heat sink: this forecasts the mean, not a quantile. Backtested
    # on real data (6 h blocks, tap energy in K of tank): the former 0.85 quantile
    # was unbiased in total (-0.15 K) but spread a noise floor over quiet hours,
    # planning 1.1 K too much cooling in the 6 h after heating; the mean with
    # only a same-time-yesterday lag gives -0.12 K overall and +0.5 K there.
    # Short lags on this noisy residual (the former 1-4 steps and rolling
    # windows) made the mean miss the evening draws (-1.2 K).

    # What the forecast expects for the rest of today is held to what a heavy
    # day of the past week - one in four drew more - leaves after what has been
    # drawn since midnight: a household that showered early does not shower
    # again (real data, 16 Oct 2026: the evening's draws at 20:30-21:30, then
    # the usual 22:00 draw forecast on top, 2.3 kWh the tank never lost). Over
    # 136 forecasts issued at 18-22 h (13 Sep - 16 Oct 2026) the rest of the
    # day came out 1.38 kWh off against 1.52 for the forecast alone, 0.19 kWh
    # short on average; held to the week's mean day 1.34, but 0.53 short - the
    # heavier days' draws cut. A night's tank from 21:30 and 22:30 came out
    # 2.6 K off by the next run against 3.1 K, and on 16 Oct 1 K against 8.5 K
    # (88 nights, Sep-Oct). Only ever lowered: raised to the budget where the
    # forecast expected little, it put up to 41 times its own draw into a
    # quarter hour.
    BUDGET_DAYS = 7
    BUDGET_QUANTILE = 0.75

    def __init__(self, models_path: Path) -> None:
        self.models_path = models_path
        # The configured state labels, handed to the boiler identifier this
        # trains on; overwritten by dataset().
        self.states = HeatPumpStates()
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
                    loss="squared_error",
                    learning_rate=0.03,
                    max_depth=7,
                    max_iter=120,
                    min_samples_leaf=5,
                    l2_regularization=5.0,
                    random_state=42,
                ),
            ),
            # Tap draws follow the daily activity rhythm, so the same quarter
            # yesterday plus the calendar features carry the signal (see above).
            lags=overrides.pop("lags", [96]),
            calendar_features=overrides.pop(
                "calendar_features",
                CalendarFeatures(
                    features=["hour", "day_of_week", "weekend"], encoding="onehot"
                ),
            ),
            # No sqrt transform: the target is signed, and a mean does not
            # survive a nonlinear transform.
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

        # The target is signed (noise, model error), but a tap draw can only
        # remove heat.
        forecast = self.forecaster.predict(
            steps=steps, last_window=last_window, exog=future_exog
        ).clip(lower=0.0)

        return self._within_todays_budget(forecast, y)

    def _within_todays_budget(self, forecast: pd.Series, y: pd.Series) -> pd.Series:
        """The forecast with the rest of today held to the day's budget (see
        BUDGET_DAYS); as it is without a full week drawn before today."""

        if y.empty:
            return forecast

        midnight = y.index[-1].normalize()
        week_start = midnight - pd.Timedelta(days=self.BUDGET_DAYS)

        if y.index[0] > week_start:
            return forecast

        week = y[(y.index >= week_start) & (y.index < midnight)]
        heavy_day = float(
            np.quantile(week.groupby(week.index.date).sum(), self.BUDGET_QUANTILE)
        )
        left = max(heavy_day - y[y.index >= midnight].sum(), 0.0)
        today = forecast.index < midnight + pd.Timedelta(days=1)
        expected = forecast[today].sum()

        if expected > left:
            forecast = forecast.copy()
            forecast[today] *= left / expected

        return forecast

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        identifier = BoilerThermalIdentifier()
        identifier.states = self.states
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
        self.states = config.heat_pump.states
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
