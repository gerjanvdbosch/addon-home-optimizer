import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features.baseload import BaseloadForecaster


def _baseload_history(days: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    time = pd.date_range("2026-08-01", periods=days * 96, freq="15min", tz="UTC")
    daily = 160.0 + 60.0 * np.sin(2 * np.pi * np.arange(len(time)) / 96)
    baseload = daily + rng.normal(0, 10, len(time))
    return pd.DataFrame({"time": time, "P_baseload": baseload})


def test_fit_rebuilds_the_model_from_code_with_tuned_settings():
    """A loaded model must not keep an outdated structure (here: a squared-error
    loss) - fit() rebuilds it from create(), reapplying tuned estimator settings."""

    forecaster = BaseloadForecaster()
    forecaster.forecaster = forecaster.create(
        estimator=HistGradientBoostingRegressor(loss="squared_error")
    )
    forecaster.best_params = {"max_iter": 80, "lags": [1, 2]}

    forecaster.fit(_baseload_history())

    assert forecaster.forecaster.estimator.loss == "absolute_error"
    assert forecaster.forecaster.estimator.max_iter == 80
    assert forecaster.forecaster.is_fitted
