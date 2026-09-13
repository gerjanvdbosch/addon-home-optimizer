from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from features.solar import (
    NATIVE_STEP_MINUTES,
    PREDICT_STEP_MINUTES,
    SolarForecaster,
    _native_steps,
)


class _ZeroModel:
    """Stand-in for the fitted HistGradientBoostingRegressor - predict()'s
    resolution translation is independent of the model's own fit quality
    (see the repo's identification/validation separation convention), so a
    trivial stub is enough to exercise it without a real training pass.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def _synthetic_predict_df(now: datetime, n_native_rows: int = 60) -> pd.DataFrame:
    """One forecast issue time (`now`) with `n_native_rows` target_time rows
    spaced at the dataset's native 30-minute resolution (see
    SolarForecaster.dataset()) - the minimum shape prepare()/predict_arguments()
    need. P_solar/P_max/P_std are NaN throughout since these rows are all in
    the future (nothing measured yet), matching real future-prediction input.
    """

    target_times = [now + timedelta(minutes=30 * (i + 1)) for i in range(n_native_rows)]
    n = len(target_times)

    return pd.DataFrame(
        {
            "time": [now] * n,
            "target_time": target_times,
            "p10": 400.0,
            "p50": 500.0,
            "p90": 600.0,
            "P_solar": np.nan,
            "P_max": np.nan,
            "P_std": np.nan,
            "temperature": 15.0,
            "global_tilted_irradiance": 300.0,
            "direct_radiation": 200.0,
            "direct_normal_irradiance": 250.0,
            "diffuse_radiation": 100.0,
            "wind_speed": 3.0,
            "cloud_cover_low": 10.0,
            "cloud_cover_mid": 5.0,
            "cloud_cover_high": 0.0,
            "precipitation": 0.0,
        }
    )


@pytest.mark.parametrize("steps", [1, 2, 4, 48, 96])
def test_native_steps_covers_the_requested_15_minute_span(steps):
    """Regression check for the 30min<->15min scaling bug: enough native
    30-minute rows must be requested to cover at least `steps` 15-minute
    points once resampled, for every steps value - not just the default.
    """

    native = _native_steps(steps)
    native_span_minutes = (native - 1) * NATIVE_STEP_MINUTES
    target_span_minutes = (steps - 1) * PREDICT_STEP_MINUTES

    assert native_span_minutes >= target_span_minutes


@pytest.mark.parametrize("steps", [1, 2, 4, 48, 96])
def test_predict_returns_exactly_the_requested_15_minute_steps(steps):
    """The whole point of the fix: callers pass `steps` meaning 15-minute
    steps (the convention every other forecaster and OptimizeConfig/
    PredictConfig use), regardless of this forecaster's native 30-minute
    resolution. Before the fix, `steps` was misread as native 30-minute
    rows, so predict() returned roughly 2x too many (or too few) points
    spanning 2x the intended horizon.
    """

    forecaster = SolarForecaster()
    forecaster.forecaster = _ZeroModel()

    now = datetime.now(UTC) - timedelta(minutes=1)
    df = _synthetic_predict_df(now)

    prediction = forecaster.predict(df, steps=steps)

    assert len(prediction) == steps

    if steps > 1:
        diffs = np.diff(prediction.index.values)
        assert all(diff == np.timedelta64(PREDICT_STEP_MINUTES, "m") for diff in diffs)
