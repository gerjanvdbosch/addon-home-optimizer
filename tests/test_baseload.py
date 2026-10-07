import numpy as np
import pandas as pd
import pytest
from skforecast.preprocessing import CalendarFeatures
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.time import LOCAL_TIMEZONE
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


def test_the_band_from_out_of_sample_errors_holds_four_in_five_of_a_new_day(
    tmp_path,
):
    """p10-p90 holds about 80% of a day the model has not seen; it is not
    centred on the median, which a biased model's errors rightly shift."""

    forecaster = BaseloadForecaster()
    # Few boosting rounds: the band's mechanics, not the model's accuracy.
    forecaster.best_params = {"max_iter": 10}
    forecaster.fit(_baseload_history(days=20))
    days = _baseload_history(days=9)
    history, actual = days.iloc[:-96], days["P_baseload"].iloc[-96:].to_numpy()

    p10, p90 = forecaster.predict_band(history, steps=96)
    inside = (actual >= p10.to_numpy()) & (actual <= p90.to_numpy())

    assert 0.65 <= inside.mean() <= 0.95

    # Saved beside the model and loaded with it.
    forecaster.save(tmp_path)
    loaded = BaseloadForecaster()
    loaded.load(tmp_path)

    assert loaded.predict_band(history, steps=96)[1].to_numpy() == pytest.approx(
        p90.to_numpy()
    )

    # Too short to train on before the walk-forward: no band rather than one
    # from in-sample errors.
    forecaster.fit(_baseload_history(days=10))
    p10, p90 = forecaster.predict_band(history, steps=96)

    assert p10.empty and p90.empty


def test_the_calendar_keeps_the_local_clock_across_a_dst_change():
    """The calendar features are the household's clock: an hour of the day is
    the same local hour in summer and winter time, an hour apart in UTC - and
    the forecast itself comes back in UTC."""

    time = pd.date_range("2026-10-15", "2026-10-28", freq="15min", tz="UTC")
    history = pd.DataFrame({"time": time, "P_baseload": 150.0})
    forecaster = BaseloadForecaster()

    hours = CalendarFeatures(features=["hour"], encoding=None).fit_transform(
        forecaster.prepare(history).index
    )["hour"]
    # Dinner at 17:45 in Amsterdam, either side of 25 October; the same local
    # hour twice wherever the system's zone has that change.
    for instant in ("2026-10-24 15:45Z", "2026-10-26 16:45Z"):
        local = pd.Timestamp(instant).tz_convert(LOCAL_TIMEZONE)

        assert hours[local] == local.hour

    forecaster.best_params = {"max_iter": 10}
    forecaster.fit(history)

    assert str(forecaster.predict(history, steps=4).index.tz) == "UTC"
