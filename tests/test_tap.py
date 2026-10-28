from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from domain.dynamics import discretize_zoh
from domain.models import BoilerThermalModel
from domain.physics import mains_water_temperature_c, tank_state_space
from domain.time import LOCAL_TIMEZONE
from features.boiler import BoilerThermalIdentifier
from features.tap import TapForecaster

TRUE_MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=0.15,
    ua_bottom_w_per_k=0.20,
    ua_mix_idle_w_per_k=0.03,
    ua_mix_active_w_per_k=9638.6,
    q_in_nominal_w=3700.0,
)

DT_SECONDS = 900.0  # 15 minutes, the forecast's own step (the dataset is per minute)
SAMPLES_PER_DAY = int(24 * 3600 / DT_SECONDS)
T_AMBIENT_C = 20.0


def _simulate_with_draws_when_present(rng: np.random.Generator, days: int):
    """A short daily heating cycle plus injected excess-loss "draw" events that
    only ever occur while present=True - so a correctly working TapForecaster
    should learn to predict near-zero excess loss when present=False.
    """

    a_idle, b_idle = tank_state_space(
        TRUE_MODEL.volume_l,
        TRUE_MODEL.ua_top_w_per_k,
        TRUE_MODEL.ua_bottom_w_per_k,
        TRUE_MODEL.ua_mix_idle_w_per_k,
    )
    a_active, b_active = tank_state_space(
        TRUE_MODEL.volume_l,
        TRUE_MODEL.ua_top_w_per_k,
        TRUE_MODEL.ua_bottom_w_per_k,
        TRUE_MODEL.ua_mix_active_w_per_k,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    heating_start_sample = 32  # ~08:00
    heating_duration_samples = 5

    n_samples = days * SAMPLES_PER_DAY
    state = np.array([45.0, 45.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)
    present = np.zeros(n_samples, dtype=bool)
    draw_remaining = 0

    for i in range(n_samples):
        sample_in_day = i % SAMPLES_PER_DAY
        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on

        # Home in the evening (samples 64-95 of each day, ~16:00-24:00).
        is_present = 64 <= sample_in_day < 96
        present[i] = is_present

        if not on and is_present:
            if draw_remaining <= 0 and rng.random() < 0.15:
                draw_remaining = int(rng.integers(1, 3))
            if draw_remaining > 0:
                state = np.maximum(state - np.array([1.0, 4.0]), T_AMBIENT_C)
                draw_remaining -= 1

        states[i] = state
        u = np.array([T_AMBIENT_C, TRUE_MODEL.q_in_nominal_w if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, 0.05, size=states.shape)
    measured = states + noise

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n_samples)],
        utc=True,
    )

    return pd.DataFrame(
        {
            "time": time,
            "T_ambient": T_AMBIENT_C,
            "T_top": measured[:, 0],
            "T_bottom": measured[:, 1],
            "state": np.where(boiler_on, "SWW", "Uit"),
            "presence_0": np.where(present, "home", "not_home"),
        }
    )


def _save_calibrated_identifier(models_path) -> None:
    identifier = BoilerThermalIdentifier()
    identifier.model = TRUE_MODEL
    identifier.save(models_path)


def test_prepare_builds_the_target(tmp_path):
    _save_calibrated_identifier(tmp_path)

    rng = np.random.default_rng(7)
    df = _simulate_with_draws_when_present(rng, days=10)

    forecaster = TapForecaster(models_path=tmp_path)
    prepared = forecaster.prepare(df)

    assert "excess_loss_w" in prepared.columns

    # No NaNs left in the target (boiler_on periods are filled with 0, not NaN).
    assert not prepared["excess_loss_w"].isna().any()

    # The target is signed (see excess_loss_w()): noise must average out.
    assert (prepared["excess_loss_w"] < 0).any()

    # The same draws per kelvin above the mains: January's are some 30 K.
    mains_c = [mains_water_temperature_c(day) for day in prepared.index.day_of_year]
    np.testing.assert_allclose(
        prepared["draw_w_per_k"] * (40.0 - np.array(mains_c)),
        prepared["excess_loss_w"],
    )


def test_prepare_raises_without_a_calibrated_model(tmp_path):
    rng = np.random.default_rng(1)
    df = _simulate_with_draws_when_present(rng, days=2)

    forecaster = TapForecaster(models_path=tmp_path)  # nothing saved at tmp_path

    with pytest.raises(RuntimeError):
        forecaster.prepare(df)


def test_draws_are_forecast_where_the_past_weeks_drew(tmp_path):
    """Draws only ever in the evening: the next day's are forecast there, from
    the past weeks' own quarter hours, and hardly any at night."""

    _save_calibrated_identifier(tmp_path)

    rng = np.random.default_rng(3)
    df = _simulate_with_draws_when_present(rng, days=30)

    prediction = TapForecaster(models_path=tmp_path).predict(df, steps=SAMPLES_PER_DAY)

    assert len(prediction) == SAMPLES_PER_DAY
    assert (prediction >= 0).all()
    evening = (prediction.index.hour >= 16).astype(bool)
    assert prediction[evening].mean() > 10 * prediction[~evening].mean()


def test_what_was_drawn_today_is_not_forecast_again(tmp_path):
    """A week of one 3 kWh shower each evening: drawn at 20:00 today, the
    usual 22:00 shower is not forecast on top of it; tomorrow's still is, and
    a day that drew nothing yet keeps its forecast."""

    forecaster = TapForecaster(tmp_path)
    index = pd.date_range(
        "2026-10-09 00:00", "2026-10-16 21:00", freq="15min", tz=LOCAL_TIMEZONE
    )
    # 12 kW for a quarter hour is 3 kWh.
    shower = (index.hour == 22) & (index.minute == 0)
    drawn = pd.Series(np.where(shower, 12000.0, 0.0), index=index)
    drawn[pd.Timestamp("2026-10-16 20:00", tz=LOCAL_TIMEZONE)] = 12000.0
    ahead = pd.date_range(
        index[-1] + pd.Timedelta(minutes=15), periods=8 * 4, freq="15min"
    )
    forecast = pd.Series(
        np.where((ahead.hour == 22) & (ahead.minute == 0), 12000.0, 0.0), index=ahead
    )
    tomorrow = ahead >= pd.Timestamp("2026-10-17", tz=LOCAL_TIMEZONE)
    forecast[tomorrow & (ahead.hour == 4)] = 1000.0

    _, heavy_day_w = forecaster._habit(
        pd.DataFrame({"excess_loss_w": drawn, "draw_w_per_k": drawn})
    )
    held = forecaster._within_todays_budget(forecast, drawn, heavy_day_w)

    assert held[~tomorrow].sum() == pytest.approx(0.0)
    np.testing.assert_allclose(held[tomorrow], forecast[tomorrow])

    nothing_yet = drawn.copy()
    nothing_yet[pd.Timestamp("2026-10-16 20:00", tz=LOCAL_TIMEZONE)] = 0.0

    np.testing.assert_allclose(
        forecaster._within_todays_budget(forecast, nothing_yet, heavy_day_w), forecast
    )
