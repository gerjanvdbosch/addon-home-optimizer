from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from domain.types import BoilerThermalModel
from features.boiler import BoilerThermalIdentifier, _state_space, discretize_zoh
from features.tap import TapForecaster

TRUE_MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=0.15,
    ua_bottom_w_per_k=0.20,
    ua_mix_idle_w_per_k=0.03,
    ua_mix_active_w_per_k=9638.6,
    q_in_nominal_w=3700.0,
)

DT_SECONDS = 900.0  # 15 minutes, matching TapForecaster.dataset()'s interval
SAMPLES_PER_DAY = int(24 * 3600 / DT_SECONDS)
T_AMBIENT_C = 20.0


def _simulate_with_draws_when_present(rng: np.random.Generator, days: int):
    """A short daily heating cycle plus injected excess-loss "draw" events that
    only ever occur while present=True - so a correctly working TapForecaster
    should learn to predict near-zero excess loss when present=False.
    """

    a_idle, b_idle = _state_space(
        TRUE_MODEL.volume_l,
        TRUE_MODEL.ua_top_w_per_k,
        TRUE_MODEL.ua_bottom_w_per_k,
        TRUE_MODEL.ua_mix_idle_w_per_k,
    )
    a_active, b_active = _state_space(
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


def test_prepare_builds_target_and_presence_columns(tmp_path):
    _save_calibrated_identifier(tmp_path)

    rng = np.random.default_rng(7)
    df = _simulate_with_draws_when_present(rng, days=10)

    forecaster = TapForecaster(models_path=tmp_path)
    prepared = forecaster.prepare(df)

    assert "excess_loss_w" in prepared.columns
    assert "present" in prepared.columns
    assert prepared.index.name == "time"

    # No NaNs left in the target (boiler_on periods are filled with 0, not NaN).
    assert not prepared["excess_loss_w"].isna().any()

    # Every excess-loss value is non-negative (clipped, see excess_loss_w()).
    assert (prepared["excess_loss_w"] >= 0).all()

    # present is a real 0/1 signal, not constant (both present and away periods
    # exist in the synthetic data).
    assert prepared["present"].nunique() > 1


def test_prepare_raises_without_a_calibrated_model(tmp_path):
    rng = np.random.default_rng(1)
    df = _simulate_with_draws_when_present(rng, days=2)

    forecaster = TapForecaster(models_path=tmp_path)  # nothing saved at tmp_path

    with pytest.raises(RuntimeError):
        forecaster.prepare(df)


def test_fit_and_predict_end_to_end(tmp_path):
    _save_calibrated_identifier(tmp_path)

    rng = np.random.default_rng(3)
    df = _simulate_with_draws_when_present(rng, days=30)

    forecaster = TapForecaster(models_path=tmp_path)
    forecaster.fit(df)

    prediction = forecaster.predict(df, steps=SAMPLES_PER_DAY)

    assert len(prediction) == SAMPLES_PER_DAY
    assert (prediction >= 0).all()
