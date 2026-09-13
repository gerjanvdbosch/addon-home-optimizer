import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from domain.types import BoilerThermalModel
from features.boiler import (
    BoilerThermalIdentifier,
    _rollout,
    _state_space,
    discretize_zoh,
)
from features.tap import TapForecaster

TRUE_VOLUME_L = 200.0
TRUE_UA_TOP_W_PER_K = 1.2
TRUE_UA_BOTTOM_W_PER_K = 1.0
TRUE_UA_MIX_IDLE_W_PER_K = 0.3
TRUE_UA_MIX_ACTIVE_W_PER_K = 600.0
TRUE_Q_IN_NOMINAL_W = 4500.0

DT_SECONDS = 300.0
SAMPLES_PER_DAY = int(24 * 3600 / DT_SECONDS)
N_DAYS = 20

T_AMBIENT_C = 20.0
MEASUREMENT_NOISE_STD_C = 0.05

# Days (0-indexed) on which a synthetic, undetectable-by-design tap draw is injected
# during an idle period, to test that the robust fit isn't badly biased by them.
DRAW_DAYS = {2, 5, 8, 11, 14, 17}
DRAW_SAMPLE_OFFSET = 200  # samples after the heating cycle ends
DRAW_BOTTOM_DROP_C = 10.0
DRAW_TOP_DROP_C = 3.0


def _simulate(rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic multi-day boiler trajectory from known ODE parameters,
    with injected discrete tap-draw perturbations, to test parameter recovery.
    """

    heating_start_sample = 100
    heating_duration_samples = 15  # 75 minutes

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    n_samples = N_DAYS * SAMPLES_PER_DAY

    state = np.array([45.0, 45.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        day = i // SAMPLES_PER_DAY
        sample_in_day = i % SAMPLES_PER_DAY

        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on

        if (
            day in DRAW_DAYS
            and sample_in_day
            == heating_start_sample + heating_duration_samples + DRAW_SAMPLE_OFFSET
        ):
            state = state - np.array([DRAW_TOP_DROP_C, DRAW_BOTTOM_DROP_C])

        states[i] = state

        u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
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
        }
    )


def test_calibrate_recovers_known_parameters():
    rng = np.random.default_rng(42)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L

    prepared = identifier.prepare(df)
    model = identifier.calibrate(prepared)

    assert model.ua_top_w_per_k == pytest.approx(TRUE_UA_TOP_W_PER_K, rel=0.3)
    assert model.ua_bottom_w_per_k == pytest.approx(TRUE_UA_BOTTOM_W_PER_K, rel=0.3)
    assert model.ua_mix_active_w_per_k == pytest.approx(
        TRUE_UA_MIX_ACTIVE_W_PER_K, rel=0.5
    )
    assert model.q_in_nominal_w == pytest.approx(TRUE_Q_IN_NOMINAL_W, rel=0.15)

    # Non-negativity is enforced by the fit bounds, not just coincidentally true here.
    assert model.ua_top_w_per_k >= 0
    assert model.ua_bottom_w_per_k >= 0
    assert model.ua_mix_idle_w_per_k >= 0
    assert model.ua_mix_active_w_per_k >= 0
    assert model.q_in_nominal_w >= 0


# UA values on the order this installation's real calibration actually produced
# (~0.13-0.19 W/K) - slow enough that the true one-step (5 min) decay signal is
# far below MEASUREMENT_NOISE_STD_C. See PASSIVE_DECAY_HORIZON_HOURS.
SLOW_UA_TOP_W_PER_K = 0.15
SLOW_UA_BOTTOM_W_PER_K = 0.20
SLOW_UA_MIX_IDLE_W_PER_K = 0.03
SLOW_UA_MIX_ACTIVE_W_PER_K = 600.0
SLOW_Q_IN_NOMINAL_W = 4500.0


def _simulate_slow_decay(rng: np.random.Generator, days: int) -> pd.DataFrame:
    """Mirrors _simulate()'s daily heating-cycle structure, but with passive-loss
    parameters slow enough that the true one-step decay signal is well below
    sensor noise - the real-world scenario PASSIVE_DECAY_HORIZON_HOURS targets.
    """

    heating_start_sample = 100
    heating_duration_samples = 15

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        SLOW_UA_TOP_W_PER_K,
        SLOW_UA_BOTTOM_W_PER_K,
        SLOW_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        SLOW_UA_TOP_W_PER_K,
        SLOW_UA_BOTTOM_W_PER_K,
        SLOW_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    n_samples = days * SAMPLES_PER_DAY
    state = np.array([45.0, 45.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        sample_in_day = i % SAMPLES_PER_DAY
        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on
        states[i] = state

        u = np.array([T_AMBIENT_C, SLOW_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
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
        }
    )


def test_calibration_recovers_slow_passive_decay_via_multi_step_residuals():
    """Regression test for a real finding: on real data, a model fit purely
    from one-step residuals predicted ~5x too little cooling over several days
    compared to a directly observed decay during a confirmed-nobody-home
    window - the true per-5-minute-step signal for a decay this slow is below
    sensor noise. calibrate()'s added decay_residuals (same-anchor rollout
    comparisons over clean idle windows, see PASSIVE_DECAY_HORIZON_HOURS) must
    recover UA_top/UA_bottom accurately despite that.
    """

    rng = np.random.default_rng(55)
    df = _simulate_slow_decay(rng, days=30)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    model = identifier.calibrate(df)

    assert model.ua_top_w_per_k == pytest.approx(SLOW_UA_TOP_W_PER_K, rel=0.2)
    assert model.ua_bottom_w_per_k == pytest.approx(SLOW_UA_BOTTOM_W_PER_K, rel=0.2)
    assert model.q_in_nominal_w == pytest.approx(SLOW_Q_IN_NOMINAL_W, rel=0.1)


def test_validate_reports_segmented_prediction_error_not_draw_claims():
    rng = np.random.default_rng(7)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L

    prepared = identifier.prepare(df)
    identifier.calibrate(prepared)
    metrics = identifier.validate(prepared)

    # The synthetic data is generated from the exact same model structure, so a
    # correct implementation should reproduce it closely despite the injected draws.
    assert metrics["r2"] > 0.9
    assert metrics["rmse"] < 2.0

    # validate() reports prediction error split by segment (idle vs SWW) - it must
    # not assert anything about detected/validated tap events.
    assert "suspected_draw_timesteps" not in metrics
    assert metrics["rmse_idle"] < 2.0
    assert metrics["rmse_sww"] < 2.0

    assert metrics["implausible_ua"] == 0.0


def test_validate_flags_q_in_pinned_at_calorimetric_bound(caplog):
    """Regression test for a real finding: q_in_nominal_w can land at the edge
    of its calorimetric confidence interval (see calibrate()) when the heating
    timesteps lacking a valid override disagree with the direct measurement.
    validate() must surface this the same way it already does for the UA_mix
    ceiling, since a std_error computed at a bound is not meaningful.
    """

    rng = np.random.default_rng(9)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    identifier.calibrate(df)

    # Force the exact pinned-at-edge condition directly, rather than
    # constructing a calorimetric-disagreement scenario end-to-end.
    lower = identifier.model.q_in_nominal_w
    identifier.q_in_calorimetric_bounds = (lower, lower + 1000.0)

    with caplog.at_level("WARNING", logger="features.boiler"):
        result = identifier.validate(df)

    assert result["implausible_q_in"] == 1.0
    assert any(
        "pinned at the edge of its calorimetric confidence interval" in r.message
        for r in caplog.records
    )


def test_validate_does_not_flag_q_in_away_from_calorimetric_bound(caplog):
    rng = np.random.default_rng(9)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    identifier.calibrate(df)

    # A wide interval comfortably containing the fitted value, away from
    # either edge - must not be flagged.
    center = identifier.model.q_in_nominal_w
    identifier.q_in_calorimetric_bounds = (center - 1000.0, center + 1000.0)

    with caplog.at_level("WARNING", logger="features.boiler"):
        result = identifier.validate(df)

    assert result["implausible_q_in"] == 0.0
    assert not any(
        "pinned at the edge of its calorimetric confidence interval" in r.message
        for r in caplog.records
    )


def test_calibrate_reports_excess_loss_as_unproven_candidate(caplog):
    """calibrate()'s residual-based diagnostic is parameter-identification support,
    not a tap-detection claim: it must be logged as 'candidate'/'unproven' excess
    heat loss, never as validated tap-water usage.
    """

    rng = np.random.default_rng(7)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L

    prepared = identifier.prepare(df)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    excess_loss_logs = [
        record.message
        for record in caplog.records
        if "excess heat loss" in record.message
    ]

    assert len(excess_loss_logs) == 1
    assert "NOT a validated count of tap events" in excess_loss_logs[0]
    assert "candidate data contamination" in excess_loss_logs[0]


def _bottom_top_ratio_from_log(caplog) -> float:
    excess_loss_logs = [
        record.message
        for record in caplog.records
        if "excess heat loss" in record.message
    ]
    assert len(excess_loss_logs) == 1

    match = re.search(
        r"bottom/top ratio among flagged timesteps: (-?[\d.]+)", excess_loss_logs[0]
    )
    assert match is not None

    return float(match.group(1))


def _simulate_with_frequent_bottom_heavy_draws(
    rng: np.random.Generator, days: int
) -> pd.DataFrame:
    """Frequent, multi-sample, bottom-heavy draws (cold mains water enters at the
    bottom), at a rate high enough that they clearly dominate the flagged set over
    background noise-driven flags - needed to test the asymmetry ratio in isolation,
    unlike `_simulate`'s sparse draws (6 events total) which get diluted by ordinary
    noise-driven flags in the ratio average.
    """

    heating_start_sample = 100
    heating_duration_samples = 15

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    n_samples = days * SAMPLES_PER_DAY
    state = np.array([45.0, 45.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)
    draw_remaining = 0

    for i in range(n_samples):
        sample_in_day = i % SAMPLES_PER_DAY
        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on

        if not on:
            if draw_remaining <= 0 and rng.random() < 0.008:
                draw_remaining = int(rng.integers(2, 5))
            if draw_remaining > 0:
                state = np.maximum(state - np.array([0.3, 2.0]), T_AMBIENT_C)
                draw_remaining -= 1

        states[i] = state
        u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
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
        }
    )


def test_bottom_top_asymmetry_distinguishes_draws_from_systematic_misfit(caplog):
    """A real tap draw injects cold mains water at the BOTTOM of the tank, so it
    should predominantly disturb T_bottom - unlike a systematic model/fit issue
    (e.g. the UA_top/UA_bottom split being only weakly identified), which tends to
    mispredict both nodes by a comparable amount.
    """

    rng = np.random.default_rng(21)
    df_with_draws = _simulate_with_frequent_bottom_heavy_draws(rng, days=90)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    prepared = identifier.prepare(df_with_draws)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    ratio_with_draws = _bottom_top_ratio_from_log(caplog)
    assert ratio_with_draws > 1.5

    caplog.clear()

    rng2 = np.random.default_rng(99)
    df_no_draws = _simulate_fine_resolution_with_misaligned_cycles(rng2, days=60)

    identifier_no_draws = BoilerThermalIdentifier()
    identifier_no_draws.volume_l = TRUE_VOLUME_L
    prepared_no_draws = identifier_no_draws.prepare(df_no_draws)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier_no_draws.calibrate(prepared_no_draws)

    ratio_without_draws = _bottom_top_ratio_from_log(caplog)
    assert ratio_without_draws < ratio_with_draws


def test_excess_loss_clustering_flags_multi_sample_disturbances(caplog):
    """Multi-sample disturbances (a real draw plausibly spans several 5-min samples
    plus an equilibration tail) must show up as RUNS of consecutive flagged idle
    timesteps, clearly above the chance baseline at the same flag rate - this is what
    lets calibrate()'s diagnostic distinguish "many short real disturbances" from
    isolated sensor noise, without ever claiming either is proven tap water.
    """

    rng = np.random.default_rng(11)

    heating_start_sample = 100
    heating_duration_samples = 15

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    n_samples = N_DAYS * SAMPLES_PER_DAY
    state = np.array([45.0, 45.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)
    draw_remaining = 0

    for i in range(n_samples):
        sample_in_day = i % SAMPLES_PER_DAY
        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on

        if not on:
            if draw_remaining <= 0 and rng.random() < 0.003:
                draw_remaining = int(rng.integers(2, 4))
            if draw_remaining > 0:
                state = np.maximum(state - np.array([0.5, 1.5]), T_AMBIENT_C)
                draw_remaining -= 1

        states[i] = state
        u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
    measured = states + noise

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n_samples)],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": T_AMBIENT_C,
            "T_top": measured[:, 0],
            "T_bottom": measured[:, 1],
            "state": np.where(boiler_on, "SWW", "Uit"),
        }
    )

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    prepared = identifier.prepare(df)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    excess_loss_logs = [
        record.message
        for record in caplog.records
        if "excess heat loss" in record.message
    ]
    assert len(excess_loss_logs) == 1

    match = re.search(
        r"(\d+(?:\.\d+)?)% of flagged timesteps are adjacent.*?"
        r"vs\. (\d+(?:\.\d+)?)% expected by pure chance",
        excess_loss_logs[0],
    )
    assert match is not None

    clustered_fraction = float(match.group(1))
    expected_by_chance = float(match.group(2))

    # A real, injected multi-sample disturbance pattern must show clustering well
    # above the pure-chance baseline at the same flag rate.
    assert clustered_fraction > expected_by_chance + 20.0


def _simulate_fine_resolution_with_misaligned_cycles(
    rng: np.random.Generator, days: int
) -> pd.DataFrame:
    """Simulate at 1-minute resolution with heating start/stop NOT aligned to the
    5-min sampling grid, then aggregate exactly like the real InfluxDB dataset does
    (T_top/T_bottom = mean over the window, state = last value in the window). This
    reproduces the real after-heat-tail labeling artifact: a bucket already labeled
    "Uit" (idle) whose mean temperature still reflects some of the preceding heating,
    purely from the aggregation/labeling mismatch - no injected draws at all.
    """

    fine_dt = 60.0
    bucket_seconds = DT_SECONDS
    steps_per_bucket = int(bucket_seconds / fine_dt)

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, fine_dt)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, fine_dt)

    fine_samples_per_day = int(24 * 3600 / fine_dt)
    state = np.array([47.0, 47.0])
    fine_states = []
    fine_on = []

    for _ in range(days):
        heating_start = int(rng.integers(90, 110))  # jitters off the 5-min grid
        heating_duration = int(rng.integers(12, 18))

        for minute in range(fine_samples_per_day):
            on = heating_start <= minute < heating_start + heating_duration
            fine_states.append(state)
            fine_on.append(on)
            u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
            a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
            state = a_d @ state + b_d @ u

    fine_states = np.array(fine_states)
    fine_on = np.array(fine_on)
    n_buckets = len(fine_states) // steps_per_bucket

    bucket_T = np.array(
        [
            fine_states[b * steps_per_bucket : (b + 1) * steps_per_bucket].mean(
                axis=0
            )
            for b in range(n_buckets)
        ]
    )
    bucket_on = np.array(
        [
            fine_on[b * steps_per_bucket : (b + 1) * steps_per_bucket][-1]
            for b in range(n_buckets)
        ]
    )

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=bucket_T.shape)
    measured = bucket_T + noise

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n_buckets)],
        utc=True,
    )

    return pd.DataFrame(
        {
            "time": time,
            "T_ambient": T_AMBIENT_C,
            "T_top": measured[:, 0],
            "T_bottom": measured[:, 1],
            "state": np.where(bucket_on, "SWW", "Uit"),
        }
    )


def _idle_training_denominator(caplog, identifier, prepared) -> int:
    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    messages = [
        record.message
        for record in caplog.records
        if "excess heat loss" in record.message
    ]
    assert len(messages) == 1

    match = re.search(r"\d+/(\d+) idle training timesteps", messages[0])
    assert match is not None

    return int(match.group(1))


def test_post_heating_tail_excludes_the_expected_number_of_rows(monkeypatch, caplog):
    """Regression test for a real finding: heating start/stop times don't align to
    the 5-min sampling grid, and InfluxDB labels each bucket by its LAST state while
    averaging temperature over the whole bucket - so the bucket(s) right after a real
    heating stop are labeled idle but their mean temperature still includes real
    after-heat (independently confirmed: a hand-built fine-resolution reproduction
    showed residuals of 0.14-0.31 degC at exactly these rows, well above typical idle
    noise, with NO draws injected at all).

    This test checks the exclusion mechanism itself deterministically (how many rows
    it removes), rather than the fitted parameters or flag count - both of those are
    also affected by a separate, independent UA_top/UA_bottom split ambiguity (their
    sum is better identified than the individual split when T_top and T_bottom track
    closely together), which would otherwise confound a system-level before/after
    comparison.
    """

    rng = np.random.default_rng(99)
    df = _simulate_fine_resolution_with_misaligned_cycles(rng, days=60)

    identifier_unfixed = BoilerThermalIdentifier()
    identifier_unfixed.volume_l = TRUE_VOLUME_L
    prepared = identifier_unfixed.prepare(df)

    monkeypatch.setattr(BoilerThermalIdentifier, "POST_HEATING_TAIL_SECONDS", 0.0)
    idle_count_without_fix = _idle_training_denominator(
        caplog, identifier_unfixed, prepared
    )
    caplog.clear()
    monkeypatch.undo()

    identifier_fixed = BoilerThermalIdentifier()
    identifier_fixed.volume_l = TRUE_VOLUME_L
    idle_count_with_fix = _idle_training_denominator(
        caplog, identifier_fixed, prepared
    )

    excluded_rows = idle_count_without_fix - idle_count_with_fix

    # ~60 days x ~1 heating stop/day x ~3 buckets (15 min at 5-min resolution) worth
    # of rows should be excluded - not zero (the mechanism must be active) and not
    # implausibly large (it must not eat far more than the known tail duration).
    assert 60 <= excluded_rows <= 300


def test_receding_horizon_avoids_open_loop_energy_runaway():
    """Regression test for a real-world failure: a model with weakly identified
    (near-zero) UA_top/UA_bottom - realistic for a well-insulated tank, since its
    passive-cooling time constant can be weeks long - diverges under a single
    open-loop rollout spanning many heating cycles, because the real boiler_on
    schedule is timed by draws this model cannot represent. validate()'s
    receding-horizon rollout must stay bounded despite that; only an unrealistically
    long single-window rollout should blow up.
    """

    model = BoilerThermalModel(
        volume_l=200.0,
        ua_top_w_per_k=0.10,
        ua_bottom_w_per_k=0.19,
        ua_mix_idle_w_per_k=0.05,
        ua_mix_active_w_per_k=250.0,
        q_in_nominal_w=3700.0,
    )

    heating_start_sample = 100
    heating_duration_samples = 15
    days = 14

    boiler_on = np.zeros(days * SAMPLES_PER_DAY, dtype=bool)
    for day in range(days):
        start = day * SAMPLES_PER_DAY + heating_start_sample
        boiler_on[start : start + heating_duration_samples] = True

    T_ambient = np.full(days * SAMPLES_PER_DAY, T_AMBIENT_C)
    dt_seconds = np.full(days * SAMPLES_PER_DAY, DT_SECONDS)

    # A single open-loop rollout over the whole trajectory: this is the failure mode
    # observed against real calibrated data, reproduced here deterministically.
    full_rollout = _rollout(model, 45.0, 45.0, T_ambient, boiler_on, dt_seconds)
    assert full_rollout[-1, 0] > 150.0  # confirms the runaway this test guards against

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = 200.0
    identifier.model = model
    identifier.parameter_std_errors = None

    # "Measured" ground truth: each day resets to a 45 degC baseline before heating -
    # standing in for the real draws this model cannot see, which is exactly what
    # keeps a real installation from running away the way the bare open-loop rollout
    # above does. Within a day, the trajectory still follows this same model's own
    # (well-identified) active-heating dynamics, so a correctly re-anchored short
    # window should track it closely.
    day_on = boiler_on[:SAMPLES_PER_DAY]
    day_ambient = T_ambient[:SAMPLES_PER_DAY]
    day_dt = dt_seconds[:SAMPLES_PER_DAY]

    measured = np.concatenate(
        [_rollout(model, 45.0, 45.0, day_ambient, day_on, day_dt) for _ in range(days)]
    )
    measured_top = measured[:, 0]
    measured_bottom = measured[:, 1]

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [
            start + timedelta(seconds=DT_SECONDS * i)
            for i in range(days * SAMPLES_PER_DAY)
        ],
        utc=True,
    )

    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": T_ambient,
            "T_top": measured_top,
            "T_bottom": measured_bottom,
            "state": np.where(boiler_on, "SWW", "Uit"),
        }
    )

    prepared = identifier.prepare(df)
    metrics = identifier.validate(prepared, horizon_hours=2.0)

    assert metrics["rmse"] < 5.0


def test_tail_sensitivity_sweep_shows_the_labeling_artifact_plateau(caplog):
    """The post-heating-tail sweep must show the excess-loss flag rate dropping from
    0 min toward the configured 15 min duration (the labeling artifact reproduced by
    `_simulate_fine_resolution_with_misaligned_cycles`), then roughly plateauing -
    confirming the diagnostic can actually reveal whether 15 min is enough, rather
    than always trending one way regardless of the data.
    """

    rng = np.random.default_rng(99)
    df = _simulate_fine_resolution_with_misaligned_cycles(rng, days=60)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    prepared = identifier.prepare(df)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    sweep_logs = [
        record.message
        for record in caplog.records
        if "if the post-heating tail exclusion had instead been" in record.message
    ]
    assert len(sweep_logs) == 1

    rates = dict(
        (int(minutes), float(pct))
        for minutes, pct in re.findall(r"(\d+)min=([\d.]+)%", sweep_logs[0])
    )
    assert set(rates) == set(BoilerThermalIdentifier.TAIL_SWEEP_MINUTES)

    # 0 min (no exclusion at all) must show a clearly higher flag rate than the
    # configured tail duration, since this dataset's only anomaly IS the labeling
    # artifact right after each heating stop.
    assert rates[0] > rates[15]

    # Beyond the real tail duration, further exclusion should not keep buying much -
    # a genuine plateau, not an ever-decreasing rate (which would suggest excluding
    # unboundedly more idle data "helps," which is not physically meaningful).
    assert abs(rates[45] - rates[60]) < 2.0


def test_presence_diagnostic_isolates_draws_from_confirmed_away_periods(caplog):
    """Draws injected ONLY while someone is confirmed present must show up as a much
    higher excess-loss flag rate during "present" periods than during confirmed-away
    periods - the decisive test residual shape alone cannot provide, since presence
    is an independent signal not derived from temperature at all.
    """

    rng = np.random.default_rng(31)

    heating_start_sample = 100
    heating_duration_samples = 15
    days = 90

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

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

        # Home roughly evenings/nights (samples 200-288 of each day), away
        # otherwise - draws can only be injected while marked present.
        is_present = sample_in_day >= 200
        present[i] = is_present

        if not on and is_present:
            if draw_remaining <= 0 and rng.random() < 0.03:
                draw_remaining = int(rng.integers(2, 5))
            if draw_remaining > 0:
                state = np.maximum(state - np.array([0.3, 2.0]), T_AMBIENT_C)
                draw_remaining -= 1

        states[i] = state
        u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
    measured = states + noise

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n_samples)],
        utc=True,
    )

    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": T_AMBIENT_C,
            "T_top": measured[:, 0],
            "T_bottom": measured[:, 1],
            "state": np.where(boiler_on, "SWW", "Uit"),
            "presence_0": np.where(present, "home", "not_home"),
        }
    )

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    identifier.presence_columns = ["presence_0"]
    prepared = identifier.prepare(df)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    presence_logs = [
        record.message
        for record in caplog.records
        if "nobody confirmed home" in record.message
    ]
    assert len(presence_logs) == 1

    match = re.search(r": ([\d.]+)% vs\s+([\d.]+)%", presence_logs[0])
    assert match is not None

    away_flag_rate = float(match.group(1))
    present_flag_rate = float(match.group(2))

    assert present_flag_rate > away_flag_rate + 5.0


def test_gradient_diagnostic_reports_flag_rate_by_stratification(caplog):
    """Sanity check for the top/bottom-gradient diagnostic (a real tap draw is
    not the only way a large gradient could produce an apparent excess-loss
    residual - see calibrate()'s comment on gradient-dependent mixing). Must
    run and report two well-formed flag-rate percentages without affecting the
    fit, regardless of whether this particular dataset has such an effect.
    """

    rng = np.random.default_rng(31)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    prepared = identifier.prepare(df)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    matches = [
        record.message
        for record in caplog.records
        if "starting top/bottom gradient" in record.message
    ]
    assert len(matches) == 1

    percentages = [float(x) for x in re.findall(r"(\d+\.\d)%", matches[0])]
    assert len(percentages) == 2
    assert all(0.0 <= p <= 100.0 for p in percentages)


def test_temperature_diagnostic_reports_flag_rate_by_excess_temperature(caplog):
    """Sanity check for the T-T_ambient diagnostic (a small top/bottom gradient
    often just coincides with a recently-heated, hotter tank - this isolates
    that confound from the gradient diagnostic above; see calibrate()'s
    comment on temperature-dependent heat transfer). Must run and report two
    well-formed flag-rate percentages without affecting the fit.
    """

    rng = np.random.default_rng(31)
    df = _simulate(rng)

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    prepared = identifier.prepare(df)

    with caplog.at_level("INFO", logger="features.boiler"):
        identifier.calibrate(prepared)

    matches = [
        record.message
        for record in caplog.records
        if "starting T-T_ambient" in record.message
    ]
    assert len(matches) == 1

    percentages = [float(x) for x in re.findall(r"(\d+\.\d)%", matches[0])]
    assert len(percentages) == 2
    assert all(0.0 <= p <= 100.0 for p in percentages)


def test_presence_handles_numeric_home_not_home_encoding():
    """Regression test for a real finding: this installation's device_tracker
    entities are exported via InfluxDB as a numeric 1.0/0.0 (home/away) rather than
    the literal "home"/"not_home" state string. prepare() must recognize both
    encodings - still as a whitelist (only an explicit away reading counts, never a
    missing/NaN value).
    """

    n = 30
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n)], utc=True
    )

    # First half: both trackers report numeric 0.0 (away). Second half: one tracker
    # flips to 1.0 (home) - confirmed_away must follow immediately once unsettled,
    # so only samples deep enough into the away run should end up settled.
    presence_0 = np.array([0.0] * 20 + [1.0] * 10)
    presence_1 = np.array([0.0] * 30)

    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": 20.0,
            "T_top": 45.0,
            "T_bottom": 45.0,
            "state": "Uit",
            "presence_0": presence_0,
            "presence_1": presence_1,
        }
    )

    identifier = BoilerThermalIdentifier()
    identifier.presence_columns = ["presence_0", "presence_1"]
    identifier.MIN_CONFIRMED_AWAY_SECONDS = 3 * DT_SECONDS

    prepared = identifier.prepare(df)

    # prepare() drops the first input row (its dt_seconds is undefined), so
    # prepared.iloc[j] corresponds to input row j+1. Samples 0-2 haven't settled
    # yet (< 3 samples into the away run); samples 3-18 are settled-away; samples
    # 19+ are not away at all (presence_0 went home at input row 20).
    assert not prepared["confirmed_away_settled"].iloc[:3].any()
    assert prepared["confirmed_away_settled"].iloc[3:19].all()
    assert not prepared["confirmed_away_settled"].iloc[19:].any()


def test_calorimetric_q_in_ignores_stale_flow_after_shutoff():
    """Regression test for a real finding: this installation's flow sensor does not
    report while idle, so InfluxDB's fill leaves the last active flow reading in
    place for a few buckets after state flips to "Uit" (observed directly: ~14
    L/min minutes after shutoff). q_in_override_w must be NaN (no override) for
    every idle row - including these stale-flow ones - and must match the real,
    time-varying calorimetric power (not a single average) during SWW.
    """

    n = 8
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n)], utc=True
    )

    state = ["Uit", "Uit", "SWW", "SWW", "SWW", "Uit", "Uit", "Uit"]
    # A ramping real heating cycle (rows 2-4): flow and delta-T both increase,
    # so the calorimetric power genuinely varies within the cycle - a single
    # constant Q_in_nominal could never reproduce this.
    flow_lpm = [0.0, 0.0, 10.0, 12.0, 14.0, 14.0, 14.0, 0.0]
    t_supply = [35.0, 35.0, 40.0, 45.0, 48.0, 48.0, 48.0, 35.0]
    t_return = [35.0, 35.0, 35.0, 36.0, 37.0, 48.0, 48.0, 35.0]

    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": 20.0,
            "T_top": 45.0,
            "T_bottom": 45.0,
            "state": state,
            "flow_lpm": flow_lpm,
            "T_supply": t_supply,
            "T_return": t_return,
        }
    )

    identifier = BoilerThermalIdentifier()
    prepared = identifier.prepare(df)

    # prepare() drops input row 0 (undefined dt_seconds); prepared.iloc[j] is
    # input row j+1.
    q_in_override = prepared["q_in_override_w"].to_numpy()

    # Input row 1 (idle, no flow) -> prepared index 0: no override.
    assert np.isnan(q_in_override[0])

    # Input rows 2-4 (SWW, ramping flow/deltaT) -> prepared indices 1-3: a real,
    # increasing calorimetric power, matching the formula directly.
    expected = (1.0 / 60.0) * 4186.0 * np.array([10.0 * 5.0, 12.0 * 9.0, 14.0 * 11.0])
    assert np.allclose(q_in_override[1:4], expected)
    assert np.all(np.diff(q_in_override[1:4]) > 0)  # genuinely time-varying

    # Input rows 5-6 (state="Uit" but flow stale at 14 L/min, deltaT stale at 11 K -
    # the exact real-world artifact) -> prepared indices 4-5: must NOT be used.
    assert np.isnan(q_in_override[4])
    assert np.isnan(q_in_override[5])

    # Input row 7 (idle, flow finally reset to 0) -> prepared index 6: no override.
    assert np.isnan(q_in_override[6])


def test_calorimetric_q_in_is_zero_when_supply_has_not_yet_exceeded_return():
    """Regression test for a real finding: right at a heating run's first
    sample, the compressor has just started and the refrigerant has not yet
    warmed the supply water above the tank's own return temperature -
    confirmed on real data, every T_supply<=T_return occurrence in this
    installation's calorimetric data landed exactly at elapsed=0 of its
    heating run. That is a real, valid measurement of "no net heat yet" (Q=0)
    given a real, positive flow reading - not a missing/invalid one that
    should fall back to q_in_nominal_w (the *rest* of the cycle's steady,
    well-measured average, which does not describe this specific instant).
    """

    n = 4
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n)], utc=True
    )

    state = ["Uit", "SWW", "SWW", "SWW"]
    # Row 1: the run's first heating sample - real, positive flow, but supply
    # not yet above return (compressor just started). Row 2 onward: normal
    # ramping calorimetric power.
    flow_lpm = [0.0, 10.0, 12.0, 14.0]
    t_supply = [35.0, 35.0, 40.0, 45.0]
    t_return = [35.0, 36.0, 35.0, 36.0]

    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": 20.0,
            "T_top": 45.0,
            "T_bottom": 45.0,
            "state": state,
            "flow_lpm": flow_lpm,
            "T_supply": t_supply,
            "T_return": t_return,
        }
    )

    identifier = BoilerThermalIdentifier()
    prepared = identifier.prepare(df)

    q_in_override = prepared["q_in_override_w"].to_numpy()

    # Input row 1 (SWW, T_supply<=T_return, real flow) -> prepared index 0: a
    # real, valid Q=0 measurement, not NaN.
    assert q_in_override[0] == pytest.approx(0.0)

    # Input rows 2-3 (SWW, T_supply>T_return) -> prepared indices 1-2: normal
    # positive calorimetric power, unaffected by the clip.
    assert q_in_override[1] > 0.0
    assert q_in_override[2] > 0.0


def test_flow_reporting_gap_bridges_while_active_but_zeroes_when_idle():
    """Regression test for _bridge_flow_reporting_gaps (mirrors the identical
    fix in HeatPumpCOPIdentifier): a genuine flow_lpm reporting gap (NaN -
    dataset() fetches it with no InfluxDB fill at all) must bridge forward to
    the last reading while state confirms heating is still active - real data
    showed T_supply/T_return rising smoothly through exactly such a gap
    during a DHW ramp-up - but reset to 0 the moment state reports idle
    regardless of how long ago the last reading was, the same real bug found
    for a stale flow reading persisting past a confirmed shutoff.
    """

    df = pd.DataFrame(
        {
            "state": ["SWW", "SWW", "SWW", "Uit"],
            "flow_lpm": [12.0, np.nan, 12.0, np.nan],
        }
    )

    identifier = BoilerThermalIdentifier()
    bridged = identifier._bridge_flow_reporting_gaps(df)

    assert bridged["flow_lpm"].iloc[1] == pytest.approx(12.0)
    assert bridged["flow_lpm"].iloc[3] == 0.0


def test_calibration_anchors_q_in_to_calorimetric_mean_not_free_fit():
    """Regression test for a real finding: q_in_nominal_w is only ever fit from
    the (typically few) heating timesteps lacking a valid calorimetric override
    (see _resolve_q_in) - an arbitrary, unrepresentative subset (e.g. the first
    couple of samples of a cycle before flow ramps up). Left entirely free, the
    fit can land far from the real average delivered power (observed on real
    data: fit landed at ~1615 W against a calorimetric mean of ~5450 W).
    calibrate() must instead anchor q_in_nominal_w to the calorimetric sample
    mean whenever enough direct measurements exist (see
    MIN_CALORIMETRIC_Q_IN_SAMPLES).
    """

    rng = np.random.default_rng(21)

    heating_start_sample = 100
    heating_duration_samples = 15
    flow_ramp_samples = 2  # no valid override for the first 2 samples of a cycle

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    n_samples = N_DAYS * SAMPLES_PER_DAY
    state = np.array([45.0, 45.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)
    flow_lpm = np.full(n_samples, np.nan)
    t_supply = np.full(n_samples, np.nan)
    t_return = np.full(n_samples, np.nan)

    for i in range(n_samples):
        sample_in_day = i % SAMPLES_PER_DAY
        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on
        states[i] = state

        if on and (sample_in_day - heating_start_sample) >= flow_ramp_samples:
            flow = 10.0 + rng.normal(0.0, 0.3)
            t_return[i] = 40.0 + rng.normal(0.0, 0.2)
            t_supply[i] = t_return[i] + TRUE_Q_IN_NOMINAL_W / (
                (1.0 / 60.0) * 4186.0 * flow
            )
            flow_lpm[i] = flow

        u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
    measured = states + noise

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n_samples)],
        utc=True,
    )

    df = pd.DataFrame(
        {
            "time": time,
            "T_ambient": T_AMBIENT_C,
            "T_top": measured[:, 0],
            "T_bottom": measured[:, 1],
            "state": np.where(boiler_on, "SWW", "Uit"),
            "flow_lpm": flow_lpm,
            "T_supply": t_supply,
            "T_return": t_return,
        }
    )

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    model = identifier.calibrate(df)

    # Tight tolerance: anchored to a direct measurement's own confidence
    # interval, not a free fit from a handful of unrepresentative timesteps.
    assert model.q_in_nominal_w == pytest.approx(TRUE_Q_IN_NOMINAL_W, rel=0.05)


def test_ua_mix_ceiling_binds_and_stays_within_computed_value(monkeypatch):
    """Regression test for a real finding: once a mixing conductance is large
    enough to equilibrate within one sampling interval, further increases are
    unobservable, and without an upper bound the optimizer can run away to a
    meaningless value on that flat objective (observed on real data: an unbounded
    fit drove UA_mix_active past 2.9 million W/K). Force the computed ceiling well
    below the true simulated value here, so a passing fit must be constrained by
    it rather than coincidentally landing below it.
    """

    rng = np.random.default_rng(42)
    df = _simulate(rng)  # TRUE_UA_MIX_ACTIVE_W_PER_K = 600.0

    # With volume=200L, dt=300s: c_node=418600, so a residual fraction of 0.9 gives
    # a ceiling of ~73 W/K - well below the true 600 W/K this data was generated
    # from, forcing the bound to visibly bind.
    monkeypatch.setattr(
        BoilerThermalIdentifier, "NEGLIGIBLE_MIXING_RESIDUAL_FRACTION", 0.9
    )

    identifier = BoilerThermalIdentifier()
    identifier.volume_l = TRUE_VOLUME_L
    prepared = identifier.prepare(df)
    model = identifier.calibrate(prepared)

    c_node = 1.0 * (TRUE_VOLUME_L / 2.0) * 4186.0
    expected_ceiling = (c_node / (2.0 * DT_SECONDS)) * np.log(1.0 / 0.9)

    assert model.ua_mix_active_w_per_k <= expected_ceiling + 1e-6
    assert model.ua_mix_idle_w_per_k <= expected_ceiling + 1e-6
    # Confirms the bound actually bound something here, rather than being loose.


def _simulate_presence_correlated_draws(
    rng: np.random.Generator, days: int, start: datetime
) -> pd.DataFrame:
    """Like _simulate(), but injects draws only while a synthetic presence
    tracker reports "home", every day at the same time - the pattern a
    TapForecaster is meant to learn (see features/tap.py and test_tap.py's own
    simulator, which this mirrors for boiler-resolution (5 min) data).
    """

    heating_start_sample = 100
    heating_duration_samples = 15
    present_start_sample = heating_start_sample + heating_duration_samples + 20
    present_duration_samples = 30

    a_idle, b_idle = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_IDLE_W_PER_K,
    )
    a_active, b_active = _state_space(
        TRUE_VOLUME_L,
        TRUE_UA_TOP_W_PER_K,
        TRUE_UA_BOTTOM_W_PER_K,
        TRUE_UA_MIX_ACTIVE_W_PER_K,
    )
    a_d_idle, b_d_idle = discretize_zoh(a_idle, b_idle, DT_SECONDS)
    a_d_active, b_d_active = discretize_zoh(a_active, b_active, DT_SECONDS)

    n_samples = days * SAMPLES_PER_DAY
    state = np.array([55.0, 55.0])
    states = np.empty((n_samples, 2))
    boiler_on = np.zeros(n_samples, dtype=bool)
    present = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        sample_in_day = i % SAMPLES_PER_DAY

        on = (
            heating_start_sample
            <= sample_in_day
            < heating_start_sample + heating_duration_samples
        )
        boiler_on[i] = on

        is_present = (
            present_start_sample
            <= sample_in_day
            < present_start_sample + present_duration_samples
        )
        present[i] = is_present

        if not on and is_present:
            state = np.maximum(
                state - np.array([DRAW_TOP_DROP_C, DRAW_BOTTOM_DROP_C]), T_AMBIENT_C
            )

        states[i] = state

        u = np.array([T_AMBIENT_C, TRUE_Q_IN_NOMINAL_W if on else 0.0])
        a_d, b_d = (a_d_active, b_d_active) if on else (a_d_idle, b_d_idle)
        state = a_d @ state + b_d @ u

    noise = rng.normal(0.0, MEASUREMENT_NOISE_STD_C, size=states.shape)
    measured = states + noise

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


def test_calibration_excludes_tap_forecaster_predicted_draws(tmp_path, caplog):
    """Integration test for BoilerThermalIdentifier._exclude_forecast_tap_draws:
    once a tap-demand forecaster is trained and saved alongside the boiler model,
    a later calibration run must use it to flag - and exclude - additional
    training rows it predicts as a likely real draw, on top of the same-run
    diagnostic the robust loss already relies on.
    """

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # A "prior" boiler model, calibrated on clean (draw-free) data and saved to
    # disk - TapForecaster.prepare() needs an already-calibrated model on disk
    # to compute its own training target (see features/tap.py).
    prior_df = _simulate(np.random.default_rng(101))
    prior = BoilerThermalIdentifier()
    prior.volume_l = TRUE_VOLUME_L
    prior.calibrate(prior_df)
    prior.save(tmp_path)

    # A single continuous window, presence-correlated draws throughout, used
    # both to train the tap forecaster (first 15 days) and, later, as the
    # boiler calibration window (all 30 days) - so the tap forecaster's own
    # training cutoff falls inside the calibration window's training split
    # (TRAIN_RATIO=0.8 of 30 days = 24 days > 15), matching how a real
    # installation's periodic re-calibration relates to its last tap-model fit.
    draw_start = start + timedelta(seconds=DT_SECONDS * len(prior_df))
    draw_df = _simulate_presence_correlated_draws(
        np.random.default_rng(102), days=30, start=draw_start
    )

    tap_forecaster = TapForecaster(models_path=tmp_path)
    tap_forecaster.fit(draw_df.iloc[: 15 * SAMPLES_PER_DAY])
    tap_forecaster.save(tmp_path)

    identifier = BoilerThermalIdentifier()
    identifier.presence_columns = ["presence_0"]
    identifier.load(tmp_path)  # loads the prior model, sets models_path

    with caplog.at_level("INFO", logger="features.boiler"):
        model = identifier.calibrate(draw_df)

    exclusion_logs = [
        record
        for record in caplog.records
        if "tap-demand forecaster predicts as a likely real draw" in record.message
    ]

    assert len(exclusion_logs) == 1
    excluded = int(re.search(r"excluding (\d+)", exclusion_logs[0].message).group(1))
    assert excluded > 0

    # Regression guard: excluding idle rows must never starve Q_in of its only
    # evidence (heating transitions) - the tap forecaster has no boiler_on input
    # and could otherwise flag a heating transition too (see
    # _exclude_forecast_tap_draws's idle_transition restriction).
    assert model.q_in_nominal_w == pytest.approx(TRUE_Q_IN_NOMINAL_W, rel=0.15)


def test_calibration_skips_tap_cleaning_without_a_saved_tap_model(tmp_path, caplog):
    """No tap model exists yet at models_path - calibration must proceed
    unaffected (graceful degradation), not raise.
    """

    df = _simulate(np.random.default_rng(42))

    identifier = BoilerThermalIdentifier()
    identifier.load(tmp_path)  # no boiler.joblib yet either - models_path only

    with caplog.at_level("INFO", logger="features.boiler"):
        model = identifier.calibrate(df)

    assert model.ua_top_w_per_k > 0
    assert not any(
        "tap-demand forecaster" in record.message for record in caplog.records
    )
