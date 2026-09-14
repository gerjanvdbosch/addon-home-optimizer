from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from features.solar import (
    ELEVATION_BINS,
    MAX_TRAIN_WINDOW_DAYS,
    MIN_OUTAGE_STEPS,
    MIN_QUANTILE_N,
    MIN_SOLAR_IRRADIANCE,
    MIN_VALIDATION_WINDOW_N,
    PREDICT_STEP_MINUTES,
    SolarBiasIdentifier,
    _arguments,
    _ElevationBiasModel,
    _fit_quantile_scale,
    _generate_walk_forward_folds,
    _mask_outages,
    _prepare,
    _quantile_scale_arrays,
    _solar_elevation,
    predict_solar,
    predict_solar_band,
)

LATITUDE = 52.0
LONGITUDE = 5.0


def test_solar_elevation_matches_expected_value_at_solstice_noon():
    """Sanity check against pvlib: at summer solstice, solar-noon elevation
    should be close to 90 - latitude + solar declination (~23.44 deg) -
    confirms the coordinates and pvlib wiring are correct, not just that
    the function runs. Solar noon at LONGITUDE (11:40 UTC, not 12:00) is
    used here so this isn't sensitive to that same longitude correction.
    """

    solar_noon = pd.Series([datetime(2026, 6, 21, 11, 40, tzinfo=UTC)])

    elevation = _solar_elevation(solar_noon, LATITUDE, LONGITUDE)

    expected_max = 90.0 - LATITUDE + 23.44
    assert elevation.iloc[0] == pytest.approx(expected_max, abs=1.0)


def test_solar_elevation_is_clipped_to_zero_below_the_horizon():
    night = pd.Series([datetime(2026, 6, 21, 0, 0, tzinfo=UTC)])

    elevation = _solar_elevation(night, LATITUDE, LONGITUDE)

    assert elevation.iloc[0] == pytest.approx(0.0)


def test_solar_elevation_preserves_row_alignment_with_duplicate_timestamps():
    """Multiple forecast-snapshot rows can share the same target_time (see
    the attribute_timeseries dataset) - the returned elevation must align
    back to each row by its own Series index, not silently deduplicate or
    reorder rows.
    """

    target_time = pd.Series(
        [
            datetime(2026, 6, 21, 12, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 6, 21, 12, 0, tzinfo=UTC),
        ],
        index=[5, 7, 9],
    )

    elevation = _solar_elevation(target_time, LATITUDE, LONGITUDE)

    assert list(elevation.index) == [5, 7, 9]
    assert elevation.loc[5] == pytest.approx(elevation.loc[9])
    assert elevation.loc[7] == pytest.approx(0.0)


def _outage_df(n_zero_steps: int, p50: float) -> pd.DataFrame:
    """n_zero_steps consecutive native 30-minute moments reading exactly
    P_solar=0.0 at the given p50 - one row per target_time, `time` an hour
    earlier (irrelevant to _mask_outages beyond being present)."""

    base = datetime(2026, 6, 1, 12, tzinfo=UTC)
    target_times = [base + timedelta(minutes=30 * i) for i in range(n_zero_steps)]

    return pd.DataFrame(
        {
            "time": [t - timedelta(hours=1) for t in target_times],
            "target_time": target_times,
            "P_solar": 0.0,
            "p50": p50,
        }
    )


def test_mask_outages_nans_a_long_daylight_zero_run():
    """A run of at least MIN_OUTAGE_STEPS consecutive exact-zero readings
    while Solcast expected daylight looks like a sensor/logging outage
    (real cloud cover essentially never holds output at *exactly* 0.0 for
    that long) - must be treated as missing data, not real production.
    """

    df = _outage_df(MIN_OUTAGE_STEPS, p50=MIN_SOLAR_IRRADIANCE + 200.0)

    masked = _mask_outages(df)

    assert masked["P_solar"].isna().all()


def test_mask_outages_leaves_a_short_zero_run_alone():
    """A run shorter than MIN_OUTAGE_STEPS is left as real data - only a
    sustained run is treated as an outage, to avoid false positives on a
    plausible brief reading."""

    df = _outage_df(MIN_OUTAGE_STEPS - 1, p50=MIN_SOLAR_IRRADIANCE + 200.0)

    masked = _mask_outages(df)

    assert not masked["P_solar"].isna().any()


def test_mask_outages_leaves_nighttime_zeros_alone():
    """Zero production while Solcast itself expected no meaningful
    daylight (p50 below MIN_SOLAR_IRRADIANCE) is genuine nighttime idle,
    however long the run - fill=0 is correct there and must not be
    second-guessed."""

    df = _outage_df(MIN_OUTAGE_STEPS + 10, p50=MIN_SOLAR_IRRADIANCE - 50.0)

    masked = _mask_outages(df)

    assert not masked["P_solar"].isna().any()


def test_elevation_bias_model_recovers_a_known_ratio_per_band():
    """The correction predict_solar() applies: fit() must learn the median
    Actual/Solcast ratio per elevation band (see ELEVATION_BINS), and
    predict() must apply the right band's ratio to each row.
    """

    # Two clearly separated bands with different, known true ratios - low
    # elevation biased low (0.8x), high elevation nearly unbiased (1.0x),
    # matching the real pattern scripts/analyze_solar_bias.py found.
    X = pd.DataFrame({"solar_elevation": [5.0, 5.0, 5.0, 70.0, 70.0, 70.0]})
    y = pd.Series([0.8, 0.79, 0.81, 1.0, 1.01, 0.99])

    model = _ElevationBiasModel(bins=ELEVATION_BINS)
    model.fit(X, y, pd.Series(500.0, index=X.index))

    predicted = model.predict(X)

    assert predicted[:3] == pytest.approx(0.8, abs=0.02)
    assert predicted[3:] == pytest.approx(1.0, abs=0.02)


def test_elevation_bias_model_weights_ratios_by_forecast_power():
    """The fitted scale must minimize absolute error in Watts, so bright
    high-p50 readings outweigh a majority of dim low-p50 ones - a plain
    median would return 0.5 here.
    """

    X = pd.DataFrame({"solar_elevation": [10.0] * 5})
    y = pd.Series([0.5, 0.5, 0.5, 0.9, 0.9])
    weight = pd.Series([150.0, 150.0, 150.0, 3000.0, 3000.0])

    model = _ElevationBiasModel(bins=ELEVATION_BINS).fit(X, y, weight)

    assert model.predict(X)[0] == pytest.approx(0.9)


def test_elevation_bias_model_falls_back_to_no_correction_for_an_unseen_band():
    """A band with zero training examples must not be guessed at - fall
    back to 1.0 (trust Solcast as-is), not extrapolate or raise.
    """

    model = _ElevationBiasModel(bins=ELEVATION_BINS)
    model.fit(
        pd.DataFrame({"solar_elevation": [5.0]}),
        pd.Series([0.8]),
        pd.Series([500.0]),
    )

    # 70 degrees falls in a band with no training examples.
    predicted = model.predict(pd.DataFrame({"solar_elevation": [70.0]}))

    assert predicted[0] == pytest.approx(1.0)


def test_elevation_bias_model_mixes_known_and_unseen_bands():
    """Regression: with seen bands mapping one-to-one plus an unseen band,
    pandas keeps the mapped result Categorical and fillna(1.0) used to
    raise instead of falling back to no correction.
    """

    model = _ElevationBiasModel(bins=ELEVATION_BINS).fit(
        pd.DataFrame({"solar_elevation": [5.0, 20.0]}),
        pd.Series([0.8, 0.9]),
        pd.Series([500.0, 500.0]),
    )

    predicted = model.predict(pd.DataFrame({"solar_elevation": [5.0, 20.0, 70.0]}))

    assert predicted == pytest.approx([0.8, 0.9, 1.0])


def test_arguments_targets_the_actual_to_solcast_ratio():
    """_arguments() must build a *multiplicative* target (Actual/Solcast),
    not an additive residual - the bias scripts/analyze_solar_bias.py found
    scales with the irradiance level itself, and must filter out
    below-daylight-floor and same-or-backward-in-time rows.
    """

    df = pd.DataFrame(
        {
            "P_solar": [400.0, 45.0, 300.0],
            "p50": [500.0, 50.0, 500.0],
            "solar_elevation": [20.0, 5.0, 25.0],
            # Row 0: normal daylight row, kept.
            # Row 1: below MIN_SOLAR_IRRADIANCE, excluded.
            # Row 2: lead_time_hours < 0.5, excluded.
            "lead_time_hours": [1.0, 1.0, 0.25],
        }
    )

    X, y, weight = _arguments(df)

    assert list(X.index) == [0]
    assert y.tolist() == pytest.approx([0.8])
    assert weight.tolist() == pytest.approx([500.0])


def test_prepare_raises_a_clear_error_for_an_empty_range():
    """An empty input (no data at all for the requested time range - e.g.
    a validation window reaching before this installation's data
    collection started) must fail with a clear, actionable error, not the
    confusing pandas "Can only use .dt accessor" AttributeError that empty
    datetime subtraction produces further down in this same function.
    """

    empty = pd.DataFrame(columns=["time", "target_time", "P_solar", "p50"])

    with pytest.raises(ValueError, match="No data available"):
        _prepare(empty, LATITUDE, LONGITUDE)


def _synthetic_walk_forward_df(n_issue_times: int = 20, max_lead_steps: int = 10):
    """n_issue_times, 30 minutes apart, each carrying forecast rows for the
    next max_lead_steps half-hour target times (up to 5h lead time) - long
    enough that some rows' target_time falls after several later folds' own
    update_time, which is exactly the case _generate_walk_forward_folds' own
    target_time < update_time guard exists to handle (see its docstring/
    comments). All target_times here have a known P_solar, mimicking real
    historical backtest data where actuals are known throughout the window.
    """

    base = datetime(2026, 1, 1, tzinfo=UTC)
    issue_times = [base + timedelta(minutes=30 * i) for i in range(n_issue_times)]

    return pd.DataFrame(
        [
            dict(
                time=t,
                target_time=t + timedelta(minutes=30 * k),
                P_solar=500.0,
                p50=500.0,
            )
            for t in issue_times
            for k in range(1, max_lead_steps + 1)
        ]
    )


def test_generate_walk_forward_folds_never_trains_on_future_outcomes():
    """The core anti-leakage invariant: every training row's outcome
    (target_time) must be strictly before the fold's own decision point
    (update_time) - a model must never be fit on a row whose real-world
    outcome wasn't yet known at that point in the walk, even when that
    row's own issue time is safely in the past (a forecast issued early but
    targeting far into the future).
    """

    df = _synthetic_walk_forward_df()

    folds = list(_generate_walk_forward_folds(df, steps=6))
    assert any(not train_df.empty for _, train_df, _, _ in folds)

    for update_time, train_df, _test_df, _need_retrain in folds:
        if not train_df.empty:
            assert (train_df["target_time"] < update_time).all()


def test_generate_walk_forward_folds_test_window_is_strictly_future():
    """Every scored row's target_time must be strictly after the fold's own
    decision point - a fold must never be scored against an outcome that
    was already known before the forecast was made.
    """

    df = _synthetic_walk_forward_df()

    folds = list(_generate_walk_forward_folds(df, steps=6))
    assert any(not test_df.empty for _, _, test_df, _ in folds)

    for update_time, _train_df, test_df, _need_retrain in folds:
        if not test_df.empty:
            assert (test_df["target_time"] > update_time).all()


def test_calibrate_fits_and_stores_the_elevation_bias_model():
    """SolarBiasIdentifier.calibrate() is the production calibration path
    (POST /api/calibrate, target=solar) - it must run the full
    _prepare()/_arguments() pipeline, fit an _ElevationBiasModel, and store
    it on self.model so save() can persist it.
    """

    # Summer-solstice noon at LATITUDE=52 puts the sun well above the
    # horizon - any elevation band with real training rows is fine here,
    # the exact band split is already covered by
    # test_elevation_bias_model_recovers_a_known_ratio_per_band.
    base_target = datetime(2026, 6, 21, 12, tzinfo=UTC)
    issue_time = base_target - timedelta(hours=2)
    target_times = [base_target + timedelta(minutes=30 * i) for i in range(3)]

    df = pd.DataFrame(
        {
            "time": [issue_time] * 3,
            "target_time": target_times,
            "P_solar": [400.0, 404.0, 396.0],
            "p10": [300.0, 300.0, 300.0],
            "p50": [500.0, 500.0, 500.0],
            "p90": [700.0, 700.0, 700.0],
        }
    )

    identifier = SolarBiasIdentifier(LATITUDE, LONGITUDE)
    model = identifier.calibrate(df)

    assert identifier.model is model
    assert not model.table.empty

    prepared = _prepare(df.copy(), LATITUDE, LONGITUDE)
    predicted = model.predict(prepared[["solar_elevation"]])
    assert predicted == pytest.approx(0.8, abs=0.02)


def test_calibrate_trains_on_the_same_window_validate_evaluates():
    """validate()'s walk-forward folds train on at most
    MAX_TRAIN_WINDOW_DAYS - calibrate() must use that same window, or the
    model going live is not the one that was validated.
    """

    recent_target = datetime(2026, 6, 21, 12, tzinfo=UTC)
    old_target = recent_target - timedelta(days=MAX_TRAIN_WINDOW_DAYS + 30)

    df = pd.DataFrame(
        {
            "time": [t - timedelta(hours=2) for t in (old_target, recent_target)],
            "target_time": [old_target, recent_target],
            "P_solar": [250.0, 400.0],
            "p10": [300.0, 300.0],
            "p50": [500.0, 500.0],
            "p90": [700.0, 700.0],
        }
    )

    model = SolarBiasIdentifier(LATITUDE, LONGITUDE).calibrate(df)

    assert model.table.to_numpy() == pytest.approx([0.8])


def _quantile_rows(n: int, lead_time_hours: float) -> pd.DataFrame:
    """n daylight rows in one lead-time bucket, actuals evenly spread from
    0 to 2000 W - wider on both sides than a fixed p10/p50/p90 of
    500/1000/1500 W, i.e. an interval that is too narrow."""

    return pd.DataFrame(
        {
            "P_solar": np.linspace(0.0, 2000.0, n),
            "p10": 500.0,
            "p50": 1000.0,
            "p90": 1500.0,
            "lead_time_hours": lead_time_hours,
        }
    )


def test_fit_quantile_scale_makes_each_tail_hold_ten_percent():
    """The fitted factors must turn Solcast's interval into one where 10%
    of actuals fall below k10 * p10 and 10% above k90 * p90 - the
    definition of p10/p90 - here for a band that is too narrow (actuals
    spread wider than 500-1500 W).
    """

    rows = _quantile_rows(MIN_QUANTILE_N * 2, lead_time_hours=1.0)

    scale = _fit_quantile_scale(rows)

    assert set(scale) == {"0-2h"}
    k10, k90 = scale["0-2h"]
    assert k10 < 1.0 < k90
    assert (rows["P_solar"] < k10 * rows["p10"]).mean() == pytest.approx(0.10, abs=0.01)
    assert (rows["P_solar"] > k90 * rows["p90"]).mean() == pytest.approx(0.10, abs=0.01)


def test_fit_quantile_scale_skips_buckets_with_too_little_data():
    """Tail quantiles from a handful of rows are noise - such a bucket must
    keep raw Solcast p10/p90 rather than a factor fitted on it."""

    rows = _quantile_rows(MIN_QUANTILE_N - 1, lead_time_hours=1.0)

    assert _fit_quantile_scale(rows) == {}


def test_fit_quantile_scale_only_calibrates_the_0_2h_bucket():
    """2-6h and 6-24h keep Solcast's raw p10/p90 even with plenty of data -
    see QUANTILE_CALIBRATED_HORIZONS."""

    rows = pd.concat(
        [
            _quantile_rows(MIN_QUANTILE_N * 2, lead_time_hours=lead)
            for lead in (1.0, 3.0, 12.0)
        ]
    )

    assert set(_fit_quantile_scale(rows)) == {"0-2h"}


def test_quantile_scale_arrays_maps_by_lead_time_and_defaults_to_raw():
    k10, k90 = _quantile_scale_arrays({"0-2h": (0.9, 1.1)}, np.array([1.0, 3.0]))

    assert k10 == pytest.approx([0.9, 1.0])
    assert k90 == pytest.approx([1.1, 1.0])


def _synthetic_solar_df(n_rows: int = 200) -> pd.DataFrame:
    """A minimal, non-empty dataset for validate()'s window-slicing loop -
    the pooling/statistics tests below monkeypatch _validate_window itself,
    so only needs to survive _prepare()/dropna() without being empty.
    """

    base = datetime(2026, 1, 1, tzinfo=UTC)
    times = [base + timedelta(minutes=30 * i) for i in range(n_rows)]

    return pd.DataFrame(
        {
            "time": times,
            "target_time": [t + timedelta(minutes=30) for t in times],
            "P_solar": 400.0,
            "p50": 500.0,
        }
    )


def test_validate_raises_when_fewer_than_two_windows_are_trustworthy():
    """Real data for this installation only reaches back so far - older
    windows legitimately have no data at all. validate() must refuse to
    report a pooled result from a single window rather than silently
    presenting one unreliable number as if it were the properly-pooled
    methodology (see the +7.1%/-9.8% single-window swing this replaced).
    """

    identifier = SolarBiasIdentifier(LATITUDE, LONGITUDE)
    df = _synthetic_solar_df()

    def fake_validate_window(self, window_df):
        raise ValueError("No validation data available in this window.")

    SolarBiasIdentifier._validate_window = fake_validate_window
    try:
        with pytest.raises(ValueError, match="Fewer than 2"):
            identifier.validate(df)
    finally:
        del SolarBiasIdentifier._validate_window


def test_validate_pools_windows_weighted_by_sample_size():
    """The pooled MAE/improvement must be weighted by each window's sample
    size, not averaged window-by-window - an unweighted average was found
    to misstate the true effect (a large, representative window and a
    small, noisy one must not count equally).
    """

    kept = [
        {"baseline_mae": 100.0, "mae": 90.0, "n": MIN_VALIDATION_WINDOW_N + 500},
        {"baseline_mae": 200.0, "mae": 190.0, "n": MIN_VALIDATION_WINDOW_N + 2500},
        {"baseline_mae": 120.0, "mae": 110.0, "n": MIN_VALIDATION_WINDOW_N + 100},
    ]
    coverage = {"below_p10_raw": 0.2, "above_p90_raw": 0.15, "above_p90": 0.1}
    short_horizon = [
        {"baseline_mae": 80.0, "mae": 60.0, "n": 100, "below_p10": 0.10, **coverage},
        {"baseline_mae": 120.0, "mae": 118.0, "n": 300, "below_p10": 0.20, **coverage},
        {"baseline_mae": 90.0, "mae": 81.0, "n": 50, "below_p10": 0.06, **coverage},
    ]
    for window, horizon in zip(kept, short_horizon, strict=True):
        window["horizons"] = {"0-2h": horizon}
    excluded = [
        {"baseline_mae": 50.0, "mae": 60.0, "n": MIN_VALIDATION_WINDOW_N - 400},
        {"baseline_mae": 80.0, "mae": 82.0, "n": MIN_VALIDATION_WINDOW_N - 490},
    ]
    # Interleaved so the excluded, too-small windows sit among the kept
    # ones - order must not matter to the pooling.
    canned = iter([kept[0], excluded[0], kept[1], excluded[1], kept[2]])

    def fake_validate_window(self, window_df):
        return next(canned)

    identifier = SolarBiasIdentifier(LATITUDE, LONGITUDE)
    df = _synthetic_solar_df()

    SolarBiasIdentifier._validate_window = fake_validate_window
    try:
        result = identifier.validate(df)
    finally:
        del SolarBiasIdentifier._validate_window

    total_n = sum(r["n"] for r in kept)
    expected_baseline = sum(r["baseline_mae"] * r["n"] for r in kept) / total_n
    expected_mae = sum(r["mae"] * r["n"] for r in kept) / total_n
    expected_improvement = (
        100 * (expected_baseline - expected_mae) / expected_baseline
    )

    assert result["windows"] == pytest.approx(3.0)
    assert result["n"] == pytest.approx(total_n)
    assert result["baseline_mae"] == pytest.approx(expected_baseline)
    assert result["mae"] == pytest.approx(expected_mae)
    assert result["improvement_pct"] == pytest.approx(expected_improvement)
    assert np.isfinite(result["t_statistic"])
    assert 0.0 <= result["p_value"] <= 1.0

    short_n = sum(h["n"] for h in short_horizon)
    short_baseline = sum(h["baseline_mae"] * h["n"] for h in short_horizon) / short_n
    short_mae = sum(h["mae"] * h["n"] for h in short_horizon) / short_n
    assert result["improvement_pct_0-2h"] == pytest.approx(
        100 * (short_baseline - short_mae) / short_baseline
    )
    assert "improvement_pct_2-6h" not in result
    # Coverage pools by sample size too: (0.10*100 + 0.20*300 + 0.06*50) / 450.
    assert result["below_p10_pct_0-2h"] == pytest.approx(100 * 73.0 / 450.0)
    assert result["below_p10_raw_pct_0-2h"] == pytest.approx(20.0)


class _UnitScaleModel:
    """Stand-in for a fitted _ElevationBiasModel - predict_solar()'s own
    resampling/interpolation logic is independent of the model's fit
    quality (see the repo's identification/validation separation
    convention), so a trivial stub (always "trust Solcast as-is",
    scale=1.0) is enough to exercise it without a real training pass.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))


def test_predict_solar_applies_multiplicative_scale_and_clips_negatives():
    """predict_solar() must scale p50 by the model's predicted ratio, not
    add a residual - and never return a negative production value.
    """

    now = datetime(2026, 6, 21, 12, tzinfo=UTC)

    class _HalfScaleModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), 0.5)

    p50 = pd.Series(
        [500.0, 500.0],
        index=pd.DatetimeIndex([now, now + timedelta(minutes=30)]),
    )

    result = predict_solar(_HalfScaleModel(), p50, LATITUDE, LONGITUDE)

    assert not result.empty
    assert (result >= 0.0).all()
    assert result.iloc[0] == pytest.approx(250.0)


def test_predict_solar_resamples_onto_the_shared_15_minute_grid():
    """Every consumer of state.predictions.solar (the MPC optimizer's
    fixed-dt_hours steps, the dashboard chart) expects the app's shared
    15-minute step convention, regardless of whatever native spacing
    Solcast's own forecast attribute happens to use.
    """

    base = datetime(2026, 6, 21, 12, tzinfo=UTC)
    p50 = pd.Series(
        [500.0, 520.0, 540.0],
        index=pd.DatetimeIndex([base + timedelta(minutes=30 * i) for i in range(3)]),
    )

    result = predict_solar(_UnitScaleModel(), p50, LATITUDE, LONGITUDE)

    assert not result.isna().any()
    diffs = np.diff(result.index.values)
    assert all(diff == np.timedelta64(PREDICT_STEP_MINUTES, "m") for diff in diffs)


def test_predict_solar_band_scales_only_calibrated_lead_times():
    """The live band must apply the calibrated factor only where one was
    fitted (0-2h here) and keep raw Solcast further out, on the same
    15-minute grid predict_solar() uses."""

    now = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    times = pd.DatetimeIndex([now + timedelta(hours=h) for h in (1.0, 1.5, 3.0, 3.5)])
    model = _ElevationBiasModel(bins=ELEVATION_BINS)
    model.quantile_scale = {"0-2h": (0.8, 1.2)}

    low, high = predict_solar_band(
        model, pd.Series(1000.0, index=times), pd.Series(2000.0, index=times), now
    )

    assert low[times[0]] == pytest.approx(800.0)
    assert high[times[0]] == pytest.approx(2400.0)
    assert low[times[2]] == pytest.approx(1000.0)
    assert high[times[2]] == pytest.approx(2000.0)
    assert all(
        diff == np.timedelta64(PREDICT_STEP_MINUTES, "m")
        for diff in np.diff(low.index.values)
    )


def test_uncalibrated_model_keeps_the_raw_solcast_band():
    """A model saved before quantile factors existed has none - the band must
    fall back to raw Solcast p10/p90, not fail."""

    now = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    times = pd.DatetimeIndex([now + timedelta(hours=1.0), now + timedelta(hours=1.5)])

    low, high = predict_solar_band(
        _ElevationBiasModel(bins=ELEVATION_BINS),
        pd.Series(1000.0, index=times),
        pd.Series(2000.0, index=times),
        now,
    )

    assert low[times[0]] == pytest.approx(1000.0)
    assert high[times[0]] == pytest.approx(2000.0)


def test_predict_solar_covers_the_last_native_period():
    """Solcast values are period averages starting at their timestamp - the
    last 30-minute value must also fill its period's final 15-minute step, so
    a day whose last value is at 23:30 still has a 23:45 prediction."""

    base = datetime(2026, 6, 21, 23, 0, tzinfo=UTC)
    p50 = pd.Series(
        [400.0, 200.0],
        index=pd.DatetimeIndex([base, base + timedelta(minutes=30)]),
    )

    result = predict_solar(_UnitScaleModel(), p50, LATITUDE, LONGITUDE)

    assert result.index[-1] == base + timedelta(minutes=45)
    assert result.iloc[-1] == pytest.approx(200.0)


def test_predict_solar_returns_empty_series_for_empty_input():
    """No live Solcast forecast yet (fresh install, forecast fetch not run)
    must return an empty series, not raise - StateManager treats an empty
    result as "leave any existing prediction untouched".
    """

    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    result = predict_solar(_UnitScaleModel(), empty, LATITUDE, LONGITUDE)

    assert result.empty
