import logging
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd
from pvlib import solarposition
from scipy import stats

from domain.types import Config
from features.dataset import DatasetBuilder, DatasetDefinition
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)

TARGET_COLUMN = "P_solar"
EXOG_COLUMNS = ["solar_elevation"]

MIN_SOLAR_IRRADIANCE = 100.0
RETRAIN_INTERVAL_HOURS = 6
MAX_TRAIN_WINDOW_DAYS = 30

# Netherlands-ish coordinates, used only to compute solar_elevation via
# pvlib's solar-position algorithm (NREL SPA) - adjust to this
# installation's actual coordinates for the most accurate elevation-band
# assignment (see ELEVATION_BINS): the further off these are, the more a
# row can land in the wrong band, especially near sunrise/sunset where
# elevation changes fastest with time - exactly the low-elevation band the
# bias correction cares about most.
LATITUDE_DEG = 52.0
LONGITUDE_DEG = 5.0

# Solar-elevation bands (degrees), low sun/high air mass through
# near-zenith - confirmed by scripts/analyze_solar_bias.py to be where a
# real, stable bias between actual PV production and Solcast's forecast
# lives (worse at low elevation - plausibly local horizon shading or
# angle-of-incidence effects - down to near zero at high elevation).
#
# A temperature-derating term and a morning/afternoon offset were tried on
# top of this *twice* (real, documented PV physics: cell-temperature power
# loss, and a directional-obstruction hypothesis) - both times backed out
# again. The first rejection was on a single noisy window's evidence, so
# the second attempt was judged properly via SolarBiasIdentifier.validate()'s
# pooled, sample-size-weighted methodology instead - and *still* came out
# consistently worse: every one of 5 independent windows regressed (pooled
# MAE improvement dropped from +2.15% to +1.56%), the temperature
# coefficient fit to 0.0 (no signal) in 4 of 5 windows, and the
# morning/afternoon offset flipped sign between windows (a hallmark of
# fitting noise, not a real effect). This is now a properly validated
# rejection, not a premature one - revisit only alongside a way to fit
# these incrementally across retrains (e.g. a Kalman/RLS-style running
# update, so a single retrain's noise can't dominate) rather than from
# scratch every 6 hours.
ELEVATION_BINS = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]

# The live forecast curve predict_solar() corrects (state.forecast.solcast.p50)
# carries whatever spacing Solcast's own forecast attribute uses, while every
# consumer of state.predictions.solar (OptimizeConfig's fixed-dt_hours MPC
# steps, the other forecasters' 15-minute asfreq) works in 15-minute (0.25h)
# steps - predict_solar() resamples/interpolates onto that shared grid so
# those consumers never need to know the raw curve's native spacing.
PREDICT_STEP_MINUTES = 15

# dataset()'s P_solar uses fill=0 - correct for a rate-like sensor that's
# genuinely idle at night, but a daytime sensor/logging outage (HA
# restart, network/inverter fault) gets silently filled with 0 too and
# would otherwise be trusted as real zero production. A run of at least
# this many consecutive native 30-minute points pinned at exactly 0 W
# while Solcast expected daylight is treated as an outage instead (see
# scripts/detect_solar_outages.py, which uses the same threshold) - real
# cloud cover almost never drives measured output to *exactly* 0.0 for
# hours at a stretch; it varies. Confirmed on real data: a single such
# 9-hour outage was enough to visibly bias a whole backtest window.
MIN_OUTAGE_STEPS = 4

# SolarBiasIdentifier.validate()'s multi-window methodology: a single
# backtest window's improvement % is not trustworthy evidence either way
# (this correction swung from +7.1% to -9.8% between two individually
# clean 90-day windows) - so validate() instead evaluates several fixed,
# 1-week-apart windows and pools them (weighted by sample size), plus a
# significance check across the windows themselves. Confirmed real data
# only goes back to roughly mid-August for this installation (a window
# reaching further back had an implausibly short test split, or no data
# at all) - VALIDATION_WEEKS_BACK stays inside that known-good range for
# now; widen it again once more history has accumulated.
VALIDATION_WEEKS_BACK = [0, 1, 2, 3, 4]
VALIDATION_TRAIN_DAYS = 90
VALIDATION_STEPS = 48
VALIDATION_TEST_RATIO = 0.2
# Lead-time buckets (hours, start exclusive, end inclusive) validate() also
# reports separately: the MPC only executes its first step before replanning,
# so the error over the next few hours drives the actual decisions - a
# 48-step average dilutes that short-horizon error by a factor of ~10.
# validate() also reports p10/p90 coverage per bucket (see
# QUANTILE_CALIBRATED_HORIZONS).
VALIDATION_HORIZONS = [(0.0, 2.0), (2.0, 6.0), (6.0, 24.0)]
# A 10th/90th-percentile estimate rests on the ~n/10 most extreme rows of a
# lead-time bucket; below ~20 of those (n=200) it is mostly noise, so that
# bucket keeps Solcast's raw p10/p90 instead.
MIN_QUANTILE_N = 200
# Only this bucket gets p10/p90 scale factors. A per-week coverage check over
# 5 weeks found Solcast's interval at 0-2h too narrow every single week (on
# average 18.7% below p10 and 17.8% above p90, an offset about twice the
# week-to-week spread). At 2-6h the offset was smaller than that spread, at
# 6-24h coverage was on target on average, and 30-day factors fitted there
# over-widened the band out of sample. The MPC also acts on 0-2h; later
# steps get replanned many times before they are executed. Revisit 2-6h
# once there is more history.
QUANTILE_CALIBRATED_HORIZONS = [(0.0, 2.0)]
# A window with fewer daylight observations than this is excluded from the
# pooled summary (see the -4w window found during development: n=239 vs.
# 600-1300 for the others) - still logged, but too noisy to trust.
MIN_VALIDATION_WINDOW_N = 500

# A live "nowcasting" blend was also tried on top of the elevation table:
# the freshest-available Actual/Solcast ratio over the trailing hour of
# already-realized daylight production, blended into the elevation scale
# with a weight decaying exponentially (2h time constant) with lead time -
# physically motivated (cloud fields have real persistence over the next
# hour or so, but say little 4+ hours out), and checked with the same
# pooled, multi-window methodology as everything else here, on the 0-2h
# bucket where it should matter most. Rejected: at 0-2h it turned the
# elevation-only +1.70% into -3.79% (8.4 W worse on average, up to 22 W in
# one window), and it also hurt at 2-6h. Solcast's freshest forecast likely
# already reflects current cloud cover, so persisting the last hour's ratio
# double-corrects on a signal that changes within the hour. The same A/B
# found the p50-weighted median ties the plain median overall and is
# slightly more consistent at 0-2h.


def _mask_outages(df: pd.DataFrame) -> pd.DataFrame:
    """Sets P_solar to NaN for rows that look like a sensor/logging outage
    (see MIN_OUTAGE_STEPS) rather than real zero production - every
    P_solar.notna()/dropna() check elsewhere (_arguments(), validate(),
    _generate_walk_forward_folds()) then already treats these exactly like
    any other data gap, with no further special-casing needed.
    """

    df = df.copy()

    per_moment = df.sort_values(["target_time", "time"]).drop_duplicates(
        "target_time", keep="last"
    )

    suspicious = (per_moment["P_solar"] == 0.0) & (
        per_moment["p50"] >= MIN_SOLAR_IRRADIANCE
    )
    run_id = (suspicious != suspicious.shift()).cumsum()
    run_length = suspicious.groupby(run_id).transform("size")

    outage_targets = per_moment.loc[suspicious & (run_length >= MIN_OUTAGE_STEPS)][
        "target_time"
    ]

    df.loc[df["target_time"].isin(outage_targets), "P_solar"] = np.nan

    return df


def _solar_elevation(target_time: pd.Series) -> pd.Series:
    """Pure astronomical geometry - depends only on target_time and this
    installation's coordinates (LATITUDE_DEG/LONGITUDE_DEG), no forecast/
    measurement data needed. Delegates to pvlib's NREL SPA implementation
    (accurate to a small fraction of a degree, including the longitude and
    equation-of-time corrections a simpler hand-rolled formula would miss)
    rather than reimplementing solar-position astronomy here. Uses the
    apparent (refraction-corrected) elevation, since that is what
    determines when the sun is actually visible above the horizon; clipped
    to 0 below the horizon, matching ELEVATION_BINS's lowest edge. Shared
    by _prepare() (historical calibration/validation rows) and
    predict_solar() (the live forecast curve).
    """

    position = solarposition.get_solarposition(
        pd.DatetimeIndex(target_time), latitude=LATITUDE_DEG, longitude=LONGITUDE_DEG
    )

    elevation = np.clip(position["apparent_elevation"].to_numpy(), 0.0, None)

    return pd.Series(elevation, index=target_time.index)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by SolarBiasIdentifier's calibrate()/validate() - masks
    outages and derives the two columns the bias correction needs:
    lead_time_hours (a training-row filter) and solar_elevation (the
    correction's own input).
    """

    if df.empty:
        raise ValueError(
            "No data available in the requested time range - an empty "
            "range would otherwise fail later with a confusing pandas "
            "dtype error instead of this one."
        )

    df = _mask_outages(df)

    df["lead_time_hours"] = (df["target_time"] - df["time"]).dt.total_seconds() / 3600.0
    df["solar_elevation"] = _solar_elevation(df["target_time"])

    return df.sort_values(["time", "target_time"])


def _arguments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Training rows for the bias correction: a *multiplicative* target
    (Actual/Solcast), since the bias scripts/analyze_solar_bias.py found
    scales with the irradiance level itself, restricted to daylight and to
    genuinely forward-looking forecasts. Also returns p50 as the fit weight
    (see _ElevationBiasModel.fit()).
    """

    df = df.dropna(subset=[TARGET_COLUMN, "p50", *EXOG_COLUMNS]).copy()

    df = df[(df["p50"] >= MIN_SOLAR_IRRADIANCE) & (df["lead_time_hours"] >= 0.5)].copy()

    y_target = df[TARGET_COLUMN] / df["p50"]

    return df[EXOG_COLUMNS], y_target, df["p50"]


def _horizon_label(start: float, end: float) -> str:
    return f"{start:g}-{end:g}h"


COVERAGE_KEYS = ("below_p10_raw", "above_p90_raw", "below_p10", "above_p90")


def _window_summary(
    columns: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    return {
        "baseline_mae": float(np.mean(columns["baseline"][mask])),
        "mae": float(np.mean(columns["ml"][mask])),
        "n": int(mask.sum()),
        **{key: float(np.mean(columns[key][mask])) for key in COVERAGE_KEYS},
    }


def _pooled_fraction(results: list[dict], key: str) -> float:
    return sum(r[key] * r["n"] for r in results) / sum(r["n"] for r in results)


def _fit_quantile_scale(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """For each QUANTILE_CALIBRATED_HORIZONS bucket, the factors (k10, k90)
    such that 10% of actual
    daylight production falls below k10 * p10 and 10% above k90 * p90 -
    Solcast's own interval was found too narrow for this site (0-2h: only
    64% of actuals inside instead of 80%). Plain count-based quantiles of
    actual/p10 and actual/p90, since coverage counts rows rather than
    weighing Watts. Buckets with fewer than MIN_QUANTILE_N rows are left out,
    so callers fall back to Solcast's raw p10/p90 there.
    """

    rows = df.dropna(subset=[TARGET_COLUMN, "p10", "p90"])
    rows = rows[rows["p50"] >= MIN_SOLAR_IRRADIANCE]
    lead = rows["lead_time_hours"]

    scale = {}
    for start, end in QUANTILE_CALIBRATED_HORIZONS:
        bucket = rows[(lead > start) & (lead <= end)]
        if len(bucket) >= MIN_QUANTILE_N:
            scale[_horizon_label(start, end)] = (
                float((bucket[TARGET_COLUMN] / bucket["p10"]).quantile(0.10)),
                float((bucket[TARGET_COLUMN] / bucket["p90"]).quantile(0.90)),
            )

    return scale


def _quantile_scale_arrays(
    scale: dict[str, tuple[float, float]], lead_time_hours: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (k10, k90) for the given lead times - 1.0 (raw Solcast) for
    rows whose bucket has no fitted scale."""

    k10 = np.ones(lead_time_hours.size)
    k90 = np.ones(lead_time_hours.size)

    for start, end in VALIDATION_HORIZONS:
        label = _horizon_label(start, end)
        if label in scale:
            in_bucket = (lead_time_hours > start) & (lead_time_hours <= end)
            k10[in_bucket], k90[in_bucket] = scale[label]

    return k10, k90


def _improvement_pct(result: dict) -> float:
    return 100 * (result["baseline_mae"] - result["mae"]) / result["baseline_mae"]


def _pool(results: list[dict]) -> tuple[int, float, float, float]:
    """Sample-size-weighted pooling across windows (equivalent to scoring
    all their observations as one sample) - see validate()."""

    n = sum(r["n"] for r in results)
    baseline = sum(r["baseline_mae"] * r["n"] for r in results) / n
    mae = sum(r["mae"] * r["n"] for r in results) / n

    return n, baseline, mae, 100 * (baseline - mae) / baseline


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy())
    sorted_values = values.to_numpy()[order]
    cumulative = np.cumsum(weights.to_numpy()[order])

    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _get_split_time(df: pd.DataFrame, test_ratio: float) -> pd.Timestamp | None:
    if test_ratio <= 0.0 or test_ratio >= 1.0:
        return None

    unique_times = pd.Series(df["time"].unique()).sort_values()
    split_idx = int(len(unique_times) * (1.0 - test_ratio))
    return pd.Timestamp(unique_times.iloc[split_idx])


def _generate_walk_forward_folds(
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
        test_df = forecast[forecast[TARGET_COLUMN].notna()].iloc[:steps].copy()

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
                & train_slice[TARGET_COLUMN].notna()
            )
            candidate_train = train_slice[train_mask]

            if not candidate_train.empty:
                train_df = candidate_train
                last_trained_time = update_time

        yield update_time, train_df, test_df, need_retrain


class _ElevationBiasModel:
    """The correction predict_solar() applies: a fixed lookup table of the
    p50-weighted median Actual/Solcast ratio per solar-elevation band (see
    ELEVATION_BINS). A whole session of trying to model the *stochastic*
    part of Solcast's error (cloud timing/intensity) with a residual
    regression (HistGradientBoostingRegressor, then Ridge, over 20+
    engineered features, extensively tuned) never reliably beat Solcast by
    more than the noise floor - see this class's own git history.
    scripts/analyze_solar_bias.py then found a real, stable, physically
    explainable *systematic* bias instead (~5% overprediction, worse at
    low sun elevation) - a multiplicative ratio, not an additive Watt
    offset, since the bias evidently scales with the irradiance level
    itself (a calibration/installation effect: panel orientation,
    soiling, degradation, or similar - not a weather effect, so unlike the
    old residual model this needs no lead-time cutoff or decay; the
    correction applies equally at every lead time). A temperature-derating
    term and a morning/afternoon split were tried on top of this twice
    (see ELEVATION_BINS's own comment) and backed out again both times -
    this single stage remains the one component actually validated as a
    real improvement. Falls back to 1.0 (trust Solcast as-is) for a band
    with no training examples, rather than guessing.
    """

    # (k10, k90) per lead-time bucket for Solcast's own p10/p90 (see
    # _fit_quantile_scale()), kept on this calibrated object so it is saved and
    # loaded with the bias table. A class-level default, so a model saved
    # before these factors existed loads with none (raw Solcast band) instead
    # of failing every state update until the next calibration - calibrate()
    # always assigns a fresh dict and never mutates this shared one.
    quantile_scale: dict[str, tuple[float, float]] = {}

    def __init__(self, bins: list[float]) -> None:
        self.bins = bins
        self.table: pd.Series = pd.Series(dtype=float)

    def _band(self, X: pd.DataFrame) -> pd.Series:
        return pd.cut(X["solar_elevation"], bins=self.bins, include_lowest=True)

    def fit(
        self, X: pd.DataFrame, y: pd.Series, weight: pd.Series
    ) -> "_ElevationBiasModel":
        # The p50-weighted median of Actual/p50 is exactly the scale k that
        # minimizes sum|actual - k * p50| in Watts - the same absolute error
        # validate() scores - whereas a plain median would weigh a 150 W
        # dawn reading as heavily as a 3 kW midday one.
        frame = pd.DataFrame({"ratio": y, "weight": weight})
        self.table = frame.groupby(self._band(X), observed=True).apply(
            lambda group: _weighted_median(group["ratio"], group["weight"])
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # astype(float): Categorical.map() stays Categorical when the table
        # happens to map bands one-to-one, and fillna(1.0) then fails.
        return self._band(X).map(self.table).astype(float).fillna(1.0).to_numpy()


def predict_solar(model: _ElevationBiasModel, p50: pd.Series) -> pd.Series:
    """Applies the calibrated elevation-band bias correction (see
    _ElevationBiasModel) to the live, forward-looking Solcast p50 curve
    (state.forecast.solcast.p50, indexed by target time) and resamples the
    result onto the app's shared 15-minute step convention (see
    PREDICT_STEP_MINUTES) - used by StateManager.update() to build
    state.predictions.solar. `model` is whatever SolarBiasIdentifier last
    calibrated (see that class).
    """

    if p50.empty:
        return p50

    p50 = p50.sort_index()
    elevation = _solar_elevation(pd.Series(p50.index))
    scale = np.nan_to_num(
        model.predict(pd.DataFrame({"solar_elevation": elevation.to_numpy()})),
        nan=1.0,
    )
    return _on_predict_grid(pd.Series(p50.to_numpy() * scale, index=p50.index))


def predict_solar_band(
    model: _ElevationBiasModel,
    p10: pd.Series,
    p90: pd.Series,
    now: datetime,
) -> tuple[pd.Series, pd.Series]:
    """Scales Solcast's live p10/p90 curves (indexed by target time) by the
    calibrated per-lead-time factors (see _fit_quantile_scale(); buckets
    without a factor keep raw Solcast) and puts them on the same 15-minute
    grid as predict_solar() - the MPC's pessimistic/optimistic solar
    scenarios (see optimizer.SOLAR_SCENARIO_WEIGHTS).
    """

    return (
        _scaled_quantile(model.quantile_scale, p10, now, column=0),
        _scaled_quantile(model.quantile_scale, p90, now, column=1),
    )


def _scaled_quantile(
    quantile_scale: dict[str, tuple[float, float]],
    forecast: pd.Series,
    now: datetime,
    column: int,
) -> pd.Series:
    if forecast.empty:
        return forecast

    forecast = forecast.sort_index()
    lead_time_hours = (forecast.index - now).total_seconds().to_numpy() / 3600.0
    scale = _quantile_scale_arrays(quantile_scale, lead_time_hours)[column]

    return _on_predict_grid(forecast * scale)


def _on_predict_grid(series: pd.Series) -> pd.Series:
    return (
        series.clip(lower=0.0)
        .resample(f"{PREDICT_STEP_MINUTES}min")
        .interpolate(method="time")
        .clip(lower=0.0)
    )


class SolarBiasIdentifier(SystemIdentifier[_ElevationBiasModel]):
    """Calibrates and validates the elevation-band bias correction (see
    _ElevationBiasModel) that predict_solar() applies to the live Solcast
    forecast (state.predictions.solar, built by StateManager.update()).
    """

    @property
    def name(self) -> str:
        return "solar"

    @property
    def label(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

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
            .attribute_timeseries(
                "solcast",
                config.forecast.solcast,
                attributes=["p10", "p50", "p90"],
                interval="30m",
                aggregation="last",
            )
            .join(
                left="solcast",
                right="P_solar",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .build()
        )

    def calibrate(self, df: pd.DataFrame) -> _ElevationBiasModel:
        df = _prepare(df)
        # Same training window as every walk-forward fold in validate(), so
        # the model going live is the one that was actually validated.
        train_start = df["time"].max() - pd.Timedelta(days=MAX_TRAIN_WINDOW_DAYS)
        df = df[df["time"] > train_start]
        X, y, weight = _arguments(df)

        model = _ElevationBiasModel(bins=ELEVATION_BINS)
        model.fit(X, y, weight)
        model.quantile_scale = _fit_quantile_scale(df)
        self.model = model

        logger.info(
            "Fitted elevation-band ratio (p50-weighted median actual/Solcast):\n%s",
            model.table.to_string(),
        )
        logger.info(
            "Fitted p10/p90 scale per lead time (k10, k90): %s",
            ", ".join(
                f"{label}=({k10:.3f}, {k90:.3f})"
                for label, (k10, k90) in model.quantile_scale.items()
            )
            or "none (too little data - raw Solcast p10/p90 kept)",
        )

        return model

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        """Pools several fixed, weekly-spaced windows (see
        VALIDATION_WEEKS_BACK) instead of trusting any single one - a
        single window's improvement % swung from +7.1% to -9.8% between
        two individually clean 90-day windows on real data, so this is the
        only methodology that's actually been shown to give a trustworthy
        answer (see ELEVATION_BINS's own comment for the two temperature/
        phase attempts this caught as net-negative). All windows are
        sliced from the same loaded `df` - `df` needs to reach back
        VALIDATION_TRAIN_DAYS + max(VALIDATION_WEEKS_BACK) weeks for every
        window to have data; shorter windows are simply skipped.
        """

        prepared = _prepare(df).dropna(subset=[TARGET_COLUMN, "p50"])
        latest = prepared["time"].max()

        window_results = []

        for weeks in VALIDATION_WEEKS_BACK:
            cutoff = latest - pd.Timedelta(weeks=weeks)
            window_start = cutoff - pd.Timedelta(days=VALIDATION_TRAIN_DAYS)
            window_df = prepared[
                (prepared["time"] > window_start) & (prepared["time"] <= cutoff)
            ]

            try:
                result = self._validate_window(window_df)
            except ValueError as error:
                logger.warning("Skipping window ending %s: %s", cutoff, error)
                continue

            logger.info(
                "Window ending %s (-%dw): baseline=%.2f W | ML=%.2f W | "
                "improvement=%+.1f%% | n=%d",
                cutoff,
                weeks,
                result["baseline_mae"],
                result["mae"],
                _improvement_pct(result),
                result["n"],
            )

            if result["n"] < MIN_VALIDATION_WINDOW_N:
                logger.info(
                    "  excluded from pooled summary: only %d observations, "
                    "below MIN_VALIDATION_WINDOW_N=%d",
                    result["n"],
                    MIN_VALIDATION_WINDOW_N,
                )
                continue

            window_results.append(result)

        if len(window_results) < 2:
            raise ValueError(
                "Fewer than 2 trustworthy windows - cannot validate reliably "
                "(pass a larger `days` so more history is available)."
            )

        total_n, pooled_baseline, pooled_mae, improvement_pct = _pool(window_results)

        per_window_improvement = [_improvement_pct(r) for r in window_results]
        ttest = stats.ttest_1samp(per_window_improvement, popmean=0.0)

        logger.info(
            "Pooled across %d windows (n=%d): baseline=%.2f W | ML=%.2f W | "
            "improvement=%+.2f%%",
            len(window_results),
            total_n,
            pooled_baseline,
            pooled_mae,
            improvement_pct,
        )
        logger.info(
            "Per-window improvements: %s",
            ", ".join(f"{v:+.1f}%" for v in per_window_improvement),
        )
        logger.info(
            "One-sample t-test (H0: mean per-window improvement = 0): t=%.2f, p=%.3f%s",
            ttest.statistic,
            ttest.pvalue,
            " (not significant at p<0.05 - few windows, low power)"
            if ttest.pvalue >= 0.05
            else " (significant at p<0.05)",
        )

        summary = {
            "mae": pooled_mae,
            "baseline_mae": pooled_baseline,
            "improvement_pct": improvement_pct,
            "n": float(total_n),
            "windows": float(len(window_results)),
            "t_statistic": float(ttest.statistic),
            "p_value": float(ttest.pvalue),
        }

        for start, end in VALIDATION_HORIZONS:
            label = _horizon_label(start, end)
            horizon_results = [
                r["horizons"][label] for r in window_results if label in r["horizons"]
            ]
            if not horizon_results:
                continue

            n, baseline, mae, improvement = _pool(horizon_results)
            logger.info(
                "  lead %s: baseline=%.2f W | ML=%.2f W | improvement=%+.2f%% | "
                "n=%d | per window: %s",
                label,
                baseline,
                mae,
                improvement,
                n,
                ", ".join(f"{_improvement_pct(h):+.1f}%" for h in horizon_results),
            )

            coverage = {
                key: 100 * _pooled_fraction(horizon_results, key)
                for key in COVERAGE_KEYS
            }
            logger.info(
                "  lead %s coverage (target 10%% each side): below p10 "
                "raw=%.1f%% -> calibrated=%.1f%% | above p90 raw=%.1f%% -> "
                "calibrated=%.1f%% | calibrated inside per window: %s",
                label,
                coverage["below_p10_raw"],
                coverage["below_p10"],
                coverage["above_p90_raw"],
                coverage["above_p90"],
                ", ".join(
                    f"{100 * (1 - h['below_p10'] - h['above_p90']):.0f}%"
                    for h in horizon_results
                ),
            )

            summary[f"improvement_pct_{label}"] = improvement
            summary.update({f"{key}_pct_{label}": v for key, v in coverage.items()})

        return summary

    def _validate_window(self, window_df: pd.DataFrame) -> dict[str, float]:
        """Runs the walk-forward backtest (see _generate_walk_forward_folds)
        over one window and returns its baseline/ML MAE, sample count and
        p10/p90 coverage (raw Solcast vs. scaled by the quantile factors
        fitted on each fold's own training data - so the calibrated coverage
        is measured on data those factors never saw). See validate() for why
        several of these get pooled rather than trusted individually.
        """

        split_time = _get_split_time(window_df, VALIDATION_TEST_RATIO)

        chunks: dict[str, list[np.ndarray]] = {
            key: [] for key in ("baseline", "ml", "lead", *COVERAGE_KEYS)
        }
        model = None
        quantile_scale: dict[str, tuple[float, float]] = {}

        folds = _generate_walk_forward_folds(window_df, steps=VALIDATION_STEPS)

        for update_time, train_df, test_df, need_retrain in folds:
            if need_retrain and not train_df.empty:
                try:
                    X_train, y_train, weight_train = _arguments(train_df)
                    if not X_train.empty:
                        model = _ElevationBiasModel(bins=ELEVATION_BINS)
                        model.fit(X_train, y_train, weight_train)
                        quantile_scale = _fit_quantile_scale(train_df)
                except ValueError:
                    continue

            if split_time is not None and update_time < split_time:
                continue

            if model is None:
                continue

            test_clean = test_df.dropna(subset=EXOG_COLUMNS).copy()
            if test_clean.empty:
                continue

            scale = np.nan_to_num(model.predict(test_clean[EXOG_COLUMNS]), nan=1.0)
            p10 = test_clean["p10"].to_numpy()
            p50 = test_clean["p50"].to_numpy()
            p90 = test_clean["p90"].to_numpy()
            pred = np.maximum(p50 * scale, 0.0)
            actual = test_clean[TARGET_COLUMN].to_numpy()
            lead = test_clean["lead_time_hours"].to_numpy()
            k10, k90 = _quantile_scale_arrays(quantile_scale, lead)

            daylight = p50 >= MIN_SOLAR_IRRADIANCE
            for key, values in (
                ("baseline", np.abs(actual - p50)),
                ("ml", np.abs(actual - pred)),
                ("lead", lead),
                ("below_p10_raw", actual < p10),
                ("above_p90_raw", actual > p90),
                ("below_p10", actual < k10 * p10),
                ("above_p90", actual > k90 * p90),
            ):
                chunks[key].append(values[daylight])

        columns = {
            key: np.concatenate([np.array([]), *parts]) for key, parts in chunks.items()
        }

        if columns["ml"].size == 0:
            raise ValueError("No validation data available in this window.")

        lead = columns["lead"]
        horizons = {}
        for start, end in VALIDATION_HORIZONS:
            mask = (lead > start) & (lead <= end)
            if mask.any():
                horizons[_horizon_label(start, end)] = _window_summary(columns, mask)

        overall = _window_summary(columns, np.ones(lead.size, dtype=bool))

        return {**overall, "horizons": horizons}
