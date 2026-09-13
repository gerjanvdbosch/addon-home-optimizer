import logging

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from domain.types import BoilerThermalModel, Config
from features.dataset import DatasetBuilder, DatasetDefinition
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)

# Physical constants (water), not fit parameters.
RHO_WATER_KG_PER_L = 1.0
CP_WATER_J_PER_KG_K = 4186.0


def _state_space(
    volume_l: float,
    ua_top_w_per_k: float,
    ua_bottom_w_per_k: float,
    ua_mix_w_per_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space matrices for dx/dt = A x + B u.

    x = [T_top, T_bottom], u = [T_ambient, Q_in]. Equal top/bottom volume split is the
    simplest unbiased assumption available: there is no sensor for the thermocline
    position, so both nodes share one capacity C_node.
    """

    c_node = RHO_WATER_KG_PER_L * (volume_l / 2.0) * CP_WATER_J_PER_KG_K

    a = (
        np.array(
            [
                [-(ua_mix_w_per_k + ua_top_w_per_k), ua_mix_w_per_k],
                [ua_mix_w_per_k, -(ua_mix_w_per_k + ua_bottom_w_per_k)],
            ]
        )
        / c_node
    )

    b = (
        np.array(
            [
                [ua_top_w_per_k, 0.0],
                [ua_bottom_w_per_k, 1.0],
            ]
        )
        / c_node
    )

    return a, b


def discretize_zoh(
    a: np.ndarray,
    b: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization via the matrix exponential (Van Loan).

    Keeps the discrete step physically identical to the continuous ODE for any dt,
    instead of introducing Euler-integration error.
    """

    n, m = b.shape

    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = a
    augmented[:n, n:] = b

    exponent = expm(augmented * dt_seconds)

    return exponent[:n, :n], exponent[:n, n:]


def _model_from_parameters(
    parameters: np.ndarray,
    volume_l: float,
) -> BoilerThermalModel:
    ua_top, ua_bottom, ua_mix_idle, ua_mix_active, q_in_nominal = parameters

    return BoilerThermalModel(
        volume_l=volume_l,
        ua_top_w_per_k=float(ua_top),
        ua_bottom_w_per_k=float(ua_bottom),
        ua_mix_idle_w_per_k=float(ua_mix_idle),
        ua_mix_active_w_per_k=float(ua_mix_active),
        q_in_nominal_w=float(q_in_nominal),
    )


def _resolve_q_in(
    model: BoilerThermalModel,
    on: bool,
    index: int,
    q_in_override: np.ndarray | None,
) -> float:
    """The fitted Q_in_nominal_w is the default heat input while boiler_on; a
    calorimetric override (real flow*deltaT power, NaN where unavailable/invalid -
    see BoilerThermalIdentifier.prepare) takes precedence wherever it exists, since
    it reflects genuine time-varying delivered power instead of one average value.
    """

    if not on:
        return 0.0

    if q_in_override is not None:
        override = q_in_override[index]
        if not np.isnan(override):
            return float(override)

    return model.q_in_nominal_w


def _predict_next_states(
    model: BoilerThermalModel,
    T_top: np.ndarray,
    T_bottom: np.ndarray,
    T_ambient: np.ndarray,
    boiler_on: np.ndarray,
    dt_seconds: np.ndarray,
    q_in_override: np.ndarray | None = None,
) -> np.ndarray:
    """One-step-ahead prediction: state[i-1] + input[i-1] held over dt_seconds[i].

    Zero-order hold: the boiler_on/T_ambient value at the START of each interval
    governs that interval, matching the standard ZOH convention used for the
    discretization itself.
    """

    n = len(T_top)

    predictions = np.empty((n - 1, 2))

    cache: dict[tuple[bool, float], tuple[np.ndarray, np.ndarray]] = {}

    for i in range(1, n):
        on = bool(boiler_on[i - 1])
        dt = float(dt_seconds[i])
        key = (on, round(dt, 3))

        if key not in cache:
            ua_mix = model.ua_mix_active_w_per_k if on else model.ua_mix_idle_w_per_k
            a, b = _state_space(
                model.volume_l,
                model.ua_top_w_per_k,
                model.ua_bottom_w_per_k,
                ua_mix,
            )
            cache[key] = discretize_zoh(a, b, dt)

        a_d, b_d = cache[key]
        q_in = _resolve_q_in(model, on, i - 1, q_in_override)
        state = np.array([T_top[i - 1], T_bottom[i - 1]])
        u = np.array([T_ambient[i - 1], q_in])

        predictions[i - 1] = a_d @ state + b_d @ u

    return predictions


def _rollout(
    model: BoilerThermalModel,
    T_top_0: float,
    T_bottom_0: float,
    T_ambient: np.ndarray,
    boiler_on: np.ndarray,
    dt_seconds: np.ndarray,
    q_in_override: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-simulate a window from a single initial measured state.

    Only the initial state plus the measured boiler_on/T_ambient trajectory are used -
    no mid-window real measurements - so this tests genuine forward-simulation
    accuracy over the window, not one-step curve-fitting. The window must be short
    enough that unmeasured disturbances (chiefly tap draws) are unlikely to have
    occurred - see BoilerThermalIdentifier.validate.
    """

    n = len(T_ambient)

    simulated = np.empty((n, 2))
    simulated[0] = [T_top_0, T_bottom_0]

    cache: dict[tuple[bool, float], tuple[np.ndarray, np.ndarray]] = {}

    for i in range(1, n):
        on = bool(boiler_on[i - 1])
        dt = float(dt_seconds[i])
        key = (on, round(dt, 3))

        if key not in cache:
            ua_mix = model.ua_mix_active_w_per_k if on else model.ua_mix_idle_w_per_k
            a, b = _state_space(
                model.volume_l,
                model.ua_top_w_per_k,
                model.ua_bottom_w_per_k,
                ua_mix,
            )
            cache[key] = discretize_zoh(a, b, dt)

        a_d, b_d = cache[key]
        q_in = _resolve_q_in(model, on, i - 1, q_in_override)
        u = np.array([T_ambient[i - 1], q_in])

        simulated[i] = a_d @ simulated[i - 1] + b_d @ u

    return simulated


class BoilerThermalIdentifier(SystemIdentifier[BoilerThermalModel]):
    # "SWW" (sanitair warm water) is the only DHW-active value observed in this
    # installation's state sensor; T_supply/T_return/flow_lpm are shared with the
    # space-heating circuit, so boiler_on must whitelist this value rather than
    # blacklist "Uit" - a hypothetical space-heating state string must NOT be read as
    # boiler heat input.
    DHW_ACTIVE_STATE = "SWW"

    # A flow reading must be strictly positive to mean anything physically (a real
    # minimum, not a fitted threshold).
    MIN_FLOW_LPM = 0.0

    # Whitelist, not a blacklist of "anything that isn't SWW" (same principle
    # as DHW_ACTIVE_STATE above): the only state prepare()'s flow-gap bridging
    # may trust as "compressor definitely off" - a hypothetical space-heating
    # state must not be forced to 0 here either, even though this identifier's
    # own calorimetric override is separately gated on boiler_on regardless.
    HEAT_PUMP_OFF_STATE = "Uit"

    TRAIN_RATIO = 0.80

    # Once a mixing conductance is large enough that the top/bottom gap decays to
    # this fraction of its initial value within a single sampling interval, further
    # increasing it is completely unobservable at this data's time resolution: the
    # discretized prediction stops changing, so the optimizer has nothing left to
    # constrain it and can run away to an arbitrarily large, meaningless value on a
    # locally flat objective (observed on real data: an unbounded fit drove
    # UA_mix_active past 2.9 million W/K with a std error larger than itself). This
    # is a numerical/identifiability ceiling tied to the sampling rate, not an
    # assumed physical maximum - see calibrate(), where it is combined with the
    # actual C_node/dt to bound UA_mix_idle and UA_mix_active.
    NEGLIGIBLE_MIXING_RESIDUAL_FRACTION = 1e-6

    # Excludes one-step transitions spanning a data gap much larger than the nominal
    # sampling interval - such a transition mostly tests steady-state convergence, not
    # the dynamics, and would otherwise dominate the fit disproportionately.
    MAX_DT_SECONDS_MULTIPLE = 6.0

    # Observed directly in this installation's own data: T_top/T_bottom kept rising
    # for ~15 minutes after `state` flipped to "Uit" (residual heat in the heat
    # exchanger/coil after the compressor stops). Idle-labeled samples within this
    # window after a heating stop are excluded from calibration - see calibrate().
    POST_HEATING_TAIL_SECONDS = 15.0 * 60.0

    # A slow passive decay's true one-step (single sampling interval) signal can be
    # far smaller than sensor noise/resolution: confirmed directly on real data,
    # where a model fit from one-step residuals alone predicted ~5x too little
    # cooling over several days compared to a directly observed decay during a
    # confirmed-nobody-home window. UA_top/UA_bottom/UA_mix_idle are therefore
    # additionally identified from same-anchor rollout residuals over clean idle
    # windows this long (see calibrate()'s decay_residuals) - long enough for the
    # true decay to clear the noise floor, short enough to keep the chance of an
    # undetected tap draw occurring within the window low. Reuses the same
    # physical trade-off, and the same default value, as validate()'s own
    # receding-horizon evaluation.
    PASSIVE_DECAY_HORIZON_HOURS = 2.0

    # Standard MAD-to-std conversion constant for a normal distribution, used to turn
    # the empirical residual spread into a robust-loss scale (not a physical constant).
    MAD_TO_STD = 1.4826
    MIN_F_SCALE = 1e-6

    # Reporting-only sensitivity for flagging candidate excess-heat-loss timesteps in
    # calibrate()'s training diagnostics; does not change the fit itself (the robust
    # loss already handles that) and is not a claim of validated tap-water usage.
    EXCESS_LOSS_FLAG_SCALE_MULTIPLE = 3.0

    # Sensitivity sweep (diagnostic only, does not change the fit): reports the
    # excess-loss flag rate as if POST_HEATING_TAIL_SECONDS had been each of these
    # durations instead, so a too-short assumed tail shows up as "flag rate keeps
    # dropping well past 15 min" rather than being guessed at.
    TAIL_SWEEP_MINUTES = (0, 5, 15, 30, 45, 60)

    # Home Assistant device_tracker convention: only an explicit "not_home" state or
    # (as InfluxDB actually exports these specific trackers) a numeric 0.0 is
    # trusted as confirmed away - whitelist, not blacklist, the same principle as
    # DHW_ACTIVE_STATE. A custom zone name could reflect a GPS-accuracy artifact
    # while someone is still actually home, and anything else (including
    # "unknown"/"unavailable" or a missing/NaN reading) means the tracker itself is
    # unreliable right now - none of those may be read as "away", since this
    # diagnostic's entire value depends on the away-bucket being trustworthy.
    AWAY_PRESENCE_STATE = "not_home"
    AWAY_PRESENCE_VALUE = 0.0

    # Settling margin against tracker latency/GPS inaccuracy right at a home<->away
    # transition - a detection-sensitivity choice, not a physical constant. Only
    # samples this far into a continuous away run are trusted as "confirmed no one
    # could have drawn water", used purely as an identification diagnostic below.
    MIN_CONFIRMED_AWAY_SECONDS = 60.0 * 60.0

    # Below this many directly-measured heating timesteps, a calorimetric sample
    # mean is not yet reliable enough to anchor q_in_nominal_w's bounds (see
    # calibrate()) - same "enough samples to trust a sub-group mean" threshold
    # already used for the presence diagnostic below.
    MIN_CALORIMETRIC_Q_IN_SAMPLES = 20

    # Width of the confidence interval (in standard errors of the calorimetric
    # sample mean) used to bound q_in_nominal_w around a direct measurement -
    # the standard 3-sigma convention (~99.7% for a Gaussian), not a fitted or
    # asserted value.
    CALORIMETRIC_Q_IN_CONFIDENCE_SIGMAS = 3.0

    # least_squares starting points, each derived from the observed order of magnitude
    # in the available sample data (not asserted as true values - refined by the fit):
    # UA_top/UA_bottom ~ a modestly insulated 200L cylinder's standby loss; UA_mix_idle
    # small (weak passive coupling); UA_mix_active from the ~15-20 min observed
    # equalization time constant during a heating start; Q_in_nominal from the ~4.5 kW
    # average implied by the observed temperature rise and elapsed time during heating.
    INITIAL_UA_TOP_W_PER_K = 1.5
    INITIAL_UA_BOTTOM_W_PER_K = 1.5
    INITIAL_UA_MIX_IDLE_W_PER_K = 0.5
    INITIAL_UA_MIX_ACTIVE_W_PER_K = 700.0
    INITIAL_Q_IN_NOMINAL_W = 4500.0

    def __init__(self) -> None:
        super().__init__()
        # Matches BoilerConfig's own default; overwritten by dataset() with the
        # configured volume once available.
        self.volume_l: float = 200.0
        self.parameter_std_errors: dict[str, float] | None = None
        # Names of the presence-tracker columns dataset() requested, if any -
        # overwritten by dataset() once available. Empty means no trackers
        # configured, or prepare()/calibrate() called directly without dataset().
        self.presence_columns: list[str] = []
        # Set by calibrate() whenever a calorimetric mean anchors
        # q_in_nominal_w's bounds (see MIN_CALORIMETRIC_Q_IN_SAMPLES); checked
        # by validate() to report whether the fit is pinned at that boundary.
        # None (the __init__ default) if this attribute doesn't survive between
        # a calibrate() and a later validate() call (e.g. a process restart) -
        # validate() then simply skips this check, same as parameter_std_errors.
        self.q_in_calorimetric_bounds: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        return "boiler"

    @property
    def label(self) -> str:
        return "Boiler temperatures"

    def _bridge_flow_reporting_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """flow_lpm is rate-like: dataset() fetches it with no InfluxDB fill at
        all, so a real reporting gap shows up here as NaN rather than a
        guessed value. The compressor's own, separately and reliably reported
        state settles what a gap actually means: bridging forward while state
        confirms it is still active (confirmed on real data - T_supply/
        T_return kept rising smoothly through a 15-minute flow_lpm gap during
        a DHW ramp-up, a real reporting hiccup, not zero flow), but resetting
        to 0 the moment state reports idle regardless of how long ago the last
        reading was - the exact bug this replaces (a stale nonzero reading
        persisting for minutes past a real, confirmed shutoff).
        """

        df = df.copy()
        df["flow_lpm"] = pd.to_numeric(df["flow_lpm"], errors="coerce")
        is_off = df["state"] == self.HEAT_PUMP_OFF_STATE

        df["flow_lpm"] = df["flow_lpm"].ffill()
        df["flow_lpm"] = df["flow_lpm"].where(~is_off, 0.0)
        # No prior reading at all (e.g. the very start of the fetched window)
        # - assume 0 rather than leaving it unresolved.
        df["flow_lpm"] = df["flow_lpm"].fillna(0.0)

        return df

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_columns = ["time", "T_ambient", "T_top", "T_bottom", "state"]

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        numeric_columns = ["T_ambient", "T_top", "T_bottom"]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=numeric_columns).copy()

        df = df.sort_values("time").reset_index(drop=True)

        df["dt_seconds"] = df["time"].diff().dt.total_seconds()

        df = df.dropna(subset=["dt_seconds"]).copy()

        df = df[df["dt_seconds"] > 0].copy()

        df["boiler_on"] = df["state"] == self.DHW_ACTIVE_STATE

        if "flow_lpm" in df.columns:
            df = self._bridge_flow_reporting_gaps(df)

        # Calorimetric heat input: Q = (rho*cp/60) * flow_lpm * max(T_supply -
        # T_return, 0) replaces the fitted constant Q_in_nominal_w wherever
        # real, valid flow data exists during SWW - real time-varying
        # delivered power instead of one average value. Confirmed for this
        # installation: no simultaneous space heating + SWW, so a positive
        # flow during SWW is unambiguously the boiler coil. Strictly gated
        # on boiler_on (never on flow being merely nonzero): this
        # installation's flow sensor does not report while idle, so
        # InfluxDB's fill leaves stale readings behind after shutoff
        # (observed directly: ~14 L/min minutes after state flipped to
        # "Uit") - trusting flow_lpm's value on its own during "Uit" would
        # inject phantom heat into idle data.
        #
        # T_supply <= T_return is clipped to Q=0 rather than treated as
        # invalid: confirmed on real data (see calibrate()'s diagnostic) -
        # every such case occurred exactly at a heating run's first sample,
        # the compressor having just started with the refrigerant not yet
        # hot enough to exceed the tank's own return temperature. That is a
        # real, valid measurement of "no net heat yet", not a sensor
        # problem - falling back to Q_in_nominal_w there would incorrectly
        # apply the *rest* of the cycle's steady, well-measured average to
        # this brief, physically distinct startup instant.
        #
        # Falls back to the fitted Q_in_nominal_w (see calibrate()) only
        # wherever flow itself is absent/non-positive, or these columns are
        # not configured at all - NaN here means "no override", not "no
        # heat".
        calorimetric_columns = ["T_supply", "T_return", "flow_lpm"]

        if all(column in df.columns for column in calorimetric_columns):
            T_supply = pd.to_numeric(df["T_supply"], errors="coerce")
            T_return = pd.to_numeric(df["T_return"], errors="coerce")
            flow_lpm = pd.to_numeric(df["flow_lpm"], errors="coerce")
            delta_t_water = (T_supply - T_return).clip(lower=0.0)

            valid = df["boiler_on"] & (flow_lpm > self.MIN_FLOW_LPM)

            q_calorimetric_w = (
                (RHO_WATER_KG_PER_L / 60.0) * CP_WATER_J_PER_KG_K
            ) * flow_lpm * delta_t_water

            df["q_in_override_w"] = q_calorimetric_w.where(valid)
        else:
            df["q_in_override_w"] = np.nan

        # Presence trackers are an independent, exogenous signal, not derived from
        # boiler temperature: nobody home rules out a tap draw (barring a timed
        # appliance). "confirmed_away_settled" is only True once every configured
        # tracker has reported an explicit away reading (whitelist, not a blacklist
        # of "everything but home" - a custom zone or GPS glitch must not count) for
        # at least MIN_CONFIRMED_AWAY_SECONDS continuously - used purely as an
        # identification diagnostic in calibrate(), never as a model input. Defaults
        # to all-False if no trackers are configured.
        present_columns = [c for c in self.presence_columns if c in df.columns]

        if present_columns:
            tracker_away = pd.concat(
                [
                    (df[column] == self.AWAY_PRESENCE_STATE)
                    | (
                        pd.to_numeric(df[column], errors="coerce")
                        == self.AWAY_PRESENCE_VALUE
                    )
                    for column in present_columns
                ],
                axis=1,
            )
            confirmed_away = tracker_away.all(axis=1)

            away_run_id = (confirmed_away != confirmed_away.shift()).cumsum()
            away_run_start = df.groupby(away_run_id)["time"].transform("first")
            seconds_into_away_run = (df["time"] - away_run_start).dt.total_seconds()

            df["confirmed_away_settled"] = confirmed_away & (
                seconds_into_away_run >= self.MIN_CONFIRMED_AWAY_SECONDS
            )
        else:
            df["confirmed_away_settled"] = False

        df = df.reset_index(drop=True)

        logger.info("Boiler thermal preparation: %d valid points", len(df))

        if df.empty:
            raise ValueError("No valid boiler measurements remain after preparation.")

        return df

    def _clean_idle_decay_windows(
        self,
        boiler_on: np.ndarray,
        in_post_heating_tail: np.ndarray,
        median_dt: float,
    ) -> list[tuple[int, int]]:
        """Maximal, non-overlapping windows of PASSIVE_DECAY_HORIZON_HOURS spent
        entirely idle - no heating and clear of the post-heating tail throughout,
        not just at the start - used by calibrate() to identify the passive-loss
        parameters from a horizon long enough for the true decay signal to clear
        sensor noise. Returns (start, end) row-index pairs into the training
        arrays (end - start equals the target window length in samples); a
        trailing remainder shorter than the target is dropped, since it would
        not clear the same noise floor.
        """

        samples_per_window = max(
            int(round(self.PASSIVE_DECAY_HORIZON_HOURS * 3600.0 / median_dt)), 2
        )

        clean_idle = (~boiler_on) & (~in_post_heating_tail)

        windows: list[tuple[int, int]] = []
        run_start: int | None = None

        def close_run(run_end: int) -> None:
            start = run_start
            while start is not None and start + samples_per_window <= run_end:
                windows.append((start, start + samples_per_window))
                start += samples_per_window

        for i, is_clean in enumerate(clean_idle):
            if is_clean and run_start is None:
                run_start = i
            elif not is_clean and run_start is not None:
                close_run(i - 1)
                run_start = None

        if run_start is not None:
            close_run(len(clean_idle) - 1)

        return windows

    def calibrate(self, df: pd.DataFrame) -> BoilerThermalModel:
        df = self.prepare(df)

        if len(df) < 10:
            raise ValueError("Not enough data points for calibration.")

        median_dt = df["dt_seconds"].median()

        df = df[df["dt_seconds"] <= self.MAX_DT_SECONDS_MULTIPLE * median_dt].copy()

        df = df.reset_index(drop=True)

        split_index = int(len(df) * self.TRAIN_RATIO)

        if split_index <= 1 or split_index >= len(df):
            raise ValueError("Invalid train/test split.")

        train_df = df.iloc[:split_index].reset_index(drop=True)

        logger.info(
            "Boiler thermal calibration: %d training points, %d validation points",
            len(train_df),
            len(df) - len(train_df),
        )

        T_top = train_df["T_top"].to_numpy(dtype=float)
        T_bottom = train_df["T_bottom"].to_numpy(dtype=float)
        T_ambient = train_df["T_ambient"].to_numpy(dtype=float)
        boiler_on = train_df["boiler_on"].to_numpy(dtype=bool)
        dt_seconds = train_df["dt_seconds"].to_numpy(dtype=float)
        q_in_override = train_df["q_in_override_w"].to_numpy(dtype=float)
        confirmed_away_settled = train_df["confirmed_away_settled"].to_numpy(
            dtype=bool
        )

        # q_in_nominal_w is only ever USED (see _resolve_q_in) on heating
        # timesteps lacking a valid calorimetric override - wherever the
        # installation's own T_supply/T_return/flow sensors are configured and
        # valid, they determine Q_in directly and q_in_nominal_w's fitted value
        # has no effect on the fit at all. Logged so a surprising fitted value
        # (e.g. far from the ~4.5 kW heat-pump-condenser order of magnitude the
        # initial guess uses) can be judged against how much evidence it
        # actually had, rather than assumed to reflect the real installation.
        heating_count = int(np.sum(boiler_on))
        override_valid = boiler_on & ~np.isnan(q_in_override)
        override_valid_count = int(np.sum(override_valid))

        # A calorimetric mean, wherever reliable, is a direct measurement of the
        # real average delivered power - a better anchor for q_in_nominal_w than
        # letting least_squares fit it freely, since that fit is only ever
        # informed by the (typically very few) heating timesteps lacking a
        # valid override (see _resolve_q_in and prepare()'s Q=0 handling for
        # T_supply<=T_return): a small, arbitrary, sensor-limited remainder
        # (real flow reading missing entirely), not representative evidence
        # for the installation's real heating power. Below
        # MIN_CALORIMETRIC_Q_IN_SAMPLES, the sample mean itself is not yet
        # reliable enough to anchor anything, so the fit stays fully free.
        calorimetric_q_in_bounds: tuple[float, float] | None = None

        if override_valid_count >= self.MIN_CALORIMETRIC_Q_IN_SAMPLES:
            calorimetric_values = q_in_override[override_valid]
            calorimetric_mean = float(np.mean(calorimetric_values))
            calorimetric_sem = float(
                np.std(calorimetric_values, ddof=1) / np.sqrt(override_valid_count)
            )
            # A near-zero calorimetric spread (e.g. a very stable flow/deltaT
            # reading) would otherwise collapse the margin to ~0, and
            # least_squares requires a strictly positive bound width.
            margin = max(
                self.CALORIMETRIC_Q_IN_CONFIDENCE_SIGMAS * calorimetric_sem,
                self.MIN_F_SCALE,
            )
            calorimetric_q_in_bounds = (
                max(0.0, calorimetric_mean - margin),
                calorimetric_mean + margin,
            )

        self.q_in_calorimetric_bounds = calorimetric_q_in_bounds

        if heating_count > 0:
            logger.info(
                "Boiler thermal calibration: calorimetric Q_in override valid "
                "for %d/%d heating training timesteps - %s",
                override_valid_count,
                heating_count,
                (
                    f"mean {float(np.mean(q_in_override[override_valid])):.1f} W "
                    f"where valid; q_in_nominal_w is bounded to "
                    f"[{calorimetric_q_in_bounds[0]:.1f}, "
                    f"{calorimetric_q_in_bounds[1]:.1f}] W around this direct "
                    f"measurement, not fit freely."
                )
                if calorimetric_q_in_bounds is not None
                else (
                    f"only {override_valid_count} valid sample(s), too few to "
                    f"anchor q_in_nominal_w (need "
                    f"{self.MIN_CALORIMETRIC_Q_IN_SAMPLES}) - fit freely from "
                    f"the {heating_count - override_valid_count} timestep(s) "
                    f"lacking a valid override."
                ),
            )

        # DIAGNOSTIC: heating timesteps still lacking a calorimetric override
        # (see prepare()'s `valid` mask - now only flow_lpm <= MIN_FLOW_LPM;
        # T_supply <= T_return is a valid Q=0 measurement, not a missing
        # one - confirmed on real data to occur exactly at a heating run's
        # first sample, a genuine startup transient, not scattered mid-cycle
        # noise). Only this smaller, genuinely sensor-limited remainder
        # still relies on q_in_nominal_w's fallback.
        missing_override = boiler_on & ~override_valid
        missing_override_count = int(np.sum(missing_override))

        if missing_override_count > 0:
            logger.info(
                "Boiler thermal calibration: %d/%d heating timesteps still "
                "lack a calorimetric override (flow_lpm<=%.1f) and rely on "
                "q_in_nominal_w.",
                missing_override_count,
                heating_count,
                self.MIN_FLOW_LPM,
            )

        # A real after-heat tail was directly observed in this installation's own
        # data: T_top/T_bottom kept rising for ~15 minutes after `state` flipped to
        # "Uit" (residual heat in the heat exchanger/coil after the compressor
        # stops) - a real physical effect the binary boiler_on switch cannot
        # represent. Samples labeled idle within this window after a heating stop
        # are not reliable evidence for the passive ambient-loss/mixing parameters:
        # including them biases UA_top/UA_bottom/UA_mix_idle toward whatever
        # reconciles a still-rising temperature with an assumed-zero heat input.
        # Excluded from calibration and its diagnostics only - validate() still
        # judges the model against this real behavior.
        last_heating_time = train_df["time"].where(train_df["boiler_on"]).ffill()
        seconds_since_heating_stop = (
            (train_df["time"] - last_heating_time).dt.total_seconds().to_numpy()
        )

        def tail_mask(tail_seconds: float) -> np.ndarray:
            return (~boiler_on) & (seconds_since_heating_stop <= tail_seconds)

        in_post_heating_tail = tail_mask(self.POST_HEATING_TAIL_SECONDS)
        clean_transition = ~in_post_heating_tail[1:]

        decay_windows = self._clean_idle_decay_windows(
            boiler_on, in_post_heating_tail, median_dt
        )

        logger.info(
            "Boiler thermal calibration: %d clean idle decay window(s) of "
            "%.1fh found - used to identify UA_top/UA_bottom/UA_mix_idle from a "
            "horizon long enough to clear one-step sensor noise (see "
            "PASSIVE_DECAY_HORIZON_HOURS). Too few here means those parameters "
            "remain identified from one-step residuals alone, as before.",
            len(decay_windows),
            self.PASSIVE_DECAY_HORIZON_HOURS,
        )

        def decay_residuals(parameters: np.ndarray) -> np.ndarray:
            if not decay_windows:
                return np.empty(0)

            model = _model_from_parameters(parameters, self.volume_l)
            pieces = []

            for start, end in decay_windows:
                simulated = _rollout(
                    model,
                    T_top[start],
                    T_bottom[start],
                    T_ambient[start : end + 1],
                    boiler_on[start : end + 1],
                    dt_seconds[start : end + 1],
                )
                actual = np.column_stack(
                    [T_top[start : end + 1], T_bottom[start : end + 1]]
                )
                pieces.append((simulated - actual)[1:].ravel())

            return np.concatenate(pieces)

        def row_residuals(parameters: np.ndarray) -> np.ndarray:
            model = _model_from_parameters(parameters, self.volume_l)

            predictions = _predict_next_states(
                model,
                T_top,
                T_bottom,
                T_ambient,
                boiler_on,
                dt_seconds,
                q_in_override,
            )

            actual = np.column_stack([T_top[1:], T_bottom[1:]])

            return predictions - actual

        def one_step_residuals(parameters: np.ndarray) -> np.ndarray:
            return row_residuals(parameters)[clean_transition].ravel()

        def combined_residuals(
            parameters: np.ndarray, decay_weight: float = 1.0
        ) -> np.ndarray:
            return np.concatenate(
                [
                    one_step_residuals(parameters),
                    decay_residuals(parameters) * decay_weight,
                ]
            )

        if calorimetric_q_in_bounds is not None:
            q_in_seed = sum(calorimetric_q_in_bounds) / 2.0
        else:
            q_in_seed = self.INITIAL_Q_IN_NOMINAL_W

        x0 = np.array(
            [
                self.INITIAL_UA_TOP_W_PER_K,
                self.INITIAL_UA_BOTTOM_W_PER_K,
                self.INITIAL_UA_MIX_IDLE_W_PER_K,
                self.INITIAL_UA_MIX_ACTIVE_W_PER_K,
                q_in_seed,
            ]
        )

        lower_bounds = np.zeros(5)

        c_node = RHO_WATER_KG_PER_L * (self.volume_l / 2.0) * CP_WATER_J_PER_KG_K
        ua_mix_ceiling = (c_node / (2.0 * median_dt)) * np.log(
            1.0 / self.NEGLIGIBLE_MIXING_RESIDUAL_FRACTION
        )
        if calorimetric_q_in_bounds is not None:
            q_in_upper = calorimetric_q_in_bounds[1]
        else:
            q_in_upper = np.inf
        upper_bounds = np.array(
            [np.inf, np.inf, ua_mix_ceiling, ua_mix_ceiling, q_in_upper]
        )
        if calorimetric_q_in_bounds is not None:
            lower_bounds[4] = calorimetric_q_in_bounds[0]
        # A seed outside the bounds would make least_squares raise outright; the
        # ceiling is data-dependent (volume/sampling rate) while the seeds are
        # fixed constants, so this can happen for an unusual installation/interval.
        x0 = np.clip(x0, lower_bounds, upper_bounds)

        ordinary_fit = least_squares(
            combined_residuals, x0=x0, bounds=(lower_bounds, upper_bounds)
        )

        if not ordinary_fit.success:
            logger.warning(
                "Boiler thermal ordinary fit did not fully converge: %s",
                ordinary_fit.message,
            )

        # Robust refit: downweight timesteps with unusually large one-step error via a
        # standard robust loss instead of a hand-picked hard exclusion threshold.
        # f_scale comes from the ordinary fit's own residual spread (MAD), not an
        # assumed constant. This is parameter-identification machinery only - it does
        # not claim to detect tap draws, it just limits their influence on the fit.
        # Kept to one-step residuals only, same meaning as before decay_residuals
        # existed (also reused as-is by _exclude_forecast_tap_draws below).
        ordinary_residuals = one_step_residuals(ordinary_fit.x)

        f_scale = self.MAD_TO_STD * float(
            np.median(np.abs(ordinary_residuals - np.median(ordinary_residuals)))
        )

        f_scale = max(f_scale, self.MIN_F_SCALE)

        # decay_residuals live on a different, much larger natural scale than
        # one_step_residuals (that is the whole point - see
        # PASSIVE_DECAY_HORIZON_HOURS). Applying the single f_scale above to both
        # in the same soft_l1 robust loss would treat every decay residual as an
        # extreme outlier and suppress exactly the signal just added. Estimating
        # decay residuals' own MAD-based scale and rescaling them to
        # one-step-equivalent units before the robust refit (below) keeps the
        # same soft_l1(f_scale) threshold meaningful for both groups, without
        # changing where either group's minimum actually is.
        ordinary_decay_residuals = decay_residuals(ordinary_fit.x)

        if len(ordinary_decay_residuals) > 0:
            f_scale_decay = self.MAD_TO_STD * float(
                np.median(
                    np.abs(
                        ordinary_decay_residuals - np.median(ordinary_decay_residuals)
                    )
                )
            )
        else:
            f_scale_decay = self.MIN_F_SCALE

        f_scale_decay = max(f_scale_decay, self.MIN_F_SCALE)
        decay_weight = f_scale / f_scale_decay

        # Purely informational: how much idle-period training data deviates from the
        # ambient-loss/mixing model by more than the robust fit already discounts.
        # This is NOT proven tap-water usage - a temperature residual alone cannot
        # distinguish a real draw from e.g. an unmodeled disturbance or the ambient
        # sensor not representing the true local boundary temperature (see the
        # bottom-node anomaly noted during model design). Reported as candidate
        # excess heat loss so the robust fit's behavior is inspectable, not asserted
        # as a validated finding.
        # Same post-heating-tail exclusion as the fit itself: those rows are already
        # known to be unreliable idle evidence, so they must not also inflate this
        # diagnostic's flag count.
        full_row_residuals = row_residuals(ordinary_fit.x)
        idle_mask_train = (~boiler_on[:-1]) & clean_transition
        idle_indices = np.nonzero(idle_mask_train)[0]
        idle_row_residuals = full_row_residuals[idle_mask_train]

        # Populated below (if this run has any idle evidence) with which rows
        # the same-run, model-free residual threshold flags - used to sanity
        # check the tap-forecast exclusion below against this independent
        # signal, since the two use unrelated methods (a same-run temperature
        # residual vs. a separately-trained forecast).
        same_run_flagged_full = np.zeros(len(clean_transition), dtype=bool)

        if len(idle_row_residuals) > 0:
            flagged = np.any(
                np.abs(idle_row_residuals)
                > self.EXCESS_LOSS_FLAG_SCALE_MULTIPLE * f_scale,
                axis=1,
            )
            excess_loss_candidates = int(np.sum(flagged))

            # Clustering diagnostic (does NOT affect the fit): a real multi-sample
            # disturbance (e.g. a shower lasting several sampling intervals, plus its
            # equilibration tail) should leave RUNS of consecutive flagged idle
            # timesteps, not isolated single-sample spikes. Compared against the
            # baseline a purely random scatter at the same flag rate would produce
            # (~2p of flagged points adjacent to another flagged point, for small p),
            # this helps distinguish "many short real disturbances" from "isolated
            # sensor noise/outliers" - it still cannot prove either is tap water.
            flagged_indices = idle_indices[flagged]
            same_run_flagged_full[flagged_indices] = True
            clustered_fraction = float("nan")
            expected_by_chance = float("nan")

            if len(flagged_indices) > 1:
                adjacent = np.diff(flagged_indices) == 1
                in_run = np.zeros(len(flagged_indices), dtype=bool)
                in_run[:-1] |= adjacent
                in_run[1:] |= adjacent
                clustered_fraction = float(np.mean(in_run))
                flag_rate = excess_loss_candidates / len(idle_row_residuals)
                expected_by_chance = float(1.0 - (1.0 - flag_rate) ** 2)

            # Asymmetry diagnostic (does NOT affect the fit): a real tap draw injects
            # cold mains water at the BOTTOM of the tank, so it should predominantly
            # disturb T_bottom, not T_top. A systematic model misspecification (e.g.
            # the UA_top/UA_bottom split being only weakly identified when the two
            # nodes track closely) instead tends to mispredict both nodes by a
            # comparable amount. This ratio still cannot prove tap water, but a large
            # bottom/top ratio is the more direct, mechanism-specific signature.
            bottom_top_ratio = float("nan")

            if excess_loss_candidates > 0:
                mean_abs_top = float(np.mean(np.abs(idle_row_residuals[flagged, 0])))
                mean_abs_bottom = float(np.mean(np.abs(idle_row_residuals[flagged, 1])))
                if mean_abs_top > 0:
                    bottom_top_ratio = mean_abs_bottom / mean_abs_top

            logger.info(
                "Boiler thermal calibration: %d/%d idle training timesteps show "
                "excess heat loss unexplained by the ambient-loss/mixing model "
                "(candidate data contamination, e.g. undetected tap draws - NOT a "
                "validated count of tap events; the robust fit below already limits "
                "their influence). %.0f%% of flagged timesteps are adjacent to "
                "another flagged timestep, vs. %.0f%% expected by pure chance at "
                "this flag rate - well above chance suggests multi-sample real "
                "disturbances rather than isolated sensor noise (still not proof). "
                "Mean |residual| bottom/top ratio among flagged timesteps: %.2f - "
                "well above 1 points to cold water entering at the bottom (real "
                "draws); close to 1 points to a model/fit issue affecting both "
                "nodes similarly instead.",
                excess_loss_candidates,
                len(idle_row_residuals),
                clustered_fraction * 100,
                expected_by_chance * 100,
                bottom_top_ratio,
            )

            # Tail-duration sensitivity sweep (diagnostic only, does not change the
            # fit): if the flag rate keeps dropping well past the currently
            # configured POST_HEATING_TAIL_SECONDS, that duration is too short for
            # this installation's actual after-heat behavior.
            def flag_rate(row_res: np.ndarray) -> float:
                if len(row_res) == 0:
                    return float("nan")
                return float(
                    np.mean(
                        np.any(
                            np.abs(row_res)
                            > self.EXCESS_LOSS_FLAG_SCALE_MULTIPLE * f_scale,
                            axis=1,
                        )
                    )
                )

            sweep_parts = []

            for minutes in self.TAIL_SWEEP_MINUTES:
                candidate_mask = (~boiler_on[:-1]) & (~tail_mask(minutes * 60.0)[1:])
                candidate_rate = flag_rate(full_row_residuals[candidate_mask])
                sweep_parts.append(f"{minutes}min={candidate_rate * 100:.1f}%")

            logger.info(
                "Boiler thermal calibration: excess-loss flag rate if the "
                "post-heating tail exclusion had instead been each of these "
                "durations: %s (diagnostic only, does not change the fit - if the "
                "rate keeps dropping well past the currently configured %.0f min, "
                "that tail is likely still too short for this installation).",
                ", ".join(sweep_parts),
                self.POST_HEATING_TAIL_SECONDS / 60.0,
            )

            # Presence diagnostic (does NOT affect the fit): confirmed absence rules
            # out a real tap draw (barring a timed appliance), so comparing the
            # excess-loss flag rate between confirmed-away and otherwise idle
            # periods is a much more direct test than any residual-shape heuristic
            # above. Requires both endpoints of the one-step interval to be settled
            # away, and is skipped (not guessed at) if too little such data exists.
            both_away = confirmed_away_settled[:-1] & confirmed_away_settled[1:]
            away_residuals = full_row_residuals[idle_mask_train & both_away]
            present_residuals = full_row_residuals[idle_mask_train & ~both_away]

            if len(away_residuals) >= 20:
                logger.info(
                    "Boiler thermal calibration: excess-loss flag rate with "
                    "nobody confirmed home (n=%d) vs. otherwise (n=%d): %.1f%% vs "
                    "%.1f%% - if these are close, the excess loss is likely NOT "
                    "tap draws (nobody could have used water); if confirmed-away "
                    "is much lower, that supports real draws as (part of) the "
                    "cause.",
                    len(away_residuals),
                    len(present_residuals),
                    flag_rate(away_residuals) * 100,
                    flag_rate(present_residuals) * 100,
                )
            else:
                logger.info(
                    "Boiler thermal calibration: not enough confirmed-away idle "
                    "data (%d timesteps, need >=20) to compare excess-loss flag "
                    "rates by presence - configure device trackers in "
                    "Config.presence for this diagnostic, or wait for more data.",
                    len(away_residuals),
                )

            # Gradient diagnostic (does NOT affect the fit): a real tap draw is
            # not the only way a large top/bottom gradient could produce an
            # apparent excess-loss residual. UA_mix_idle is a constant (linear)
            # conductance - it cannot represent a gradient-dependent process
            # such as a stronger convective mixing burst once stratification
            # becomes more pronounced, which would masquerade as unexplained
            # heat loss without any water actually leaving the tank. Comparing
            # the flag rate at a large vs. small starting gradient (split at
            # this run's own median, not an assumed threshold) tests this
            # alternative directly.
            starting_gradient = np.abs(T_top[:-1] - T_bottom[:-1])
            idle_gradient = starting_gradient[idle_mask_train]
            median_gradient = float(np.median(idle_gradient))

            large_gradient_residuals = full_row_residuals[
                idle_mask_train & (starting_gradient > median_gradient)
            ]
            small_gradient_residuals = full_row_residuals[
                idle_mask_train & (starting_gradient <= median_gradient)
            ]

            logger.info(
                "Boiler thermal calibration: excess-loss flag rate at a large "
                "(>%.2f K, n=%d) vs small (<=%.2f K, n=%d) starting top/bottom "
                "gradient: %.1f%% vs %.1f%% - if the large-gradient rate is "
                "much higher, a gradient-dependent mixing process (not "
                "captured by the current constant UA_mix_idle) is a plausible "
                "alternative to tap draws for at least part of this signal; "
                "if they are close, the gradient itself is likely not the "
                "driver.",
                median_gradient,
                len(large_gradient_residuals),
                median_gradient,
                len(small_gradient_residuals),
                flag_rate(large_gradient_residuals) * 100,
                flag_rate(small_gradient_residuals) * 100,
            )

            # Absolute-temperature diagnostic (does NOT affect the fit): a small
            # gradient often coincides with a recently-heated, hotter tank (both
            # nodes near their post-heating peak) rather than the gradient
            # itself driving anything. UA_top/UA_bottom are constants (Newton's
            # law of cooling - loss proportional to T-T_ambient), but real
            # natural-convection heat transfer to ambient air can scale
            # super-linearly with T-T_ambient - a temperature-dependent effect
            # this model cannot represent. Comparing the flag rate at a large
            # vs. small starting T-T_ambient (split at this run's own median)
            # isolates that from the gradient effect above.
            starting_excess_temperature = (T_top[:-1] + T_bottom[:-1]) / 2.0 - (
                T_ambient[:-1]
            )
            idle_excess_temperature = starting_excess_temperature[idle_mask_train]
            median_excess_temperature = float(np.median(idle_excess_temperature))

            hot_residuals = full_row_residuals[
                idle_mask_train
                & (starting_excess_temperature > median_excess_temperature)
            ]
            cool_residuals = full_row_residuals[
                idle_mask_train
                & (starting_excess_temperature <= median_excess_temperature)
            ]

            logger.info(
                "Boiler thermal calibration: excess-loss flag rate at a large "
                "(>%.1f K, n=%d) vs small (<=%.1f K, n=%d) starting "
                "T-T_ambient: %.1f%% vs %.1f%% - if the hotter-tank rate is "
                "much higher, a temperature-dependent (not just gradient- or "
                "draw-driven) heat-transfer effect - e.g. super-linear natural "
                "convection at higher T-T_ambient, which the current constant "
                "UA cannot represent - is a plausible alternative explanation.",
                median_excess_temperature,
                len(hot_residuals),
                median_excess_temperature,
                len(cool_residuals),
                flag_rate(hot_residuals) * 100,
                flag_rate(cool_residuals) * 100,
            )

        clean_transition = self._exclude_forecast_tap_draws(
            train_df,
            clean_transition,
            boiler_on,
            f_scale,
            median_dt,
            same_run_flagged_full,
        )

        robust_fit = least_squares(
            lambda parameters: combined_residuals(parameters, decay_weight),
            x0=ordinary_fit.x,
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=f_scale,
        )

        if not robust_fit.success:
            logger.warning(
                "Boiler thermal robust fit did not fully converge: %s",
                robust_fit.message,
            )

        std_errors = self._parameter_std_errors(robust_fit)

        parameter_names = [
            "ua_top_w_per_k",
            "ua_bottom_w_per_k",
            "ua_mix_idle_w_per_k",
            "ua_mix_active_w_per_k",
            "q_in_nominal_w",
        ]

        self.parameter_std_errors = dict(
            zip(parameter_names, std_errors.tolist(), strict=True)
        )

        logger.info(
            "Boiler thermal parameters calibrated: "
            "UA_top=%.3f±%.3f W/K, UA_bottom=%.3f±%.3f W/K, "
            "UA_mix_idle=%.3f±%.3f W/K, UA_mix_active=%.1f±%.1f W/K, "
            "Q_in=%.1f±%.1f W",
            robust_fit.x[0],
            std_errors[0],
            robust_fit.x[1],
            std_errors[1],
            robust_fit.x[2],
            std_errors[2],
            robust_fit.x[3],
            std_errors[3],
            robust_fit.x[4],
            std_errors[4],
        )

        self.model = _model_from_parameters(robust_fit.x, self.volume_l)

        return self.model

    def _exclude_forecast_tap_draws(
        self,
        train_df: pd.DataFrame,
        clean_transition: np.ndarray,
        boiler_on: np.ndarray,
        f_scale: float,
        median_dt: float,
        same_run_flagged: np.ndarray,
    ) -> np.ndarray:
        """Extends clean_transition with IDLE rows the tap-demand forecaster (see
        features/tap.py) predicts as a likely real draw. Uses the forecaster
        exactly as already trained (initial_train_size=None, refit=False - no
        retraining happens here, keeping this lightweight) to produce a genuine
        one-step-ahead prediction at every row from its own learned
        hour/day-of-week/presence pattern, instead of relying only on this
        same-run's own residual: a model trained across many past cycles gives a
        more stable signal than one noisy single-row measurement, which is the
        only reason this is a meaningful addition on top of the robust loss
        above (already limits the influence of any large one-off residual on
        its own). Restricted to idle transitions only, same as excess_loss_w()
        itself (NaN while boiler_on): the tap forecaster has no boiler_on input
        and can output a large value during a heating transition too, but a
        heating transition is the only evidence Q_in has - excluding one would
        starve that estimate, not clean it. Falls back to the unchanged mask
        whenever no tap model exists yet, or train_df is too short for even one
        prediction. Best effort only: any failure here must never break the
        physical model fit itself, so exceptions are swallowed and logged.

        `same_run_flagged` (aligned to clean_transition) is the same-run,
        model-free residual flag computed above, passed in purely to log how
        much the two independent signals agree - it does not affect the mask.
        """

        if self.models_path is None:
            return clean_transition

        try:
            from features.tap import TapForecaster  # local: tap.py imports this
            # module at top level, so importing it back here would be a
            # module-level cycle if done at the top of this file.

            tap_forecaster = TapForecaster(models_path=self.models_path)
            tap_forecaster.load(self.models_path)

            if not tap_forecaster.forecaster.is_fitted:
                return clean_transition

            # The forecaster trains on 15-minute data (see TapForecaster.dataset());
            # resample this calibration window the same way instead of re-fetching
            # it, using the same aggregation as that dataset (mean for
            # temperatures, last for the event-like state/presence columns).
            presence_columns = [
                c for c in self.presence_columns if c in train_df.columns
            ]
            indexed = train_df.set_index("time")
            resampled = pd.concat(
                [
                    indexed[["T_ambient", "T_top", "T_bottom"]]
                    .resample("15min")
                    .mean(),
                    indexed[["state"] + presence_columns].resample("15min").last(),
                ],
                axis=1,
            ).dropna(subset=["T_ambient", "T_top", "T_bottom", "state"])

            prepared = tap_forecaster.prepare(resampled.reset_index())

            if len(prepared) <= tap_forecaster.forecaster.window_size:
                return clean_transition

            _, backtest = backtesting_forecaster(
                forecaster=tap_forecaster.forecaster,
                y=prepared["excess_loss_w"],
                exog=prepared[["present"]],
                cv=TimeSeriesFold(steps=1, initial_train_size=None, refit=False),
                metric="mean_absolute_error",
                n_jobs=1,
            )

            # A 15-minute prediction applies to every native-resolution row inside
            # that bucket.
            predicted_frame = backtest[["pred"]].rename(
                columns={"pred": "predicted_excess_loss_w"}
            )
            predicted_frame.index.name = "time"

            aligned = pd.merge_asof(
                train_df[["time"]].sort_values("time"),
                predicted_frame.sort_index().reset_index(),
                on="time",
                direction="backward",
                tolerance=pd.Timedelta(minutes=15),
            )["predicted_excess_loss_w"].to_numpy()

            # f_scale (see calibrate()) is a Kelvin-domain per-node residual
            # noise scale - not directly comparable to excess_loss_w (Watts).
            # excess_loss_w() itself is mostly exact zeros with occasional
            # positive spikes (a rectified, zero-inflated quantity), so a MAD
            # taken directly on it collapses to 0 (over half the values are
            # identically 0) and is not usable as a noise scale either.
            # Converting f_scale through excess_loss_w()'s own definition
            # (excess_energy_j = sum(top, bottom residuals) * c_node, divided by
            # dt) instead gives the Watts-scale noise floor implied by the same
            # temperature-measurement/model noise the Kelvin threshold already
            # represents - two independent node residuals of scale f_scale sum
            # to a scale of sqrt(2)*f_scale.
            c_node = RHO_WATER_KG_PER_L * (self.volume_l / 2.0) * CP_WATER_J_PER_KG_K
            f_scale_w = c_node * np.sqrt(2.0) * f_scale / median_dt

            # Reusing EXCESS_LOSS_FLAG_SCALE_MULTIPLE - the exact same
            # statistically-derived sensitivity already used for the same-run
            # diagnostic flag above - rather than inventing a second one. A
            # missing (unaligned) prediction must never exclude a row.
            tap_predicted_high = np.where(
                np.isnan(aligned),
                False,
                aligned > self.EXCESS_LOSS_FLAG_SCALE_MULTIPLE * f_scale_w,
            )

            idle_transition = ~boiler_on[:-1]
            tap_excluded_idle = tap_predicted_high[1:] & idle_transition
            newly_excluded = tap_excluded_idle & clean_transition
            excluded = int(np.sum(newly_excluded))

            if excluded > 0:
                # Sanity check against a completely independent signal: the
                # same-run diagnostic flags timesteps from this run's own
                # temperature residual alone, with no forecaster involved. High
                # overlap means both methods point at the same rows (the
                # forecaster is not inventing contamination nobody else sees);
                # low overlap would mean the forecaster's (now deliberately
                # conservative - see TapForecaster.TAP_FORECAST_QUANTILE)
                # predictions are excluding rows the raw residual itself does
                # not consider unusual, worth revisiting the exclusion
                # threshold or quantile if that shows up repeatedly.
                same_run_flagged_count = int(
                    np.sum(same_run_flagged & clean_transition)
                )
                overlap = int(np.sum(newly_excluded & same_run_flagged))

                logger.info(
                    "Boiler thermal calibration: excluding %d additional idle "
                    "training timesteps the tap-demand forecaster predicts as a "
                    "likely real draw (%.0f%% of these are also flagged by the "
                    "independent same-run residual threshold above, which "
                    "flags %d idle timesteps in total).",
                    excluded,
                    100.0 * overlap / excluded,
                    same_run_flagged_count,
                )

            return clean_transition & ~tap_excluded_idle
        except Exception:
            logger.info(
                "Boiler thermal calibration: skipping tap-forecast cleaning "
                "(no usable tap model yet for this window).",
                exc_info=True,
            )
            return clean_transition

    def validate(
        self, df: pd.DataFrame, horizon_hours: float = 2.0
    ) -> dict[str, float]:
        df = self.prepare(df)

        if self.model is None:
            raise RuntimeError("Model must be calibrated before validation.")

        if len(df) < 2:
            raise ValueError("Not enough data points for validation.")

        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be greater than zero.")

        median_dt = df["dt_seconds"].median()

        df = df[df["dt_seconds"] <= self.MAX_DT_SECONDS_MULTIPLE * median_dt].copy()

        df = df.reset_index(drop=True)

        split_index = int(len(df) * self.TRAIN_RATIO)

        test_df = df.iloc[split_index:].reset_index(drop=True)

        if len(test_df) < 2:
            raise ValueError("No validation data available.")

        T_top_measured = test_df["T_top"].to_numpy(dtype=float)
        T_bottom_measured = test_df["T_bottom"].to_numpy(dtype=float)
        T_ambient = test_df["T_ambient"].to_numpy(dtype=float)
        boiler_on = test_df["boiler_on"].to_numpy(dtype=bool)
        dt_seconds = test_df["dt_seconds"].to_numpy(dtype=float)
        q_in_override = test_df["q_in_override_w"].to_numpy(dtype=float)

        # Receding-horizon rollout: re-anchor to the real measured state every
        # horizon_hours and only simulate forward that far, instead of one long
        # open-loop rollout over the whole validation set. This matches how the model
        # is actually meant to be used (an MPC re-solves from real sensor readings at
        # every control step); it is also a physical necessity, not a convenience -
        # this model has no way to represent tap draws (the dominant real heat-removal
        # mechanism between heating cycles), so an open-loop rollout spanning many
        # cycles inevitably accumulates that unmodeled energy and diverges regardless
        # of how correct the passive-loss/mixing dynamics are.
        samples_per_window = max(int(round(horizon_hours * 3600.0 / median_dt)), 2)

        simulated_windows = []
        measured_windows = []
        boiler_on_windows = []

        for start in range(0, len(test_df) - 1, samples_per_window):
            end = min(start + samples_per_window, len(test_df))

            if end - start < 2:
                continue

            window_simulated = _rollout(
                self.model,
                T_top_measured[start],
                T_bottom_measured[start],
                T_ambient[start:end],
                boiler_on[start:end],
                dt_seconds[start:end],
                q_in_override[start:end],
            )

            simulated_windows.append(window_simulated)
            measured_windows.append(
                np.column_stack(
                    [T_top_measured[start:end], T_bottom_measured[start:end]]
                )
            )
            boiler_on_windows.append(boiler_on[start:end])

        simulated = np.concatenate(simulated_windows)
        window_measured = np.concatenate(measured_windows)
        window_boiler_on = np.concatenate(boiler_on_windows)

        measured = np.concatenate([window_measured[:, 0], window_measured[:, 1]])
        predicted = np.concatenate([simulated[:, 0], simulated[:, 1]])

        r2 = float(r2_score(measured, predicted))
        mae = float(mean_absolute_error(measured, predicted))
        rmse = float(np.sqrt(mean_squared_error(measured, predicted)))

        logger.info(
            "Boiler thermal validation (%.1fh receding-horizon rollout, %d windows): "
            "R2=%.4f, MAE=%.4f, RMSE=%.4f",
            horizon_hours,
            len(simulated_windows),
            r2,
            mae,
            rmse,
        )

        # Prediction error split by segment, not a claim about detected tap events:
        # this reports how well the model tracks reality while actively heating vs.
        # idle, given the real (draw-driven) boiler_on trajectory - it does not label
        # any individual residual as a validated draw.
        def _segment_metrics(mask: np.ndarray) -> dict[str, float]:
            if not np.any(mask) or np.sum(mask) < 2:
                return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}

            seg_measured = np.concatenate(
                [window_measured[mask, 0], window_measured[mask, 1]]
            )
            seg_predicted = np.concatenate([simulated[mask, 0], simulated[mask, 1]])

            return {
                "r2": float(r2_score(seg_measured, seg_predicted)),
                "mae": float(mean_absolute_error(seg_measured, seg_predicted)),
                "rmse": float(np.sqrt(mean_squared_error(seg_measured, seg_predicted))),
            }

        idle_metrics = _segment_metrics(~window_boiler_on)
        sww_metrics = _segment_metrics(window_boiler_on)

        logger.info(
            "Boiler thermal validation by segment: idle R2=%.4f MAE=%.4f RMSE=%.4f "
            "(n=%d) | SWW R2=%.4f MAE=%.4f RMSE=%.4f (n=%d)",
            idle_metrics["r2"],
            idle_metrics["mae"],
            idle_metrics["rmse"],
            int(np.sum(~window_boiler_on)),
            sww_metrics["r2"],
            sww_metrics["mae"],
            sww_metrics["rmse"],
            int(np.sum(window_boiler_on)),
        )

        # Physical plausibility: a bare, uninsulated 200L cylinder has a surface area
        # on the order of ~2 m^2 per node and a combined convective+radiative surface
        # coefficient on the order of 5-10 W/(m^2 K), giving a loose upper bound of
        # roughly 20 W/K for a completely uninsulated tank. Any calibrated UA well
        # above that is not "a better fit" - it signals a misspecified model or data
        # issue (CLAUDE.md: a better fit is not evidence of a better physical model).
        uninsulated_ua_ceiling_w_per_k = 20.0

        implausible_ua = (
            self.model.ua_top_w_per_k > uninsulated_ua_ceiling_w_per_k
            or self.model.ua_bottom_w_per_k > uninsulated_ua_ceiling_w_per_k
        )

        if implausible_ua:
            logger.warning(
                "Boiler thermal validation: UA_top=%.2f or UA_bottom=%.2f W/K exceeds "
                "the loose uninsulated-tank ceiling (%.1f W/K) - check the model "
                "structure and ambient sensor placement before trusting this fit.",
                self.model.ua_top_w_per_k,
                self.model.ua_bottom_w_per_k,
                uninsulated_ua_ceiling_w_per_k,
            )

        # calibrate() already hard-bounds UA_mix_idle/UA_mix_active at the same
        # sampling-resolution ceiling (see NEGLIGIBLE_MIXING_RESIDUAL_FRACTION), so
        # landing near it means the fit wanted to go further - a sign the mixing
        # dynamics are not reliably captured by this data, not a genuine finding.
        c_node = RHO_WATER_KG_PER_L * (self.model.volume_l / 2.0) * CP_WATER_J_PER_KG_K
        ua_mix_ceiling = (c_node / (2.0 * median_dt)) * np.log(
            1.0 / self.NEGLIGIBLE_MIXING_RESIDUAL_FRACTION
        )

        implausible_mixing = (
            self.model.ua_mix_idle_w_per_k >= 0.9 * ua_mix_ceiling
            or self.model.ua_mix_active_w_per_k >= 0.9 * ua_mix_ceiling
        )

        if implausible_mixing:
            logger.warning(
                "Boiler thermal validation: UA_mix_idle=%.2f or UA_mix_active=%.2f "
                "W/K is pinned near the sampling-resolution ceiling (%.1f W/K) - "
                "the mixing dynamics are effectively unconstrained by this data, "
                "not a confirmed physical finding.",
                self.model.ua_mix_idle_w_per_k,
                self.model.ua_mix_active_w_per_k,
                ua_mix_ceiling,
            )

        # calibrate() anchors q_in_nominal_w to a calorimetric confidence interval
        # whenever enough direct measurements exist (see
        # MIN_CALORIMETRIC_Q_IN_SAMPLES); landing at its edge means the heating
        # timesteps lacking a valid override disagree with that direct
        # measurement, so the bound itself - not a free fit - determined this
        # value. Not necessarily wrong (the interval is centered on a real
        # measurement), but worth surfacing rather than reporting a std_error
        # that assumes an unconstrained optimum.
        implausible_q_in = False

        if self.q_in_calorimetric_bounds is not None:
            lower, upper = self.q_in_calorimetric_bounds
            width = upper - lower
            implausible_q_in = width > 0 and (
                self.model.q_in_nominal_w - lower <= 0.05 * width
                or upper - self.model.q_in_nominal_w <= 0.05 * width
            )

            if implausible_q_in:
                logger.warning(
                    "Boiler thermal validation: q_in_nominal_w=%.1f W is pinned "
                    "at the edge of its calorimetric confidence interval "
                    "[%.1f, %.1f] W - the heating timesteps lacking a valid "
                    "calorimetric override disagree with the directly measured "
                    "average, so this edge value (not a free fit) determines "
                    "it; investigate those timesteps if this matters for "
                    "planning.",
                    self.model.q_in_nominal_w,
                    lower,
                    upper,
                )

        # A parameter whose standard error exceeds its own point estimate is not
        # meaningfully pinned down by the data - report it explicitly rather than
        # letting a precise-looking number hide that (see calibrate()'s docstring
        # note on UA_mix_idle's identifiability from short idle windows).
        weakly_identified = 0

        if self.parameter_std_errors is not None:
            parameter_values = {
                "ua_top_w_per_k": self.model.ua_top_w_per_k,
                "ua_bottom_w_per_k": self.model.ua_bottom_w_per_k,
                "ua_mix_idle_w_per_k": self.model.ua_mix_idle_w_per_k,
                "ua_mix_active_w_per_k": self.model.ua_mix_active_w_per_k,
                "q_in_nominal_w": self.model.q_in_nominal_w,
            }

            for name, std_error in self.parameter_std_errors.items():
                if not np.isfinite(std_error) or std_error > abs(
                    parameter_values[name]
                ):
                    weakly_identified += 1
                    logger.warning(
                        "Boiler thermal validation: %s is weakly identified "
                        "(std error %.4g vs. estimate %.4g) - treat as order of "
                        "magnitude, not a precise value.",
                        name,
                        std_error,
                        parameter_values[name],
                    )

        result = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "r2_idle": idle_metrics["r2"],
            "mae_idle": idle_metrics["mae"],
            "rmse_idle": idle_metrics["rmse"],
            "r2_sww": sww_metrics["r2"],
            "mae_sww": sww_metrics["mae"],
            "rmse_sww": sww_metrics["rmse"],
            "implausible_ua": float(implausible_ua),
            "implausible_mixing": float(implausible_mixing),
            "implausible_q_in": float(implausible_q_in),
            "weakly_identified_parameters": float(weakly_identified),
        }

        if self.parameter_std_errors is not None:
            result.update(
                {
                    f"std_error_{name}": value
                    for name, value in self.parameter_std_errors.items()
                }
            )

        return result

    def excess_loss_w(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-timestep one-step-ahead unexplained heat-loss power (W) during idle
        periods, using the currently calibrated model - the same physical quantity
        already computed internally during calibrate()'s diagnostics, exposed here
        for reuse (e.g. as a training signal for a separate tap-draw forecaster).

        `df` must already be prepared (see prepare()). Returns a DataFrame with
        "time" and "excess_loss_w" columns, one row shorter than `df` (there is no
        prediction for the first row). Values are clipped at 0 (only genuine excess
        *loss* is meaningful for this purpose) and are NaN while boiler_on - a real
        draw during active heating is confounded with Q_in and out of scope, same
        as the calibration diagnostics. This does NOT prove tap-water usage any
        more than the calibration diagnostics do - it is the same unproven,
        candidate signal, just exposed per-timestep instead of only summarized.

        A real, additional confound (found via calibrate()'s own diagnostics on
        real data, not yet corrected for): UA_top/UA_bottom are constants
        (Newton's law of cooling, loss linear in T-T_ambient), but real
        natural-convection heat transfer to ambient air can scale
        super-linearly with T-T_ambient (the convective coefficient itself
        rises with temperature). On real data, the flag rate at a large vs.
        small starting T-T_ambient was 51.8% vs. 19.2% - a far larger gap than
        the presence-based (39.0% vs. 34.7%) or gradient-based (20.7% vs.
        50.3%, itself likely confounded by temperature - a small gradient
        often just means "recently heated, still hot") comparisons. This means
        a meaningful share of excess_loss_w is plausibly this modeling gap, not
        tap draws - not yet corrected, since representing it properly would
        make the ODE nonlinear in T, breaking the exact linear ZOH
        discretization this module (and the MPC optimizer) relies on
        throughout. Treat this quantity - and anything trained on it, e.g.
        TapForecaster - as a mix of both effects, not a pure tap-draw signal.
        """

        if self.model is None:
            raise RuntimeError(
                "Model must be calibrated before estimating excess loss."
            )

        T_top = df["T_top"].to_numpy(dtype=float)
        T_bottom = df["T_bottom"].to_numpy(dtype=float)
        T_ambient = df["T_ambient"].to_numpy(dtype=float)
        boiler_on = df["boiler_on"].to_numpy(dtype=bool)
        dt_seconds = df["dt_seconds"].to_numpy(dtype=float)
        q_in_override = (
            df["q_in_override_w"].to_numpy(dtype=float)
            if "q_in_override_w" in df.columns
            else None
        )

        predictions = _predict_next_states(
            self.model, T_top, T_bottom, T_ambient, boiler_on, dt_seconds, q_in_override
        )
        actual = np.column_stack([T_top[1:], T_bottom[1:]])

        # Positive residual = measured cooled more than the model predicts.
        residual_k = predictions - actual

        c_node = RHO_WATER_KG_PER_L * (self.model.volume_l / 2.0) * CP_WATER_J_PER_KG_K
        excess_energy_j = np.sum(residual_k, axis=1) * c_node
        excess_power_w = np.clip(excess_energy_j / dt_seconds[1:], 0.0, None)

        idle_mask = ~boiler_on[:-1]
        excess_power_w = np.where(idle_mask, excess_power_w, np.nan)

        return pd.DataFrame(
            {
                "time": df["time"].to_numpy()[1:],
                "excess_loss_w": excess_power_w,
            }
        )

    def dataset(self, config: Config) -> DatasetDefinition:
        self.volume_l = float(config.heat_pump.boiler.volume)
        self.presence_columns = [f"presence_{i}" for i in range(len(config.presence))]

        builder = (
            DatasetBuilder()
            .timeseries(
                "T_ambient",
                config.heat_pump.boiler.ambient_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_supply",
                config.heat_pump.supply_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_return",
                config.heat_pump.return_temperature,
                interval="5m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "flow_lpm",
                config.heat_pump.flow,
                interval="5m",
                aggregation="mean",
                # Unlike a state string, flow physically drops to ~0 as soon as the
                # circulation pump stops. If the sensor itself stops publishing
                # while idle (no raw points in a bucket), fill="previous" would
                # keep carrying forward the last active flow reading indefinitely -
                # observed directly in this installation's data (flow still showing
                # ~14 L/min minutes after state flipped to "Uit"). fill="none" (no
                # InfluxDB fill at all) leaves a genuine reporting gap as a real
                # gap instead of guessing at either extreme here - prepare()
                # resolves it using the compressor's own separately-reported
                # state: bridge a brief gap while state confirms it is still
                # active, but trust 0 the moment state reports idle, regardless of
                # how long ago the last reading was.
                fill="none",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="5m",
                aggregation="last",
                fill="previous",
            )
        )

        # Device trackers are an independent, exogenous signal (not derived from
        # boiler temperature at all): nobody home rules out a tap draw (barring a
        # timed appliance), so this can confirm genuinely draw-free idle windows in a
        # way no residual-based inference can. Optional - config.presence may be
        # empty if the user hasn't set trackers up.
        for name, sensor in zip(self.presence_columns, config.presence, strict=True):
            builder = builder.timeseries(
                name,
                sensor,
                interval="5m",
                aggregation="last",
                fill="previous",
            )

        return builder.build()
