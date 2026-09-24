import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.models import HeatPumpCOPModel
from domain.physics import CP_WATER_J_PER_KG_K, RHO_WATER_KG_PER_L
from features.boiler import booster_active
from features.dataset import DatasetBuilder
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)


class HeatPumpCOPIdentifier(SystemIdentifier[HeatPumpCOPModel]):
    """Identifies a per-mode (SWW/Verwarmen/Koelen - see `mode`) heat pump COP
    model: COP = eta_carnot * T_cond_K / (T_cond_K - T_evap_K), the Carnot COP
    scaled by a second-law efficiency factor. T_cond/T_evap are offset from the
    measured water supply temperature and outdoor air temperature by a fitted
    heat-exchanger approach temperature, since the refrigerant itself is never
    measured directly.

    A single combined fit across all modes would average over genuinely
    different compressor operating points (e.g. SWW targets a much higher
    supply temperature than Verwarmen), hiding real per-mode efficiency
    differences relevant to the optimizer's costing - hence one instance per
    mode rather than one shared model.

    HEATING MODES ONLY (HeatPumpStates.dhw and .heating). This does not
    describe cooling, and an instance for the cooling mode was tried and
    removed. Two
    independent reasons: prepare() requires delta_t_water = T_supply - T_return
    to be positive, which it never is while cooling (the supply is the colder
    side), so every genuine cooling row is discarded and only transitional
    artifacts survive; and the formula itself assumes the water is the hot
    side. Cooling reverses the roles - the chilled water is the evaporator and
    outdoor air the condenser - so its efficiency is EER = eta * T_evap_K /
    (T_cond_K - T_evap_K), with the useful output being heat removed. Applying
    the expression above to cooling gives T_cond < T_evap and hence a negative
    COP. A cooling model needs its own identifier, not another instance of
    this one.
    """

    TRAIN_RATIO = 0.80

    MIN_FLOW_LPM = 0.0
    # Watts, not kW - matches the rest of this codebase's convention (e.g.
    # boiler.py's q_in_nominal_w, MPCConfig.boiler_electrical_power_w) and the
    # Home Assistant sensor's own native unit, avoiding an unnecessary
    # kW<->W conversion.
    MIN_ELECTRICAL_POWER_W = 100.0

    # eta_carnot is the fraction of the theoretical (Carnot) COP a real
    # compressor achieves - a standard "second-law efficiency". Real
    # residential heat pumps typically land around 0.4-0.6; the bounds are a
    # loose sanity range (below ~5% would be a barely-functioning machine,
    # above 90% would exceed practical compressor limits), not the expected
    # value itself.
    MIN_ETA = 0.05
    MAX_ETA = 0.90
    INITIAL_ETA = 0.45

    # The refrigerant's condensing/evaporating temperature is never measured
    # directly - only the water supply and outdoor air temperatures are. These
    # represent the heat-exchanger "approach" (pinch) temperature: how much
    # hotter/colder the refrigerant must be than the water/air it exchanges
    # heat with.
    #
    # delta_t_cond and delta_t_evap are NOT both fit: the temperature lift
    # that dominates the COP formula's denominator is
    # T_supply - T_outdoor + delta_t_cond + delta_t_evap - the two approach
    # temperatures enter only as their sum there, and the numerator's much
    # weaker separate dependence on delta_t_cond alone is not enough to
    # resolve them individually on real data (confirmed: std errors an order
    # of magnitude larger than the estimates themselves for both, with
    # delta_t_cond pinned at its own upper sanity bound - a structural
    # non-identifiability, not a data quantity problem). delta_t_cond is
    # therefore fixed at a typical residential plate/coil heat-exchanger
    # approach (single digit Kelvin) rather than fitted, leaving delta_t_evap
    # to absorb the (well-identified) remaining total approach - eta_carnot
    # and delta_t_evap are simultaneously well separable, since one is a
    # multiplicative scale and the other a nonlinear (lift) effect.
    FIXED_DELTA_T_COND = 5.0

    # 0.5 K rules out a physically-impossible zero-approach (infinite
    # exchanger area); 20 K rules out an implausibly poor one. NOTE: raising
    # this bound to "let the fit converge inside it" was tried and rejected -
    # on real data, delta_t_evap pinned exactly at 20 K, then, after raising
    # the bound to 30 K, pinned exactly at 30 K instead (with eta_carnot
    # shifting by ~27% between the two fits). A parameter that keeps chasing
    # whatever ceiling is set is not converging to an interior optimum, so
    # there is no honest value to raise the bound to - the fit is not
    # evidence for a larger true approach temperature (see CLAUDE.md: "a
    # better fit is not evidence of a better physical model"). This bound is
    # a physical sanity check only; validate()'s pinned-at-bound warning is
    # the correct, permanent way to surface that delta_t_evap (and by
    # extension eta_carnot) is not reliably identified from this data, not
    # something to be tuned away. The chase itself came from the data, not
    # the model: start-up readings (see STARTUP) and a back-filled outdoor
    # temperature (see prepare()). Without them the same 90 days settle at
    # 16 +/- 1.5 K, well inside the bound.
    MIN_DELTA_T_EVAP = 0.5
    MAX_DELTA_T_EVAP = 20.0
    INITIAL_DELTA_T_EVAP = 5.0

    # Q_th (real, calorimetric thermal output - see prepare()'s diagnostic) at
    # HeatPumpCOPModel.POWER_FIT_T_LOW_C/HIGH_C comes from a line fitted to
    # real data rather than assumed constant (see _fit_q_th_line()): real data
    # confirmed Q_th is NOT constant across a cycle - using a single fixed
    # q_in_nominal_w (the boiler's own separately calibrated nominal thermal
    # output, validated for the tank's temperature *trajectory*, a different
    # purpose) understated real electrical draw by up to ~40% through the
    # middle of a cycle.
    # Rows below this supply temperature are left out of that Q_th fit: the
    # compressor is still ramping up there (real data: ~600 W electrical and
    # ~2.8 kW thermal at 20-30 degC supply, against ~6.5 kW thermal once
    # running), a start-up transient the planning line is never evaluated in -
    # MPCOptimizer maps tank temperature plus the reference margin to supply
    # temperature, in practice 45 degC and up. Estimating Q_th at the reference
    # points as local medians instead (+/-5 degC windows at 30 and 60 degC) was
    # dragged down by exactly those start-up rows and by the modulating-down
    # tail, and planned DHW power ~20% below real in every one of 20 runs.
    POWER_FIT_MIN_SUPPLY_C = 35.0
    # validate() treats readings further apart than this as separate compressor
    # runs: prepare() keeps only active readings at the dataset's 5-minute
    # interval, so more than two missing readings in a row means the
    # compressor stopped in between.
    RUN_GAP = pd.Timedelta(minutes=15)
    # The COP formula describes a compressor in steady operation. For the first
    # quarter hour after entering a mode it is not: the compressor ramps up
    # and the water circuit warms, so Q_th is still climbing (real DHW data,
    # median per 5 minutes since entering SWW: 2.1, 4.5 and 5.8 kW, then a
    # steady 6.6-7.0 kW - the boiler's own identified ramp is ~13 minutes).
    # Left in, those readings pulled delta_t_evap onto its upper bound; left
    # out from 15 minutes on, it settles at 12-14 K whether the cut is at 15,
    # 20 or 25 minutes.
    STARTUP = pd.Timedelta(minutes=15)

    def __init__(
        self, key: Literal["dhw", "heating"], models_path: Path | None = None
    ) -> None:
        super().__init__()
        # Which HeatPumpStates mode this instance fits, and its stable name
        # (see name()): independent of the raw state label, so a model keeps
        # its name when that label is configured differently.
        self.key = key
        # Where the DHW model is found, whose evaporator approach a heating fit
        # falls back on (see _shared_delta_t_evap).
        self.models_path = models_path
        # The configured state labels; overwritten by dataset().
        self.states = HeatPumpStates()
        self.parameter_std_errors: dict[str, float] | None = None

    @property
    def mode(self) -> str:
        """The heat_pump.state value this instance's data is filtered to."""

        return getattr(self.states, self.key)

    @property
    def name(self) -> str:
        return f"cop_{self.key}"

    @property
    def label(self) -> str:
        return f"COP ({self.key.upper()})"

    def _bridge_reporting_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """P_el and flow_lpm are rate-like: dataset() fetches them with no
        InfluxDB fill at all, so a real reporting gap shows up here as NaN
        rather than a guessed value. The compressor's own, separately and
        reliably reported state settles what a gap actually means: bridging
        forward while state confirms it is still running (confirmed on real
        data - T_supply/T_return kept rising smoothly through a 15-minute
        flow_lpm gap during a DHW ramp-up: a real reporting hiccup, not zero
        flow), but resetting to 0 the moment state reports idle regardless of
        how long ago the last reading was - the exact bug already found and
        fixed for flow_lpm in boiler.py (a stale nonzero reading persisting
        for minutes past a real, confirmed shutoff) applies equally to P_el
        here (confirmed on real data: a frozen, bit-identical P_el reading
        persisting for 20+ minutes into an idle period).
        """

        df = df.copy()
        is_off = df["state"] == self.states.off

        for column in ["P_el", "flow_lpm"]:
            df[column] = df[column].ffill()
            df[column] = df[column].where(~is_off, 0.0)
            # No prior reading at all (e.g. the very start of the fetched
            # window) - assume 0 rather than leaving it unresolved.
            df[column] = df[column].fillna(0.0)

        return df

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        # dataset() anchors the join on the 5-minute heat-pump readings (not
        # the other way around, unlike SolarBiasIdentifier - that identifier needs
        # the full forecast-snapshot history to learn how forecasts evolve
        # with lead time; COP only needs the best-known outdoor temperature at
        # each reading). The outdoor-temperature attribute only reports hourly
        # target times, so it matches at most 1 in 12 of these 5-minute rows
        # exactly - forward-filling holds that hourly value for the following
        # readings, a reasonable approximation since outdoor air temperature
        # changes slowly relative to an hour. Only forward: readings from before
        # the first outdoor temperature have none and are dropped below.
        # Back-filling gave them the first value ever recorded - on real data
        # half of all DHW readings, 45 days of them, at one and the same
        # 22.6 degC. A duplicate 5-minute reading can
        # still occur if the outdoor-temperature source reported more than one
        # overlapping forecast for the same hour (a real possibility - see
        # AttributeTimeSeriesLoader, which returns the full snapshot history,
        # not just the latest); reduce back to one row per reading first.
        if "temperature" in df.columns:
            df = df.sort_values("time").drop_duplicates(subset="time", keep="last")
            df["temperature"] = df["temperature"].ffill()

        df = df.rename(columns={"temperature": "T_outdoor"})

        required_columns = [
            "T_outdoor",
            "T_supply",
            "T_return",
            "flow_lpm",
            "P_el",
            "state",
        ]

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        numeric_columns = [
            "T_outdoor",
            "T_supply",
            "T_return",
            "flow_lpm",
            "P_el",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = self._bridge_reporting_gaps(df)

        # Before any reading is dropped, so a run's start is its real start.
        running = df["state"] == self.mode
        entered = df["time"].where(running & ~running.shift(fill_value=False))
        start_up = running & (df["time"] - entered.ffill() < self.STARTUP)

        df = df.dropna(subset=numeric_columns).copy()

        df["delta_t_water"] = df["T_supply"] - df["T_return"]

        # Calorimetric thermal output in Watts - identical formula to
        # boiler.py's own calorimetric Q_in override (reuses the same water
        # constants), kept in the same unit as P_el (config.heat_pump.power,
        # Watts - the Home Assistant convention for power sensors) so
        # COP_measured needs no unit conversion. Confirmed on real data: P_el
        # readings up to ~2600 during a DHW cycle are a normal electrical draw
        # in W, not kW - comparing them against a Q_th computed in kW had
        # made COP_measured come out ~1000x too small, never clearing MIN_COP
        # and discarding every real measurement.
        df["Q_th"] = (
            (RHO_WATER_KG_PER_L / 60.0)
            * CP_WATER_J_PER_KG_K
            * df["flow_lpm"]
            * df["delta_t_water"]
        )

        df["COP_measured"] = df["Q_th"] / df["P_el"]

        # A single combined fit across modes would average over genuinely
        # different compressor operating points (see class docstring) - only
        # this instance's own mode may contribute evidence to its fit.
        # A resistive booster heater follows entirely different physics (no
        # compressor, COP ~1 - real data: 1.37 kWh heat for 1.38 kWh
        # electrical) and must not be mixed into the heat pump's own COP fit.
        # Recognised with or without its own sensor (see booster_active) - it
        # engages above ~55 degC tank temperature, exactly the top of the
        # T_supply range this fit would otherwise treat as heat-pump behavior.
        booster = booster_active(df, self.mode)
        in_mode = df["state"] == self.mode

        valid = (
            in_mode
            & ~booster
            & ~start_up
            & (df["flow_lpm"] > self.MIN_FLOW_LPM)
            & (df["P_el"] > self.MIN_ELECTRICAL_POWER_W)
            & (df["delta_t_water"] > 0.0)
            & (df["Q_th"] > 0.0)
            & (df["COP_measured"] < HeatPumpCOPModel.MAX_COP)
        )

        invalid_count = int((~valid).sum())
        booster_count = int((in_mode & booster).sum())
        start_up_count = int((start_up & ~booster).sum())

        df = df.loc[valid].copy()

        logger.info(
            "Heat pump COP preparation (%s): %d valid points, %d points removed "
            "(%d of them booster heater, %d compressor start-up)",
            self.mode,
            len(df),
            invalid_count,
            booster_count,
            start_up_count,
        )

        if df.empty:
            raise ValueError(
                f"No valid heat-pump COP measurements remain for mode "
                f"'{self.mode}' after filtering."
            )

        # DIAGNOSTIC: confirms Q_th (real, calorimetric thermal output) is
        # not constant across a compressor run - it rises from a low start,
        # peaks mid-cycle, then falls as the compressor modulates down
        # approaching setpoint (confirmed on real data: using
        # BoilerThermalModel's fixed q_in_nominal_w - calibrated for the
        # tank's temperature *trajectory*, a different purpose - as a stand-
        # in for Q_th at MPCOptimizer's electrical-planning reference points
        # understated real electrical draw by up to ~40% through the middle
        # of a cycle). calibrate() below uses this same real Q_th, not
        # q_in_nominal_w, at its own two reference points. Kept permanently
        # (not a one-off diagnostic) so future calibrations keep surfacing
        # whether this still holds.
        # Median, not mean, for the same reason reference_supply_temperature_c
        # uses a percentile rather than the max: several bins show a std
        # comparable to a third of their own mean (real transients, and -
        # until HeatPumpConfig.booster has enough history - residual
        # booster-heater contamination), so a mean is more outlier-sensitive
        # than this diagnostic needs it to be.
        supply_bin_edges = [0.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, np.inf]
        supply_bin = pd.cut(df["T_supply"], bins=supply_bin_edges, right=False)
        q_th_summary = df.groupby(supply_bin, observed=True).agg(
            count=("Q_th", "count"),
            q_th_median=("Q_th", "median"),
            q_th_std=("Q_th", "std"),
            p_el_median=("P_el", "median"),
        )
        logger.info(
            "Heat pump COP diagnostic (%s): Q_th/P_el by T_supply bin:\n%s",
            self.mode,
            q_th_summary,
        )

        return df

    @staticmethod
    def _predict_cop(
        parameters: np.ndarray,
        T_outdoor: np.ndarray,
        T_supply: np.ndarray,
        delta_t_cond: float,
    ) -> np.ndarray:
        eta_carnot, delta_t_evap = parameters

        # Delegates to the same formula the calibrated model exposes (see
        # HeatPumpCOPModel.cop()) - a trial parameter vector during fitting
        # is just a not-yet-final model, so this avoids duplicating the
        # physics in two places. HeatPumpCOPModel.cop()'s plain arithmetic
        # works unchanged on the numpy arrays passed in here.
        # reference_supply_temperature_c is planning-only metadata that
        # cop() never reads - irrelevant to a trial fit, so left at its
        # default here.
        trial_model = HeatPumpCOPModel(
            eta_carnot=eta_carnot,
            delta_t_cond=delta_t_cond,
            delta_t_evap=delta_t_evap,
        )

        return trial_model.cop(T_outdoor, T_supply)

    def calibrate(
        self,
        df: pd.DataFrame,
    ) -> HeatPumpCOPModel:
        df = self.prepare(df)

        if len(df) < 10:
            raise ValueError("Not enough data points for calibration.")

        split_index = int(len(df) * self.TRAIN_RATIO)

        if split_index <= 0 or split_index >= len(df):
            raise ValueError("Invalid train/test split.")

        train_df = df.iloc[:split_index].copy()

        logger.info(
            "Heat pump COP calibration (%s): %d training points, %d validation points",
            self.mode,
            len(train_df),
            len(df) - len(train_df),
        )

        T_outdoor = train_df["T_outdoor"].to_numpy(dtype=float)

        T_supply = train_df["T_supply"].to_numpy(dtype=float)

        COP_measured = train_df["COP_measured"].to_numpy(dtype=float)

        def residuals(
            parameters: np.ndarray,
        ) -> np.ndarray:
            COP_predicted = self._predict_cop(
                parameters,
                T_outdoor,
                T_supply,
                self.FIXED_DELTA_T_COND,
            )

            return COP_predicted - COP_measured

        x0 = np.array(
            [
                self.INITIAL_ETA,
                self.INITIAL_DELTA_T_EVAP,
            ]
        )

        lower_bounds = np.array(
            [
                self.MIN_ETA,
                self.MIN_DELTA_T_EVAP,
            ]
        )

        upper_bounds = np.array(
            [
                self.MAX_ETA,
                self.MAX_DELTA_T_EVAP,
            ]
        )

        result = least_squares(
            residuals,
            x0=x0,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
        )

        if not result.success:
            logger.warning(
                "Heat pump COP calibration (%s) did not fully converge: %s",
                self.mode,
                result.message,
            )

        eta_carnot = float(result.x[0])
        delta_t_evap = float(result.x[1])

        std_errors = self._parameter_std_errors(result)
        parameter_names = ["eta_carnot", "delta_t_evap"]
        self.parameter_std_errors = dict(
            zip(parameter_names, std_errors.tolist(), strict=True)
        )

        shared = self._shared_delta_t_evap(delta_t_evap, float(std_errors[1]))

        if shared is not None:
            # With the approach fixed the COP is eta_carnot times the Carnot
            # COP, so eta_carnot is a least-squares solve through the origin.
            delta_t_evap = shared
            carnot = HeatPumpCOPModel(
                eta_carnot=1.0,
                delta_t_cond=self.FIXED_DELTA_T_COND,
                delta_t_evap=delta_t_evap,
            ).cop(T_outdoor, T_supply)
            eta_carnot = float(np.dot(carnot, COP_measured) / np.dot(carnot, carnot))
            residual = COP_measured - eta_carnot * carnot
            eta_std_error = float(
                np.sqrt(
                    np.dot(residual, residual)
                    / max(len(residual) - 1, 1)
                    / np.dot(carnot, carnot)
                )
            )
            std_errors = np.array([eta_std_error, 0.0])
            self.parameter_std_errors = {"eta_carnot": eta_std_error}

        logger.info(
            "Heat pump COP parameters calibrated (%s): "
            "eta_carnot=%.4f±%.4f, "
            "delta_t_cond=%.3f K (fixed), "
            "delta_t_evap=%.3f±%.3f K",
            self.mode,
            eta_carnot,
            std_errors[0],
            self.FIXED_DELTA_T_COND,
            delta_t_evap,
            std_errors[1],
        )

        # Planning (MPCOptimizer) needs a stand-in for the heat pump's actual
        # supply temperature - a quantity it has no forecast for - to
        # evaluate this model's COP. The boiler's own configured target
        # temperature is not that stand-in: T_supply must run hotter than
        # the tank it is charging for heat to keep flowing in, confirmed on
        # real data reaching a mean of ~50 degC late in a cycle against a
        # configured 45 degC target. The 95th percentile (not the max, to
        # limit sensitivity to a single noisy outlier reading) of this
        # mode's own observed T_supply is the direct, measured answer to
        # "how hot does this specific installation's heat pump actually run
        # a full SWW cycle", without inventing an unmeasured tank-to-supply
        # approach parameter.
        self.model = HeatPumpCOPModel(
            eta_carnot=eta_carnot,
            delta_t_cond=self.FIXED_DELTA_T_COND,
            delta_t_evap=delta_t_evap,
        )

        # The reference supply and the Q_th line are what DHW planning costs a
        # tank charge with, at 30-60 degC supply. Space heating is costed at
        # its heating curve's own supply with cop() alone (see
        # MPCOptimizer._space_cop), so a heating fit has no use for them - and
        # a line fitted at ~25-30 degC floor supply extrapolated to 60 degC
        # would mean nothing.
        if self.key != "dhw":
            return self.model

        self.model.reference_supply_temperature_c = float(df["T_supply"].quantile(0.95))
        # See the class docstring on POWER_FIT_T_LOW_C/HIGH_C and
        # POWER_FIT_MIN_SUPPLY_C. Fitted on the training split only, like the
        # COP parameters, so validate() scores it on data it never saw.
        (
            self.model.q_th_at_power_fit_low_w,
            self.model.q_th_at_power_fit_high_w,
        ) = self._fit_q_th_line(train_df, self.model)

        logger.info(
            "Heat pump COP calibration (%s): reference_supply_temperature_c="
            "%.2f degC, Q_th at %.0f/%.0f degC = %.1f/%.1f W",
            self.mode,
            self.model.reference_supply_temperature_c,
            HeatPumpCOPModel.POWER_FIT_T_LOW_C,
            HeatPumpCOPModel.POWER_FIT_T_HIGH_C,
            self.model.q_th_at_power_fit_low_w,
            self.model.q_th_at_power_fit_high_w,
        )

        return self.model

    def _shared_delta_t_evap(
        self, delta_t_evap: float, std_error: float
    ) -> float | None:
        """The DHW model's evaporator approach, for a heating fit that cannot
        pin down its own.

        Both modes draw heat through the same outdoor evaporator, so its
        approach is one physical quantity; what differs between them is the
        supply temperature, which eta_carnot and the condenser side carry.
        Separating the approach from eta_carnot takes a spread in lift, which
        a few heating runs at one floor supply do not have - there the DHW
        fit, over a far wider lift range, is the better estimate. None when
        the heating fit's own approach is identified, or when there is no DHW
        model to take it from.
        """

        if self.key == "dhw" or self.models_path is None:
            return None

        if np.isfinite(std_error) and std_error <= abs(delta_t_evap):
            return None

        dhw = HeatPumpCOPIdentifier(key="dhw")
        dhw.load(self.models_path)

        if dhw.model is None:
            return None

        logger.info(
            "Heat pump COP calibration (%s): delta_t_evap=%.2f+/-%.2f K is not "
            "identified here - using the DHW fit's %.2f K (same evaporator) "
            "and fitting eta_carnot alone.",
            self.mode,
            delta_t_evap,
            std_error,
            dhw.model.delta_t_evap,
        )

        return float(dhw.model.delta_t_evap)

    def _fit_q_th_line(
        self, df: pd.DataFrame, cop_model: HeatPumpCOPModel
    ) -> tuple[float, float]:
        """Q_th at POWER_FIT_T_LOW_C/HIGH_C from a line
        Q_th = q_0 + q_slope * T_supply, chosen so that Q_th / COP reproduces
        measured P_el as closely as possible - electrical power is what
        planning costs, so that is the error to minimize, not the Q_th error
        itself. Linear in (q_0, q_slope), so an ordinary least-squares solve.
        Uses only rows at or above POWER_FIT_MIN_SUPPLY_C (see its class
        docstring). Falls back to a flat Q_th when supply temperature barely
        varies (the slope is then not identifiable), and to all rows, with a
        warning, when fewer than two lie in the operating range.
        """

        rows = df[df["T_supply"] >= self.POWER_FIT_MIN_SUPPLY_C]

        if len(rows) < 2:
            logger.warning(
                "Heat pump COP calibration (%s): fewer than 2 readings at or "
                "above %.1f degC supply - fitting Q_th on all readings instead.",
                self.mode,
                self.POWER_FIT_MIN_SUPPLY_C,
            )
            rows = df

        T_supply = rows["T_supply"].to_numpy(dtype=float)
        P_el = rows["P_el"].to_numpy(dtype=float)
        inverse_cop = 1.0 / cop_model.clamped_cop(
            rows["T_outdoor"].to_numpy(dtype=float), T_supply
        )

        design = np.column_stack([inverse_cop, T_supply * inverse_cop])
        (q_0, q_slope), _, rank, _ = np.linalg.lstsq(design, P_el, rcond=None)

        if rank < 2:
            q_0 = float(np.dot(inverse_cop, P_el) / np.dot(inverse_cop, inverse_cop))
            q_slope = 0.0

        return (
            float(q_0 + q_slope * HeatPumpCOPModel.POWER_FIT_T_LOW_C),
            float(q_0 + q_slope * HeatPumpCOPModel.POWER_FIT_T_HIGH_C),
        )

    def _planned_power_w(self, df: pd.DataFrame) -> np.ndarray:
        """The electrical power MPCOptimizer plans with at each row's real
        supply and outdoor temperature. Its line is expressed in tank
        temperature there, but shifting by the constant reference margin
        leaves it this same straight line in supply-temperature terms."""

        power_low, power_high = self.model.planned_power_at_reference_points(
            df["T_outdoor"].to_numpy(dtype=float)
        )
        fraction = (
            df["T_supply"].to_numpy(dtype=float) - HeatPumpCOPModel.POWER_FIT_T_LOW_C
        ) / (HeatPumpCOPModel.POWER_FIT_T_HIGH_C - HeatPumpCOPModel.POWER_FIT_T_LOW_C)

        return power_low + (power_high - power_low) * fraction

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        df = self.prepare(df)

        if self.model is None:
            raise RuntimeError("Model must be calibrated before validation.")

        if len(df) < 2:
            raise ValueError("Not enough data points for validation.")

        split_index = int(len(df) * self.TRAIN_RATIO)

        test_df = df.iloc[split_index:].copy()

        if test_df.empty:
            raise ValueError("No validation data available.")

        T_outdoor = test_df["T_outdoor"].to_numpy(dtype=float)

        T_supply = test_df["T_supply"].to_numpy(dtype=float)

        COP_measured = test_df["COP_measured"].to_numpy(dtype=float)

        parameters = np.array(
            [
                self.model.eta_carnot,
                self.model.delta_t_evap,
            ]
        )

        COP_predicted = self._predict_cop(
            parameters,
            T_outdoor,
            T_supply,
            self.model.delta_t_cond,
        )

        r2 = float(
            r2_score(
                COP_measured,
                COP_predicted,
            )
        )

        mae = float(
            mean_absolute_error(
                COP_measured,
                COP_predicted,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    COP_measured,
                    COP_predicted,
                )
            )
        )

        logger.info(
            "Heat pump COP validation (%s): R2=%.4f, MAE=%.4f, RMSE=%.4f",
            self.mode,
            r2,
            mae,
            rmse,
        )

        logger.info(
            "Heat pump COP validation (%s): measured mean=%.3f, predicted mean=%.3f",
            self.mode,
            float(np.mean(COP_measured)),
            float(np.mean(COP_predicted)),
        )

        # delta_t_evap is only ever loosely sanity-bounded (see its class
        # docstring), not asserted as the true approach temperature - landing
        # at or near either bound means the fit wanted to go further, a sign
        # it is not reliably separable from eta_carnot on this data, not a
        # confirmed physical finding. delta_t_cond is fixed, not fitted, so
        # it can no longer be "pinned" - only delta_t_evap is checked here.
        # Same style as BoilerThermalIdentifier.validate()'s UA_mix ceiling
        # check.
        near_bound_fraction = 0.1

        def near_bound(value: float, lower: float, upper: float) -> bool:
            return value >= upper - near_bound_fraction * (
                upper - lower
            ) or value <= lower + near_bound_fraction * (upper - lower)

        implausible_delta_t = near_bound(
            self.model.delta_t_evap, self.MIN_DELTA_T_EVAP, self.MAX_DELTA_T_EVAP
        )

        if implausible_delta_t:
            logger.warning(
                "Heat pump COP validation (%s): delta_t_evap=%.2f K is "
                "pinned at or near its sanity bound ([%.1f, %.1f] K) - this "
                "approach temperature is not reliably separable from "
                "eta_carnot on this data, not a confirmed physical finding.",
                self.mode,
                self.model.delta_t_evap,
                self.MIN_DELTA_T_EVAP,
                self.MAX_DELTA_T_EVAP,
            )

        # A parameter whose standard error exceeds its own point estimate is
        # not meaningfully pinned down by the data - same convention as
        # BoilerThermalIdentifier.validate().
        weakly_identified = 0

        if self.parameter_std_errors is not None:
            parameter_values = {
                "eta_carnot": self.model.eta_carnot,
                "delta_t_evap": self.model.delta_t_evap,
            }

            for name, std_error in self.parameter_std_errors.items():
                if not np.isfinite(std_error) or std_error > abs(
                    parameter_values[name]
                ):
                    weakly_identified += 1
                    logger.warning(
                        "Heat pump COP validation (%s): %s is weakly "
                        "identified (std error %.4g vs. estimate %.4g) - "
                        "treat as order of magnitude, not a precise value.",
                        self.mode,
                        name,
                        std_error,
                        parameter_values[name],
                    )

        # An accurate COP fit does not guarantee accurate planning: the
        # optimizer costs with a straight power line through two Q_th
        # reference values (see _fit_q_th_line()), so that line is scored here
        # against measured power, in the supply range planning evaluates it in.
        # Only DHW planning has such a line (see calibrate()).
        in_planning_range = (T_supply >= self.POWER_FIT_MIN_SUPPLY_C) & (
            self.key == "dhw"
        )

        if in_planning_range.any():
            planned_power_w = self._planned_power_w(test_df)
            measured_power_w = test_df["P_el"].to_numpy(dtype=float)
            error_w = (planned_power_w - measured_power_w)[in_planning_range]
            power_bias_w = float(np.mean(error_w))
            power_mae_w = float(np.mean(np.abs(error_w)))

            runs = (test_df["time"].diff() > self.RUN_GAP).cumsum().to_numpy()
            power = pd.DataFrame(
                {
                    "run": runs,
                    "T_supply": T_supply,
                    "planned": planned_power_w,
                    "measured": measured_power_w,
                }
            )
            peaks = power.groupby("run")[["planned", "measured"]].max()
            run_peak_ratio = float((peaks["planned"] / peaks["measured"]).median())

            supply_bins = np.arange(
                self.POWER_FIT_MIN_SUPPLY_C,
                HeatPumpCOPModel.POWER_FIT_T_HIGH_C + 5.0,
                5.0,
            )
            by_supply = power.groupby(
                pd.cut(power["T_supply"], bins=supply_bins, right=False),
                observed=True,
            )[["planned", "measured"]].median()

            logger.info(
                "Heat pump COP validation (%s): planned vs measured power at "
                ">= %.0f degC supply: bias=%+.0f W, MAE=%.0f W, median run "
                "peak planned/measured=%.3f (%d runs); median per supply bin:\n%s",
                self.mode,
                self.POWER_FIT_MIN_SUPPLY_C,
                power_bias_w,
                power_mae_w,
                run_peak_ratio,
                len(peaks),
                by_supply.round(0).to_string(),
            )
        else:
            power_bias_w = power_mae_w = run_peak_ratio = float("nan")

        if self.key == "dhw" and not in_planning_range.any():
            logger.warning(
                "Heat pump COP validation (%s): no test readings at or above "
                "%.0f degC supply - planned power not scored.",
                self.mode,
                self.POWER_FIT_MIN_SUPPLY_C,
            )

        result = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "power_bias_w": power_bias_w,
            "power_mae_w": power_mae_w,
            "run_peak_ratio": run_peak_ratio,
            "implausible_delta_t": float(implausible_delta_t),
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

    def dataset(
        self,
        config: Config,
    ) -> DatasetDefinition:
        self.states = config.heat_pump.states

        # Outdoor air temperature, not the boiler's own room-ambient sensor
        # (features/boiler.py's T_ambient, a different physical quantity): for
        # an air-water heat pump, the evaporator draws heat from outdoor air.
        # attribute_series() (a single latest-snapshot lookup, ignores the
        # requested time range) cannot supply this historically -
        # attribute_timeseries() fetches the full history of forecast
        # snapshots instead (same mechanism SolarBiasIdentifier already relies on
        # for this exact sensor).
        #
        # Anchored on T_supply (a water temperature sensor - see below for why
        # that specific one, not P_el/flow_lpm), not on the outdoor-temperature
        # attribute - a real bug found on real data: anchoring on the
        # attribute's own hourly target times keeps only 1 in 12 of the
        # 5-minute readings (whichever happens to fall exactly on the hour),
        # and a real DHW cycle rarely lines up with the hour mark, so real DHW
        # rows were silently discarded. Unlike SolarBiasIdentifier (which
        # genuinely needs the full forecast-snapshot history to learn how
        # forecasts evolve with lead time), COP only needs the best-known
        # outdoor temperature at each reading - see prepare(), which
        # forward-fills between the hourly updates this join leaves mostly
        # unmatched.
        builder = (
            DatasetBuilder()
            # Water temperature sensors keep reporting regardless of
            # compressor state (they measure real, slowly-changing pipe
            # temperature, not a rate that vanishes at zero flow) - anchors
            # the join below.
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
            # P_el physically drops to ~0 the moment the compressor stops, and
            # (confirmed on real data: a frozen, bit-identical reading
            # persisting for 20+ minutes into an idle "Uit" period) this
            # installation's electrical-power sensor stops actively reporting
            # while idle - the same fill="previous"-on-a-rate-sensor bug
            # already found and fixed for flow_lpm in boiler.py. fill="none"
            # (no InfluxDB fill at all) leaves a genuine reporting gap as a
            # real gap instead of guessing at either extreme here - prepare()
            # resolves it using the compressor's own separately-reported
            # state: bridge a brief gap while state confirms it is still
            # running (also observed on real data: T_supply/T_return kept
            # rising smoothly through a 15-minute flow_lpm gap during a DHW
            # ramp-up - a real reporting hiccup, not zero flow), but trust 0
            # the moment state reports idle, regardless of how long ago the
            # last reading was.
            .timeseries(
                "P_el",
                config.heat_pump.power,
                interval="5m",
                aggregation="mean",
                fill="none",
            )
            # Same reporting-stops-when-idle behavior as P_el, resolved the
            # same way in prepare() - see the P_el comment above. boiler.py's
            # own dataset() hits the identical bug for this exact sensor
            # (~14 L/min still showing minutes after a real shutoff).
            .timeseries(
                "flow_lpm",
                config.heat_pump.flow,
                interval="5m",
                aggregation="mean",
                fill="none",
            )
            # An event-driven state string only changes on a real transition -
            # carrying the last known state forward through a reporting gap is
            # the physically correct assumption, same as boiler.py's own
            # state fetch.
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="5m",
                aggregation="last",
                fill="previous",
            )
            .attribute_timeseries(
                "open_meteo",
                config.forecast.open_meteo,
                attributes=["temperature"],
                interval="30m",
                aggregation="last",
            )
            .join(
                left="T_supply",
                right="T_return",
                on=("time",),
                how="left",
            )
            .join(
                left="T_supply",
                right="P_el",
                on=("time",),
                how="left",
            )
            .join(
                left="T_supply",
                right="flow_lpm",
                on=("time",),
                how="left",
            )
            .join(
                left="T_supply",
                right="state",
                on=("time",),
                how="left",
            )
            .join(
                left="T_supply",
                right="open_meteo",
                left_on=("time",),
                right_on=("target_time",),
                how="left",
            )
        )

        # Recognises the booster heater without its own sensor (see
        # booster_active). Only reported on change - also the change to 0 Hz
        # when the compressor stops - so the previous reading is the current
        # one: fill="none" hid booster rows sitting at a constant 0 Hz, and
        # fill=0 turned a compressor running at a constant frequency into false
        # booster rows (both confirmed on real data).
        builder = builder.timeseries(
            "compressor_frequency",
            config.heat_pump.compressor_frequency,
            interval="5m",
            aggregation="mean",
            fill="previous",
        ).join(
            left="T_supply",
            right="compressor_frequency",
            on=("time",),
            how="left",
        )

        # Optional: only some installations report this separately (see
        # HeatPumpConfig.booster). An event-driven on/off state, same
        # fill="previous" reasoning as "state" above.
        if config.heat_pump.booster is not None:
            builder = builder.timeseries(
                "booster",
                config.heat_pump.booster,
                interval="5m",
                aggregation="last",
                fill="previous",
            ).join(
                left="T_supply",
                right="booster",
                on=("time",),
                how="left",
            )

        return builder.build()
