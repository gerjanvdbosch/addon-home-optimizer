import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from domain.types import (
    BoilerThermalModel,
    HeatPumpCOPModel,
    MPCConfig,
    MPCInput,
    MPCResult,
)
from features.boiler import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    discretize_zoh,
    lumped_state_space,
)
from features.cop import HeatPumpCOPIdentifier

logger = logging.getLogger(__name__)

# The tank's maximum water temperature (deg C) until a booster run has shown the
# real one (see BoilerThermalIdentifier._identify_booster) - a typical limit for a
# domestic hot water tank, chosen for this installation.
DEFAULT_MAX_TANK_TEMPERATURE_C = 65.0

# How close to optimal a plan must be proven (EUR): one cent, the precision the
# price itself is given in. The solver's default tolerance is relative to the
# objective instead, which an unavoidable temperature shortfall inflates into the
# thousands - at 0.01% of that, a plan was accepted on real data that kept the
# heat pump 'on' without heating for 1.5 hours.
MIP_ABSOLUTE_GAP_EUR = 0.01

# Swanson's rule: the standard three-point weights for a distribution's
# expectation from its P10/P50/P90 (exact for a symmetric distribution, close
# for moderately skewed ones). Grid import is costed as this expectation over
# solar outcomes: import cost is convex in solar (max(0, P_el - solar)), so
# costing at P50 alone systematically understates the expected cost of
# relying on uncertain sun.
SOLAR_SCENARIO_WEIGHTS = (0.3, 0.4, 0.3)


@dataclass
class _StepPlan:
    """Maps MPCInput's native fine-resolution forecast arrays onto a coarser
    set of decision steps for the solver, and back - see
    MPCOptimizer._build_step_plan for why, and _extract_result for how the
    solved decisions are replayed back onto the full fine-resolution grid
    for reporting.
    """

    # fine_to_model[i] = which model step covers original fine index i.
    fine_to_model: list[int]
    # dt_hours[k] = real elapsed hours model step k covers.
    dt_hours: list[float]

    @property
    def num_steps(self) -> int:
        return len(self.dt_hours)


class MPCOptimizer:
    def __init__(
        self,
        thermal_model: BoilerThermalModel,
        config: MPCConfig,
        cop_model: HeatPumpCOPModel | None = None,
    ) -> None:
        self.thermal_model = thermal_model
        self.config = config
        # None until a cop_dhw model has actually been calibrated (see
        # HeatPumpCOPIdentifier) - _power_line_coefficients() falls back to
        # the flat boiler_electrical_power_w assumption until then.
        self.cop_model = cop_model

    def solve(self, data: MPCInput) -> MPCResult:
        self._validate_input(data)

        model = self._build_model(data)

        solver = Highs()
        solver.highs_options = {
            "mip_abs_gap": MIP_ABSOLUTE_GAP_EUR,
            "mip_rel_gap": 0.0,
        }
        results = solver.solve(model)

        if results.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(
                "MPC optimization failed. Termination condition: "
                f"{results.termination_condition}"
            )

        return self._extract_result(model, data, results.termination_condition)

    def _validate_input(self, data: MPCInput) -> None:
        horizon = len(data.solar_forecast_w)

        if horizon < 2:
            raise ValueError("MPC horizon must contain at least 2 steps.")

        if len(data.target_temperature_top) != horizon:
            raise ValueError(
                "target_temperature_top must have the same length as "
                f"solar_forecast_w ({horizon}), got "
                f"{len(data.target_temperature_top)}."
            )

        # Empty means "no forecast available" (treated as no draws elsewhere) -
        # only a non-empty, mismatched length is an actual bug.
        if data.tap_forecast_w and len(data.tap_forecast_w) != horizon:
            raise ValueError(
                "tap_forecast_w must be empty or have the same length as "
                f"solar_forecast_w ({horizon}), got {len(data.tap_forecast_w)}."
            )

        # Same "empty means no forecast" convention as tap_forecast_w above.
        if (
            data.outdoor_temperature_forecast
            and len(data.outdoor_temperature_forecast) != horizon
        ):
            raise ValueError(
                "outdoor_temperature_forecast must be empty or have the same "
                f"length as solar_forecast_w ({horizon}), got "
                f"{len(data.outdoor_temperature_forecast)}."
            )

        if len(data.solar_p10_w) != len(data.solar_p90_w) or (
            data.solar_p10_w and len(data.solar_p10_w) != horizon
        ):
            raise ValueError(
                "solar_p10_w and solar_p90_w must both be empty or both have "
                f"the same length as solar_forecast_w ({horizon}), got "
                f"{len(data.solar_p10_w)} and {len(data.solar_p90_w)}."
            )

        if data.heating_elapsed_hours < 0:
            raise ValueError("heating_elapsed_hours cannot be negative.")

        if data.baseload_forecast_w and len(data.baseload_forecast_w) != horizon:
            raise ValueError(
                "baseload_forecast_w must be empty or have the same length as "
                f"solar_forecast_w ({horizon}), got {len(data.baseload_forecast_w)}."
            )

        if self.config.step_hours <= 0:
            raise ValueError("step_hours must be greater than zero.")

        if self.config.boiler_electrical_power_w < 0:
            raise ValueError("boiler_electrical_power_w cannot be negative.")

        if self.config.boiler_min_runtime_steps < 1:
            raise ValueError("boiler_min_runtime_steps must be at least 1.")

        if self.config.fine_horizon_hours < 0:
            raise ValueError("fine_horizon_hours cannot be negative.")

        if self.config.coarse_step_hours <= 0:
            raise ValueError("coarse_step_hours must be greater than zero.")

    def _build_step_plan(self, horizon: int) -> _StepPlan:
        """See _StepPlan's docstring for why. fine_horizon_hours=0 disables
        coarsening entirely (every step stays at step_hours resolution) - a
        valid, simplest-possible configuration, not a special case requiring
        its own branch here (the loop below just runs zero fine steps).
        """

        step_hours = self.config.step_hours

        fine_steps = min(
            horizon, max(0, round(self.config.fine_horizon_hours / step_hours))
        )

        fine_to_model = list(range(fine_steps))
        dt_hours = [step_hours] * fine_steps

        coarse_span = max(1, round(self.config.coarse_step_hours / step_hours))

        i = fine_steps

        while i < horizon:
            block = min(coarse_span, horizon - i)
            model_step = len(dt_hours)
            fine_to_model.extend([model_step] * block)
            dt_hours.append(block * step_hours)
            i += block

        return _StepPlan(fine_to_model=fine_to_model, dt_hours=dt_hours)

    def _aggregate(
        self,
        values: Sequence[float],
        plan: _StepPlan,
        reducer: Callable[[list[float]], float],
    ) -> list[float]:
        buckets: list[list[float]] = [[] for _ in range(plan.num_steps)]

        for i, value in enumerate(values):
            buckets[plan.fine_to_model[i]].append(float(value))

        return [reducer(bucket) for bucket in buckets]

    def _build_model(self, data: MPCInput) -> pyo.ConcreteModel:
        horizon = len(data.solar_forecast_w)
        plan = self._build_step_plan(horizon)
        num_steps = plan.num_steps

        def mean(values: list[float]) -> float:
            return sum(values) / len(values)

        scenarios = (
            zip(
                SOLAR_SCENARIO_WEIGHTS,
                (data.solar_p10_w, data.solar_forecast_w, data.solar_p90_w),
                strict=True,
            )
            if data.solar_p10_w
            else [(1.0, data.solar_forecast_w)]
        )
        # Only solar beyond the rest of the house's own draw (the baseload) is
        # available to the heat pump - taken per fine step, before a coarse block
        # is averaged, since the surplus is not linear in solar.
        baseload_w = data.baseload_forecast_w or (0.0,) * horizon
        solar_scenarios = [
            (
                weight,
                self._aggregate(
                    [
                        max(0.0, float(solar) - float(baseload))
                        for solar, baseload in zip(values, baseload_w, strict=True)
                    ],
                    plan,
                    mean,
                ),
            )
            for weight, values in scenarios
        ]
        tap_w = self._aggregate(
            data.tap_forecast_w or (0.0,) * horizon, plan, mean
        )
        outdoor_c = (
            self._aggregate(data.outdoor_temperature_forecast, plan, mean)
            if data.outdoor_temperature_forecast
            else []
        )
        # max, not mean: a real requirement anywhere within a coarse block
        # must not be silently relaxed by averaging it with a quieter
        # neighboring step.
        target_c = self._aggregate(data.target_temperature_top, plan, max)
        overall_target_max = max(data.target_temperature_top)
        heat_pump_max_c = self.thermal_model.heat_pump_max_tank_temperature_c
        identified_max_c = self.thermal_model.max_tank_temperature_c
        max_tank_c = (
            identified_max_c
            if identified_max_c is not None
            else DEFAULT_MAX_TANK_TEMPERATURE_C
        )
        booster_heat_w = self.thermal_model.booster_heat_w

        model = pyo.ConcreteModel()

        model.K = pyo.RangeSet(0, num_steps - 1)

        initial_temperature = (data.current_temp_top + data.current_temp_bottom) / 2.0

        # Exact zero-order-hold dynamics for the lumped tank node. Unlike the
        # two-node calibration model, this simplified model has no on/off
        # mixing-regime switch (only the heat input, not the loss
        # coefficient, depends on boiler_on), so (A_d, B_d) depends only on
        # each step's own duration - cached per distinct duration (fine vs.
        # coarse - see _build_step_plan) rather than recomputed per step.
        a, b = lumped_state_space(
            self.thermal_model.volume_l,
            self.thermal_model.ua_top_w_per_k + self.thermal_model.ua_bottom_w_per_k,
        )
        discretization_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        def discretized(dt_hours: float) -> tuple[np.ndarray, np.ndarray]:
            if dt_hours not in discretization_cache:
                discretization_cache[dt_hours] = discretize_zoh(a, b, dt_hours * 3600.0)

            return discretization_cache[dt_hours]

        ambient_c = float(data.ambient_temperature)

        def unheated(start_c: float) -> list[float]:
            """The tank temperature at each step with no heating from start_c on."""

            temperatures = [start_c]

            for k in range(num_steps - 1):
                a_d, b_d = discretized(plan.dt_hours[k])
                passive = b_d[0, 0] * ambient_c + b_d[0, 2] * float(tap_w[k])
                temperatures.append(float(a_d[0, 0] * temperatures[-1] + passive))

            return temperatures

        # The booster only matters where the tank may need to be above the heat
        # pump's limit: it is the only source that heats there. That is so for a
        # target above what a tank at that limit now would have cooled to by
        # then - a run that ends at the limit, even any earlier (sunnier) one,
        # cannot hold such a target without it. Below that, with a tank that
        # starts below the limit too, running the booster could only ever cost
        # money. Leaving it out of the model there is what keeps the solve quick:
        # its 112 extra decisions over a 40-hour horizon took the real model from
        # 6 to 34 seconds. Only counting targets above the limit itself left out
        # a midday run on the sun with a booster finish (real data: a 55 degC
        # target at 18:00 against a 55.4 degC limit was heated in the evening on
        # grid power, a third dearer).
        booster_possible = (
            booster_heat_w is not None
            and heat_pump_max_c is not None
            and (
                initial_temperature > heat_pump_max_c
                or any(
                    target > limit
                    for target, limit in zip(
                        target_c, unheated(heat_pump_max_c), strict=True
                    )
                )
            )
        )

        # The tank cannot get hotter than whichever source can heat it: the
        # boiler's maximum plus the single booster step it may be cut out in
        # (it cannot modulate), or the heat pump's own limit without it - nor
        # colder at the start than it already is.
        # Over the longest step, not the shortest: beyond fine_horizon_hours a
        # step spans a whole coarse block, and a bound that only fits a 15-minute
        # booster step would rule the booster out there entirely - a far-away
        # legionella target then looked unreachable and the plan pre-heated a day
        # early instead.
        booster_step_k = (
            (booster_heat_w or 0.0)
            * max(plan.dt_hours)
            * 3600.0
            / (RHO_WATER_KG_PER_L * self.thermal_model.volume_l * CP_WATER_J_PER_KG_K)
        )
        if booster_possible:
            reachable_c = max_tank_c + booster_step_k
        elif heat_pump_max_c is not None:
            reachable_c = min(heat_pump_max_c, max_tank_c)
        else:
            reachable_c = max_tank_c
        t_upper = max(initial_temperature, reachable_c)

        # Per-step temperature bounds that follow from those dynamics alone. The
        # step response is monotone in the heat input (A_d, B_d > 0) and heat input
        # is never negative, so the trajectory with no heating at all is a floor
        # for every step, and the one with full heat every step - capped at the
        # ceiling above - is a ceiling. They describe exactly the same model, but
        # the big-M constraints below are built from them: with one loose range
        # for every step, the solver's relaxation could run the booster at a
        # fraction everywhere and had to branch on nearly every step of it.
        max_heat_w = max(self.thermal_model.q_in_nominal_w, booster_heat_w or 0.0)
        t_floor = unheated(initial_temperature)
        t_ceiling = [initial_temperature]

        for k in range(num_steps - 1):
            a_d, b_d = discretized(plan.dt_hours[k])
            passive = b_d[0, 0] * ambient_c + b_d[0, 2] * float(tap_w[k])
            heated = a_d[0, 0] * t_ceiling[-1] + passive + b_d[0, 1] * max_heat_w
            t_ceiling.append(min(t_upper, float(heated)))

        model.T = pyo.Var(model.K, bounds=lambda m, k: (t_floor[k], t_ceiling[k]))

        model.boiler_on = pyo.Var(model.K, domain=pyo.Binary)

        model.boiler_start = pyo.Var(model.K, domain=pyo.Binary)

        # Heat pump heat (W): continuous up to q_in_nominal_w while on. Below
        # that, a step is only partly used: the heat pump stops by itself once
        # the tank reaches its setpoint (published per run - see
        # app.optimization), or modulates down near its tank limit (real data:
        # ~7 kW at 36-47 degC, ~3 kW at 55 degC).
        model.q_heat_pump_w = pyo.Var(
            model.K, bounds=(0.0, self.thermal_model.q_in_nominal_w)
        )

        # The booster heater is a resistive element: it cannot modulate, so it is
        # all-or-nothing at its identified rating.
        model.booster_on = pyo.Var(model.K, domain=pyo.Binary)

        model.slack = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        # initial_temperature (above) rests on the equal-volume-node assumption,
        # same as the calibrated identification model: the average of the two
        # measured sensors approximates the tank's current total stored thermal
        # energy per unit mass.
        model.initial_temperature = pyo.Constraint(
            expr=model.T[0] == float(initial_temperature)
        )

        # A target above a step's ceiling cannot be met by any plan; that part of
        # the shortfall is the same constant whatever is decided, so it is left
        # out and the slack only measures what planning can still change. The
        # optimum is the same plan - but an unreachable legionella target had
        # inflated the objective into the thousands, and proving a plan to the
        # cent against that took the solver half a minute.
        def temperature_rule(m: pyo.ConcreteModel, k: int):
            return m.T[k] + m.slack[k] >= min(float(target_c[k]), t_ceiling[k])

        model.temperature_constraint = pyo.Constraint(
            model.K,
            rule=temperature_rule,
        )

        initial_boiler_on = int(data.boiler_on_current)

        # Inequality, not equality: boiler_start must be 1 on a real 0->1
        # transition (RHS=1, forcing boiler_start[k]>=1), but on a 1->0 stop the
        # RHS is -1 and boiler_start[k]=0 already satisfies ">=-1" trivially. An
        # equality here would force boiler_start=-1 on every stop, which is
        # infeasible against its own binary domain - making any schedule that
        # ever turns the boiler back off unsolvable, and forcing it to stay on
        # forever once started (confirmed: this was the actual cause of an
        # apparently-wasteful "never stops heating" result before this fix).
        # weight_switching in the objective still drives it to 0 except at real
        # starts, since setting it higher only adds cost.
        # A run is a start of heating by either source: a booster-only run is
        # still a DHW run the heat pump has to start, and handing over from the
        # compressor to the booster within a run is not a second start (the two
        # never heat together - see heat_source_constraints). Counting only
        # compressor starts once let a free night-time booster run beat a
        # cheaper solar heat pump run the next day on its start cost alone.
        def heating(m: pyo.ConcreteModel, k: int):
            return m.boiler_on[k] + m.booster_on[k]

        def startup_rule(m: pyo.ConcreteModel, k: int):
            if k == 0:
                return m.boiler_start[k] >= (heating(m, k) - initial_boiler_on)

            return m.boiler_start[k] >= (heating(m, k) - heating(m, k - 1))

        model.startup_constraint = pyo.Constraint(
            model.K,
            rule=startup_rule,
        )

        # Only enforced with a `start` in the fine-resolution region: a
        # single coarse step already spans far more real time than any
        # sensible minimum runtime (see MPCConfig.coarse_step_hours), so it
        # is trivially satisfied there without an explicit constraint.
        model.minimum_runtime = pyo.ConstraintList()

        min_runtime = self.config.boiler_min_runtime_steps
        fine_steps = sum(1 for dt in plan.dt_hours if dt <= self.config.step_hours)

        for start in range(fine_steps):
            for offset in range(min_runtime):
                k = start + offset

                if k >= num_steps:
                    continue

                model.minimum_runtime.add(
                    heating(model, k) >= model.boiler_start[start]
                )

        # A run already heating keeps heating until its minimum runtime has
        # passed. Otherwise the next replan, minutes after the start, can plan it
        # off at step 0 while the heat pump finishes the run anyway - the plan,
        # and anything acting on it, would flip for nothing. Skipped where the
        # model cannot represent that run: a tank above the heat pump's limit
        # with no booster to plan with.
        remaining_steps = math.ceil(
            (min_runtime * self.config.step_hours - data.heating_elapsed_hours)
            / self.config.step_hours
            - 1e-9
        )
        unrepresentable = (
            heat_pump_max_c is not None
            and booster_heat_w is None
            and initial_temperature >= heat_pump_max_c
        )
        model.running_run = pyo.ConstraintList()

        if data.boiler_on_current and not unrepresentable:
            for k in range(min(max(remaining_steps, 0), num_steps)):
                model.running_run.add(heating(model, k) >= 1)

        # The heat pump stays off for heat_pump_min_off_steps after a run ends,
        # and does not start at all until that long after the last real run
        # ended. Only enforced in the fine region, like the minimum runtime
        # above: a coarse step already spans longer than the pause.
        model.minimum_off_time = pyo.ConstraintList()
        min_off = self.config.heat_pump_min_off_steps

        for stop in range(fine_steps):
            for offset in range(1, min_off + 1):
                k = stop + offset

                if k >= num_steps:
                    continue

                model.minimum_off_time.add(
                    heating(model, k)
                    <= 1 - (heating(model, stop) - heating(model, stop + 1))
                )

        if not data.boiler_on_current:
            waiting_steps = math.ceil(
                (min_off * self.config.step_hours - data.idle_elapsed_hours)
                / self.config.step_hours
                - 1e-9
            )

            for k in range(min(max(waiting_steps, 0), num_steps)):
                model.minimum_off_time.add(heating(model, k) == 0)

        model.thermal_dynamics = pyo.ConstraintList()

        for k in range(num_steps - 1):
            a_d, b_d = discretized(plan.dt_hours[k])

            model.thermal_dynamics.add(
                model.T[k + 1]
                == a_d[0, 0] * model.T[k]
                + b_d[0, 0] * float(data.ambient_temperature)
                + b_d[0, 1] * model.q_heat_pump_w[k]
                + b_d[0, 1] * (booster_heat_w or 0.0) * model.booster_on[k]
                + b_d[0, 2] * float(tap_w[k])
            )

        # The heat pump cannot lift the tank past its own limit; above it only
        # the booster heats, never together with the compressor (real data: 0 Hz
        # throughout). Both identified from booster runs (see
        # BoilerThermalIdentifier._identify_booster) - until one has been
        # observed, planning stays heat-pump-only and unlimited, as before.
        model.heat_source_constraints = pyo.ConstraintList()
        heat_pump_limit_c = (
            min(heat_pump_max_c, max_tank_c)
            if heat_pump_max_c is not None
            else max_tank_c
        )

        # The tank temperature while each source runs, 0 otherwise: the products
        # T * boiler_on and T * booster_on, linearized exactly for on/off
        # decisions within the step's floor and ceiling (McCormick). The limits
        # below are stated on these - exact for a real plan, and tight in the
        # solver's relaxation. Stated on T itself with a big-M instead, a heat
        # pump at a fraction of 'on' could heat past its limit there, so the
        # relaxation never needed the dearer booster: its bound stayed near 0
        # against a real optimum of 0.45 EUR, and a booster day took half a
        # minute to prove - the booster decisions being the hard part.
        model.heat_pump_temperature = pyo.Var(model.K)
        model.booster_temperature = pyo.Var(model.K)

        def add_product(running_c, decision, k: int) -> None:
            for bound in (
                running_c <= t_ceiling[k] * decision,
                running_c >= t_floor[k] * decision,
                running_c <= model.T[k] - t_floor[k] * (1 - decision),
                running_c >= model.T[k] - t_ceiling[k] * (1 - decision),
            ):
                model.heat_source_constraints.add(bound)

        for k in range(num_steps):
            on = model.boiler_on[k]
            model.heat_source_constraints.add(
                model.q_heat_pump_w[k] <= self.thermal_model.q_in_nominal_w * on
            )
            add_product(model.heat_pump_temperature[k], on, k)

            # The heat pump runs at its own power until the setpoint and then
            # stops, so a step it still runs after is a full one: only a run's
            # last step is partly used. Otherwise the plan could spread a run's
            # heat over part-used steps to follow the sun, which the heat pump
            # never does. With the minimum runtime this also keeps a run from
            # delivering less than one full step.
            if k + 1 < num_steps:
                model.heat_source_constraints.add(
                    model.q_heat_pump_w[k]
                    >= self.thermal_model.q_in_nominal_w
                    * (on + model.boiler_on[k + 1] - 1)
                )

            # The heat pump modulates, so it stops exactly at its limit: the tank
            # temperature after a step it runs in - the dynamics below, times on -
            # stays under it. Skipped where the step's ceiling already rules that
            # out.
            if k + 1 < num_steps and t_ceiling[k + 1] > heat_pump_limit_c:
                a_d, b_d = discretized(plan.dt_hours[k])
                passive = b_d[0, 0] * ambient_c + b_d[0, 2] * float(tap_w[k])
                model.heat_source_constraints.add(
                    a_d[0, 0] * model.heat_pump_temperature[k]
                    + passive * on
                    + b_d[0, 1] * model.q_heat_pump_w[k]
                    <= heat_pump_limit_c * on
                )

            # Fixed, not merely bounded to zero: a variable the solver can
            # presolve away comes back without a value at all. Also where this
            # step's ceiling shows the tank cannot be above the heat pump's limit
            # yet - the booster could not run there in any plan.
            if (
                not booster_possible
                or heat_pump_max_c is None
                or t_ceiling[k] < heat_pump_max_c
            ):
                model.booster_on[k].fix(0)
                model.booster_temperature[k].fix(0.0)
                continue

            booster = model.booster_on[k]
            model.heat_source_constraints.add(on + booster <= 1)
            # The booster only runs above the heat pump's limit and below the
            # boiler's maximum, where the tank's thermostat cuts it out. Being
            # all-or-nothing, it may overshoot within the step it is cut out in.
            add_product(model.booster_temperature[k], booster, k)
            model.heat_source_constraints.add(
                model.booster_temperature[k] >= heat_pump_max_c * booster
            )
            model.heat_source_constraints.add(
                model.booster_temperature[k] <= max_tank_c * booster
            )
            # And only as the continuation of a run already heating: the booster
            # takes over from a compressor that cannot lift the tank any further,
            # it never starts a DHW run by itself.
            previous = (
                float(data.boiler_on_current) if k == 0 else heating(model, k - 1)
            )
            model.heat_source_constraints.add(booster <= previous)

        # active_power_w[k, s] is the grid draw at step k if solar scenario s
        # comes true: max(0, electrical power - solar). One schedule is shared by
        # all scenarios (the plan cannot know which one will happen; replanning
        # every step corrects course once it does), and only the cost differs
        # between them.
        #
        # A full heat pump step draws alpha * on + beta * T * on (see
        # _power_line_coefficients), with T * on the exact product
        # heat_pump_temperature (see heat_source_constraints) - so a heat pump
        # at a fraction of 'on' in the solver's relaxation also draws its share.
        # A partly used step draws for the part it runs, (alpha + beta * T) *
        # q / q_in_nominal_w, which is not linear. The unused part is instead
        # subtracted at the step's floor temperature, which never overstates the
        # saving: exact for a full step, a little too dear for a partial one.
        # Costed at full power instead, a step once on made the rest of its heat
        # free, so the plan heated past the target on grid power as well.
        model.S = pyo.RangeSet(0, len(solar_scenarios) - 1)
        model.active_power_w = pyo.Var(model.K, model.S, domain=pyo.NonNegativeReals)

        model.active_power_constraint = pyo.ConstraintList()

        power_lines = [
            self._power_line_coefficients(
                outdoor_c[k] if outdoor_c else None, overall_target_max
            )
            for k in range(num_steps)
        ]

        for k in range(num_steps):
            alpha, beta = power_lines[k]
            unused = (
                model.boiler_on[k]
                - model.q_heat_pump_w[k] / self.thermal_model.q_in_nominal_w
            )
            heat_pump_power_w = (
                alpha * model.boiler_on[k]
                + beta * model.heat_pump_temperature[k]
                - (alpha + beta * t_floor[k]) * unused
            )

            # The booster is a resistive element: its electrical power equals its
            # heat (COP 1 - real data: 1.37 kWh heat for 1.38 kWh electrical).
            #
            # For a real plan the grid draw is max(0, power - sun) of whichever
            # source runs (never both), for the part of the step it runs, 0 while
            # neither does: a heat pump running only part of a step uses only
            # that part's sun, the rest is exported. Counting the whole step's sun
            # made a partly used step's heat look free up to the sun's power.
            #
            # It is stated as power - sun * running: the same for a real plan, and
            # the tightest linear form in the solver's relaxation, where a
            # fraction of a source may only count on the same fraction of the
            # sun. Setting each
            # source against the whole sun separately let a few percent of both
            # run on it for free there, and the relaxation's bound stayed so far
            # below any real plan that a booster day took up to a minute.
            electrical_w = (
                heat_pump_power_w + (booster_heat_w or 0.0) * model.booster_on[k]
            )
            running = (
                model.q_heat_pump_w[k] / self.thermal_model.q_in_nominal_w
                + model.booster_on[k]
            )

            for s, (_, solar_w) in enumerate(solar_scenarios):
                solar_available_w = max(0.0, float(solar_w[k]))

                model.active_power_constraint.add(
                    model.active_power_w[k, s]
                    >= electrical_w - solar_available_w * running
                )

        # Heat left in the tank when the horizon ends has no value here, so the
        # plan heats only as far as the targets it can see need. A value priced
        # as the grid heat it would save later made the plan fill a run's last,
        # mostly grid-powered step for less than a cent (real data: 51 instead of
        # 46 degC ahead of a 45 degC target) - while this household's runs
        # mostly have sun, so later heat is rarely pure grid heat. Heat needed
        # after tomorrow's targets is planned once it comes within the horizon;
        # free surplus sun is not stored for demand beyond it.
        model.objective = pyo.Objective(
            expr=self._build_objective(
                model, plan, [weight for weight, _ in solar_scenarios]
            ),
            sense=pyo.minimize,
        )

        model.mpc_step_plan = plan

        return model

    def _power_line_coefficients(
        self, T_outdoor: float | None, overall_target_max: float
    ) -> tuple[float, float]:
        """Coefficients (alpha, beta) of a linear approximation
        electrical_power_w[k] = alpha + beta * T for step k, where T stands
        for the MPC's own tank temperature at that step (model.T[k]).

        The true relationship (via HeatPumpCOPModel.cop(), a ratio of
        temperatures) is concave in T, not linear - and a concave function
        cannot be represented by inequality "tangent cut" constraints under
        minimization the way a convex one can (tangent lines of a concave
        function are upper bounds; minimizing gives the solver no pressure
        to rise to meet them, so it would just drive the variable to 0
        instead of the true curve). Representing it exactly would need a
        real piecewise-linear formulation (SOS2 or binary-selected
        segments), adding real complexity. Real data on this installation
        confirmed electrical power is very close to linear in supply
        temperature over a DHW cycle's normal active-heating range
        (HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C to ...HIGH_C) - so a single
        secant line through the two ends of that range is an adequate, much
        simpler stand-in, and (being a genuine straight line, not a bound)
        is usable directly inside the objective, exactly as used for
        reporting (see _extract_result) - the same numbers the optimizer
        actually costs with are the ones shown.

        Q_th at each reference point comes from the calibrated
        q_th_at_power_fit_low_w/high_w (a line fitted to real calorimetric
        thermal output so it reproduces measured electrical power - see
        HeatPumpCOPIdentifier._fit_q_th_line()), not BoilerThermalModel's fixed
        q_in_nominal_w: real data confirmed Q_th is not constant across a
        compressor run (it rises from a low start, peaks mid-cycle, then
        falls as the compressor modulates down approaching setpoint), and
        q_in_nominal_w - calibrated for the tank's temperature *trajectory*,
        a different purpose - understated real electrical draw through the
        middle of a cycle by using a constant well below the true mid-cycle
        Q_th.

        T[k] (rather than a single fixed reference temperature) is what
        this line is ultimately evaluated against: the heat pump's actual
        supply temperature physically tracks the tank it is currently
        charging (it must stay hotter to keep pushing heat in), which is
        exactly what T[k] represents, already decision-consistent with the
        rest of the model. The margin between T[k] and the real supply
        temperature is not directly measured (cop_dhw has no
        tank-temperature column to calibrate it against), so it is
        approximated as reference_supply_temperature_c's own margin above
        the highest configured target - the same "how much hotter does
        supply run than the target it's aiming for" gap already implied by
        that calibrated value, floored at 0 so supply is never modelled as
        colder than the tank it is heating. POWER_FIT_T_LOW_C/HIGH_C are
        real T_supply values (q_th_at_power_fit_low_w/high_w are that fitted
        line evaluated at those *real* T_supply values - see
        HeatPumpCOPIdentifier._fit_q_th_line()), so COP is evaluated directly at
        those values, with no margin added there - the margin only enters
        when re-expressing the resulting (T_supply -> power) line in T[k]
        terms below (T_supply = T[k] + margin, so T[k] = T_supply - margin
        at each reference point); adding it a second time inside the COP
        evaluation itself would evaluate COP at an unrealistically hot,
        never-measured T_supply, understating COP and so overstating power.

        Falls back to (boiler_electrical_power_w, 0.0) - flat, independent
        of T - when no calibrated model or outdoor-temperature forecast
        exists for this step.
        """

        if self.cop_model is None or T_outdoor is None:
            return self.config.boiler_electrical_power_w, 0.0

        margin = self._supply_margin_c(overall_target_max)

        power_low, power_high = map(
            float,
            HeatPumpCOPIdentifier.planned_power_at_reference_points(
                self.cop_model, T_outdoor
            ),
        )

        fit_range_c = (
            HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C
            - HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C
        )
        beta = (power_high - power_low) / fit_range_c

        # power_low is valid at real T_supply=POWER_FIT_T_LOW_C, i.e. at
        # T[k] = POWER_FIT_T_LOW_C - margin (since T_supply = T[k] +
        # margin) - alpha must be anchored there, not at
        # POWER_FIT_T_LOW_C itself, for the line to be correct in T[k]
        # terms (beta is unaffected: margin is a constant shift common to
        # both endpoints, so it cancels out of their difference).
        t_k_low = HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C - margin
        alpha = power_low - beta * t_k_low

        return alpha, beta

    def _supply_margin_c(self, overall_target_max: float) -> float:
        """How much hotter the heat pump's supply runs than the tank it charges
        (see _power_line_coefficients); 0 without a calibrated COP model."""

        if self.cop_model is None:
            return 0.0

        return max(
            self.cop_model.reference_supply_temperature_c - overall_target_max, 0.0
        )

    def _build_objective(
        self,
        model: pyo.ConcreteModel,
        plan: _StepPlan,
        scenario_weights: list[float],
    ):
        objective = 0.0

        for k in model.K:
            expected_grid_power_w = sum(
                weight * model.active_power_w[k, s]
                for s, weight in enumerate(scenario_weights)
            )
            grid_energy_kwh = expected_grid_power_w * plan.dt_hours[k] / 1000.0

            objective += self.config.price_eur_per_kwh * grid_energy_kwh

            objective += self.config.weight_switching * model.boiler_start[k]

            objective += self.config.weight_temperature_slack * model.slack[k]

        return objective

    def _extract_result(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
        termination_condition: TerminationCondition,
    ) -> MPCResult:
        horizon = len(data.solar_forecast_w)
        plan: _StepPlan = model.mpc_step_plan

        objective_value = float(pyo.value(model.objective))

        booster_heat_w = self.thermal_model.booster_heat_w or 0.0
        booster_on_model = [
            round(float(pyo.value(model.booster_on[k]))) for k in range(plan.num_steps)
        ]
        heat_w_model = [
            float(pyo.value(model.q_heat_pump_w[k]))
            + booster_heat_w * booster_on_model[k]
            for k in range(plan.num_steps)
        ]
        # "On" whenever the boiler is heated, by either source.
        schedule = tuple(
            max(round(float(pyo.value(model.boiler_on[m]))), booster_on_model[m])
            for m in plan.fine_to_model
        )

        # Replayed at the input's own fine resolution using the
        # *un-aggregated* ambient/tap data - only the on/off decision is
        # coarse for the far, look-ahead-only portion of the horizon (see
        # _build_step_plan); the physics used to report the resulting
        # trajectory stays exactly as fine-grained as the input.
        a, b = lumped_state_space(
            self.thermal_model.volume_l,
            self.thermal_model.ua_top_w_per_k + self.thermal_model.ua_bottom_w_per_k,
        )
        a_d, b_d = discretize_zoh(a, b, self.config.step_hours * 3600.0)
        tap_forecast_w = data.tap_forecast_w or (0.0,) * horizon

        initial_temperature = (data.current_temp_top + data.current_temp_bottom) / 2.0
        temperatures = [float(initial_temperature)]

        for i in range(horizon - 1):
            t = temperatures[-1]

            next_t = (
                a_d[0, 0] * t
                + b_d[0, 0] * float(data.ambient_temperature)
                + b_d[0, 1] * heat_w_model[plan.fine_to_model[i]]
                + b_d[0, 2] * float(tap_forecast_w[i])
            )
            temperatures.append(float(next_t))

        temperatures = tuple(temperatures)

        # Same formula the objective itself is built from (see
        # _power_line_coefficients), evaluated at each fine step's own
        # (un-aggregated) outdoor forecast and the replayed T[i] above - the
        # reported curve reflects exactly what the optimizer costed with,
        # not a separate display-only estimate.
        overall_target_max = max(data.target_temperature_top)
        electrical_power_w = []

        for i in range(horizon):
            if booster_on_model[plan.fine_to_model[i]]:
                # Resistive element: electrical power equals its heat (COP 1).
                electrical_power_w.append(booster_heat_w)
                continue

            T_outdoor = (
                data.outdoor_temperature_forecast[i]
                if data.outdoor_temperature_forecast
                else None
            )
            alpha, beta = self._power_line_coefficients(T_outdoor, overall_target_max)
            # For the part of the step the heat pump runs (see active_power_w).
            used = (
                heat_w_model[plan.fine_to_model[i]] / self.thermal_model.q_in_nominal_w
            )
            electrical_power_w.append((alpha + beta * temperatures[i]) * used)

        return MPCResult(
            schedule=schedule,
            temperatures=temperatures,
            electrical_power_w=tuple(electrical_power_w),
            heat_w=tuple(heat_w_model[m] for m in plan.fine_to_model),
            objective_value=objective_value,
            solver_status=str(termination_condition),
            termination_condition=str(termination_condition),
        )
