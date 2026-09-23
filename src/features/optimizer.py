import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from domain.dynamics import discretize_zoh
from domain.models import (
    BoilerThermalModel,
    BuildingLumpedModel,
    BuildingThermalModel,
    HeatPumpCOPModel,
)
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.physics import (
    lumped_tank_state_space,
    zone_observation,
    zone_state_space,
)

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
        building_model: BuildingThermalModel | BuildingLumpedModel | None = None,
    ) -> None:
        self.thermal_model = thermal_model
        self.config = config
        # None until a building model has been calibrated AND trusted. Without
        # it no space heating is planned at all and this is the domestic-hot-
        # water optimizer it has always been: the zone's dynamics are the one
        # thing that cannot be guessed at, and planning floor runs against a
        # model that cannot predict their effect would be worse than not
        # planning them.
        self.building_model = building_model
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

        if data.compressor_elapsed_hours < 0:
            raise ValueError("compressor_elapsed_hours cannot be negative.")

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
        tap_w = self._aggregate(data.tap_forecast_w or (0.0,) * horizon, plan, mean)
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
        a, b = lumped_tank_state_space(
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
        # boiler's maximum, where the tank's thermostat cuts the booster out, or
        # the heat pump's own limit without it - nor colder at the start than it
        # already is.
        if booster_possible:
            reachable_c = max_tank_c
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
        max_heat_w = max(self._heat_w, booster_heat_w or 0.0)
        t_floor = unheated(initial_temperature)
        t_ceiling = [initial_temperature]

        for k in range(num_steps - 1):
            a_d, b_d = discretized(plan.dt_hours[k])
            passive = b_d[0, 0] * ambient_c + b_d[0, 2] * float(tap_w[k])
            heated = a_d[0, 0] * t_ceiling[-1] + passive + b_d[0, 1] * max_heat_w
            t_ceiling.append(min(t_upper, float(heated)))

        model.T = pyo.Var(model.K, bounds=lambda m, k: (t_floor[k], t_ceiling[k]))

        model.boiler_on = pyo.Var(model.K, domain=pyo.Binary)

        model.compressor_start = pyo.Var(model.K, domain=pyo.Binary)

        # Heat pump heat (W): continuous up to q_in_nominal_w while on. Below
        # that, a step is only partly used: the heat pump stops by itself once
        # the tank reaches its setpoint (published per run - see
        # app.optimization), or modulates down near its tank limit (real data:
        # ~7 kW at 36-47 degC, ~3 kW at 55 degC).
        model.q_heat_pump_w = pyo.Var(model.K, bounds=(0.0, self._heat_w))

        # The booster heater is a resistive element: it cannot modulate, so while
        # it runs it heats at its identified rating. Its heat per step is still
        # continuous, for the same reason as the heat pump's: the tank's
        # thermostat cuts it out at the boiler's maximum partway through a step,
        # and it takes over from the heat pump partway through one (see
        # heat_source_constraints). As a whole-step decision it could do
        # neither: a 1-hour look-ahead block overshot the maximum by 7 K, and a
        # heat pump reaching its limit early in a step left the rest of that
        # step empty before the booster started - a gap the real run never has.
        model.booster_on = pyo.Var(model.K, domain=pyo.Binary)
        model.q_booster_w = pyo.Var(model.K, bounds=(0.0, booster_heat_w or 0.0))

        # Whether anything heats the tank in a step, 0 or 1 even in a step both
        # sources share: the bounds below leave no other value once boiler_on
        # and booster_on are decided, so it needs no integrality of its own.
        model.tank_heating = pyo.Var(model.K, bounds=(0.0, 1.0))

        model.slack = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        # Always declared, fixed to zero when no space heating is planned, so
        # compressor() below can name it unconditionally. Fixed rather than
        # merely bounded: a variable the solver presolves away comes back
        # without a value at all.
        model.space_on = pyo.Var(model.K, domain=pyo.Binary)

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

        # T[k] is the temperature at the START of step k, so the rule above only
        # checks a step's warmest moment while the tank is coasting. Over a
        # quarter hour that is worth about 0.03 K and does not matter; over a
        # one-hour look-ahead block (see _build_step_plan) it is around 0.25 K,
        # and the plan then satisfies its target at 18:00 while the tank really
        # sags below it by 18:45 - seen exactly that way on this installation.
        #
        # Requiring the same target at the step's end closes it. With constant
        # inputs a first-order tank moves monotonically within a step, so its
        # extremes are the two endpoints: constraining both bounds the whole
        # step, whether it is heating or coasting.
        def end_temperature_rule(m: pyo.ConcreteModel, k: int):
            # The final step has no T[k + 1]: the state vector describes the
            # starts of num_steps intervals, so the very end of the horizon is
            # not represented. Left that way on purpose - it is the far edge of
            # a look-ahead that is re-solved many times before it is ever acted
            # on, and a terminal state would add a variable and a constraint to
            # bound a moment no decision depends on.
            if k + 1 > max(m.K):
                return pyo.Constraint.Skip

            return m.T[k + 1] + m.slack[k] >= min(float(target_c[k]), t_ceiling[k])

        model.end_temperature_constraint = pyo.Constraint(
            model.K,
            rule=end_temperature_rule,
        )

        initial_boiler_on = int(data.boiler_on_current)

        # Inequality, not equality: compressor_start must be 1 on a real 0->1
        # transition (RHS=1, forcing compressor_start[k]>=1), but on a 1->0 stop the
        # RHS is -1 and compressor_start[k]=0 already satisfies ">=-1" trivially. An
        # equality here would force compressor_start=-1 on every stop, which is
        # infeasible against its own binary domain - making any schedule that
        # ever turns the boiler back off unsolvable, and forcing it to stay on
        # forever once started (confirmed: this was the actual cause of an
        # apparently-wasteful "never stops heating" result before this fix).
        # weight_switching in the objective still drives it to 0 except at real
        # starts, since setting it higher only adds cost.
        # Two different questions, deliberately not one helper.
        #
        # heating() answers "was anything putting heat into the tank", which is
        # what the booster's continuation rule needs: it may only take over from
        # a run already under way, and a booster step must be able to follow
        # another booster step.
        #
        # compressor() answers "was the compressor running", which is what a
        # start costs. It covers BOTH demands: the heat pump serves the tank or
        # the zone from the same compressor, never at once, so switching the
        # three-way valve mid-run is not a second start. That is what lets one
        # run cover both - nothing rewards chaining them, it simply stops
        # costing extra. The booster is resistive, so it runs with the compressor
        # off; a handover from compressor to booster is therefore the end of a
        # compressor run, and anything that needs the compressor afterwards is a
        # genuine second start. Counting the booster as compressor heating hid
        # exactly that. The two were the same expression before, which also made
        # a booster-only run look like a compressor start - it cannot occur at
        # all (see the continuation rule in heat_source_constraints), so nothing
        # is lost by no longer pricing it.
        def heating(m: pyo.ConcreteModel, k: int):
            return m.tank_heating[k]

        model.tank_heating_constraint = pyo.ConstraintList()

        for k in range(num_steps):
            model.tank_heating_constraint.add(
                model.tank_heating[k] >= model.boiler_on[k]
            )
            model.tank_heating_constraint.add(
                model.tank_heating[k] >= model.booster_on[k]
            )
            model.tank_heating_constraint.add(
                model.tank_heating[k] <= model.boiler_on[k] + model.booster_on[k]
            )

        def compressor(m: pyo.ConcreteModel, k: int):
            return m.boiler_on[k] + m.space_on[k]

        def startup_rule(m: pyo.ConcreteModel, k: int):
            if k == 0:
                return m.compressor_start[k] >= (compressor(m, k) - initial_boiler_on)

            return m.compressor_start[k] >= (compressor(m, k) - compressor(m, k - 1))

        model.startup_constraint = pyo.Constraint(
            model.K,
            rule=startup_rule,
        )

        # And no more than that: a start is a start. The rule above only forces
        # one on a real 0->1 transition, which was enough while a start merely
        # cost weight_switching, but the ramp below hangs its lower heat on this
        # very variable - a plan could otherwise claim a start it does not make
        # to excuse itself from a full step of heat.
        model.start_exact = pyo.ConstraintList()

        for k in range(num_steps):
            previous = initial_boiler_on if k == 0 else compressor(model, k - 1)
            model.start_exact.add(model.compressor_start[k] <= compressor(model, k))
            model.start_exact.add(model.compressor_start[k] <= 1 - previous)

        # Only enforced with a `start` in the fine-resolution region: a
        # single coarse step already spans far more real time than any
        # sensible minimum runtime (see MPCConfig.coarse_step_hours), so it
        # is trivially satisfied there without an explicit constraint.
        self._add_space_heating(model, data, plan, outdoor_c)

        model.minimum_runtime = pyo.ConstraintList()

        min_runtime = self.config.boiler_min_runtime_steps
        fine_steps = sum(1 for dt in plan.dt_hours if dt <= self.config.step_hours)

        for start in range(fine_steps):
            for offset in range(min_runtime):
                k = start + offset

                if k >= num_steps:
                    continue

                # A compressor start buys compressor time, and space heating
                # is compressor time: the heat pump serves the zone or the tank
                # from the same machine. Leaving space_on out meant a plan that
                # started on the zone could only satisfy this by heating the
                # tank - two steps of DHW with no target at all, seen in a
                # zone-only scenario.
                #
                # The booster stays in, but for a physical reason rather than
                # the old one. It is resistive and runs with the compressor off,
                # so its time does not protect the compressor from cycling. It
                # is admitted here only because heat_source_constraints already
                # gates it on the tank having reached the heat pump's own limit:
                # where it can run at all, the compressor has nothing left to
                # give and the run is over on the machine's terms, not the
                # planner's. Dropping it made a tank one degree under that limit
                # refuse to start and pay the slack instead.
                model.minimum_runtime.add(
                    compressor(model, k) + model.booster_on[k]
                    >= model.compressor_start[start]
                )

        # A run already heating keeps heating until its minimum runtime has
        # passed. Otherwise the next replan, minutes after the start, can plan it
        # off at step 0 while the heat pump finishes the run anyway - the plan,
        # and anything acting on it, would flip for nothing. Skipped where the
        # model cannot represent that run: a tank above the heat pump's limit
        # with no booster to plan with.
        remaining_steps = math.ceil(
            (min_runtime * self.config.step_hours - data.compressor_elapsed_hours)
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
                + b_d[0, 1] * model.q_booster_w[k]
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

        def add_product(running_c, decision, k: int) -> None:
            for bound in (
                running_c <= t_ceiling[k] * decision,
                running_c >= t_floor[k] * decision,
                running_c <= model.T[k] - t_floor[k] * (1 - decision),
                running_c >= model.T[k] - t_ceiling[k] * (1 - decision),
            ):
                model.heat_source_constraints.add(bound)

        def busy(k: int):
            """The part of step k the tank's sources run. A heat pump starting
            in it runs longer than its heat alone says: the ramp's shortfall
            (see _ramp_fraction) is time, not heat. Only used where the booster
            may run."""

            return (
                model.q_heat_pump_w[k] / self._heat_w
                + (1.0 - self._ramp_fraction(plan.dt_hours[k]))
                * model.compressor_start[k]
                + model.q_booster_w[k] / booster_heat_w
            )

        for k in range(num_steps):
            on = model.boiler_on[k]

            # What this step can deliver: full output, less the part of it the
            # compressor spends coming up to speed if it starts here (see
            # _ramp_fraction).
            available = self._heat_w * (
                on
                - (1.0 - self._ramp_fraction(plan.dt_hours[k]))
                * model.compressor_start[k]
            )

            model.heat_source_constraints.add(model.q_heat_pump_w[k] <= available)
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
                    >= available - self._heat_w * (1 - model.boiler_on[k + 1])
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
            # presolve away comes back without a value at all. Also where the
            # ceiling at the step's end shows the tank cannot pass the heat
            # pump's limit by then - the booster could not run there in any plan.
            last = k + 1 == num_steps
            if (
                not booster_possible
                or heat_pump_max_c is None
                or t_ceiling[k if last else k + 1] < heat_pump_max_c
            ):
                model.booster_on[k].fix(0)
                model.q_booster_w[k].fix(0.0)
                continue

            booster = model.booster_on[k]
            model.heat_source_constraints.add(
                model.q_booster_w[k] <= booster_heat_w * booster
            )
            # One tank, one source at a time: a step the two share is a
            # handover, the heat pump for the part up to its limit and the
            # booster for the rest.
            model.heat_source_constraints.add(busy(k) <= 1)
            # The booster only heats above the heat pump's limit: on its own, a
            # step starts there; sharing one, the heat pump's part reaches it.
            # Each slack is exactly the room between this step's own bound and
            # that edge, so a rule says nothing unless the case it covers holds.
            model.heat_source_constraints.add(
                model.T[k]
                >= heat_pump_max_c
                - max(heat_pump_max_c - t_floor[k], 0.0) * (1 - booster + on)
            )

            # And only as the continuation of a run already heating: the booster
            # takes over from a compressor that cannot lift the tank any further,
            # it never starts a DHW run by itself.
            previous = (
                float(data.boiler_on_current) if k == 0 else heating(model, k - 1)
            )
            model.heat_source_constraints.add(booster <= previous + on)

            # Both sources heat until their setpoint, so a step the booster runs
            # in on its own follows one with no idle time left - the run is one
            # piece, as it really is.
            if k > 0:
                model.heat_source_constraints.add(busy(k - 1) >= booster - on)

            if last:
                continue

            a_d, b_d = discretized(plan.dt_hours[k])
            passive = b_d[0, 0] * ambient_c + b_d[0, 2] * float(tap_w[k])
            model.heat_source_constraints.add(
                a_d[0, 0] * model.heat_pump_temperature[k]
                + passive * on
                + b_d[0, 1] * model.q_heat_pump_w[k]
                >= heat_pump_limit_c * on
                - max(heat_pump_limit_c - a_d[0, 0] * t_floor[k] - passive, 0.0)
                * (1 - booster)
            )
            # The tank's thermostat cuts the booster out at the boiler's maximum.
            model.heat_source_constraints.add(
                model.T[k + 1]
                <= max_tank_c + max(t_ceiling[k + 1] - max_tank_c, 0.0) * (1 - booster)
            )
            # Once it has taken over, the compressor run is over.
            model.heat_source_constraints.add(model.boiler_on[k + 1] + booster <= 1)

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
            unused = model.boiler_on[k] - model.q_heat_pump_w[k] / self._heat_w
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
            electrical_w = heat_pump_power_w + model.q_booster_w[k]
            running = model.q_heat_pump_w[k] / self._heat_w
            if booster_heat_w:
                running = running + model.q_booster_w[k] / booster_heat_w

            # Space heating draws through the same compressor. Its efficiency
            # is a per-step constant rather than a line in the zone's
            # temperature: the supply temperature a floor circuit runs at is
            # set by the weather compensation curve, not by the room, so there
            # is no tank-like relationship to follow here - and fitting one
            # would need the heating COP model a heating season has to produce.
            if hasattr(model, "q_space_w"):
                electrical_w = electrical_w + model.q_space_w[k] / model.space_cop[k]
                running = running + model.q_space_w[k] / self._heat_w

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

    @property
    def _heat_w(self) -> float:
        """The heat the compressor puts into the tank once it is up to speed
        (W). The calorimetric measurement where runs have shown it, the ODE's
        own fitted constant otherwise - that one is an average over whole runs,
        including their ramp, so it understates a running compressor.
        """

        return self.thermal_model.q_in_steady_w or self.thermal_model.q_in_nominal_w

    def _ramp_fraction(self, dt_hours: float) -> float:
        """Share of _heat_w a step delivers when the compressor starts in it.

        A compressor rising linearly to full output over its ramp delivers half
        of it while still rising, so a step shorter than the ramp carries
        dt / 2T of it and a longer one all but T / 2T of its own length. 1
        without an identified ramp, which is the constant-output model this
        replaced.
        """

        ramp_seconds = self.thermal_model.q_in_ramp_seconds

        if not ramp_seconds:
            return 1.0

        dt_seconds = dt_hours * 3600.0

        if dt_seconds <= ramp_seconds:
            return dt_seconds / (2.0 * ramp_seconds)

        return 1.0 - ramp_seconds / (2.0 * dt_seconds)

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
        (HeatPumpCOPModel.POWER_FIT_T_LOW_C to ...HIGH_C) - so a single
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
            float, self.cop_model.planned_power_at_reference_points(T_outdoor)
        )

        fit_range_c = (
            HeatPumpCOPModel.POWER_FIT_T_HIGH_C - HeatPumpCOPModel.POWER_FIT_T_LOW_C
        )
        beta = (power_high - power_low) / fit_range_c

        # power_low is valid at real T_supply=POWER_FIT_T_LOW_C, i.e. at
        # T[k] = POWER_FIT_T_LOW_C - margin (since T_supply = T[k] +
        # margin) - alpha must be anchored there, not at
        # POWER_FIT_T_LOW_C itself, for the line to be correct in T[k]
        # terms (beta is unaffected: margin is a constant shift common to
        # both endpoints, so it cancels out of their difference).
        t_k_low = HeatPumpCOPModel.POWER_FIT_T_LOW_C - margin
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

    # Typical floor-heating supply temperature. Space heating runs the
    # compressor far below a DHW cycle, which is most of why its efficiency
    # differs - see HeatPumpCOPIdentifier, where each mode gets its own fit.
    # Used only to evaluate a calibrated COP at a representative operating
    # point; a temperature-dependent line like the boiler's would need the
    # heating COP model that only a heating season can produce.
    SPACE_HEATING_SUPPLY_C = 35.0

    def _space_cop(self, outdoor_c: float | None) -> float:
        """Coefficient of performance for space heating at one step."""

        if self.cop_model is None or outdoor_c is None:
            # Same fallback the tank uses: a flat electrical assumption over
            # its nominal heat output.
            return self._heat_w / max(self.config.boiler_electrical_power_w, 1.0)

        return float(self.cop_model.clamped_cop(outdoor_c, self.SPACE_HEATING_SUPPLY_C))

    def _add_space_heating(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
        plan: "_StepPlan",
        outdoor_c: Sequence[float] | None,
    ) -> None:
        """Adds the zone as a second demand on the same compressor.

        Without a building model, or without a zone temperature to start from,
        space_on stays fixed at zero and the model is exactly the
        domestic-hot-water one it was before. That is deliberate: the zone's
        dynamics are the one thing here that cannot be guessed at, and planning
        floor runs against a model that cannot predict their effect would be
        worse than not planning them at all (see features/building.py for the
        metrics that decide when it can).
        """

        num_steps = plan.num_steps
        building = self.building_model

        # The unmeasured mass temperature is as much a required initial
        # condition as the measured air one: a two-node plan started with a
        # cold screed is a different plan from one started with a charged one.
        # Missing it means the filter has not run, which is not something to
        # paper over with a guess - the zone is simply not planned, exactly as
        # when no model has been calibrated.
        needs_mass = isinstance(building, BuildingThermalModel)
        planned = (
            building is not None
            and data.zone_temperature is not None
            and bool(data.zone_target_temperature)
            and (not needs_mass or data.zone_mass_temperature is not None)
        )

        if not planned:
            for k in range(num_steps):
                model.space_on[k].fix(0)

            return

        def compressor_of(m: pyo.ConcreteModel, k: int):
            return m.boiler_on[k] + m.space_on[k]

        target_c = self._aggregate(data.zone_target_temperature, plan, max)

        def gains(series: Sequence[float]) -> list[float]:
            # mean, not max: a gain is a rate, so a coarse block carries the
            # average of the fine steps it covers, not their peak.
            if not series:
                return [0.0] * num_steps

            return self._aggregate(
                series, plan, lambda values: sum(values) / len(values)
            )

        internal_gain_w = gains(data.zone_internal_gain_w)
        solar_gain_w = gains(data.zone_solar_gain_w)

        # The same compressor serves both, so its output is the tank's nominal
        # heat. A space-heating-specific figure would come from the heating COP
        # calibration, which needs a heating season first.
        max_heat_w = self._heat_w

        a, b = zone_state_space(building)
        observation = zone_observation(building)
        num_states = a.shape[0]

        # State 0 is the air temperature the thermostats measure and comfort is
        # judged on, in both structures; any further state is the thermal mass,
        # which only the two-node one has. Carrying the zone as a vector is
        # what lets the MPC plan against whichever structure the data supports
        # without a second copy of this block.
        model.ZONE_STATES = pyo.RangeSet(0, num_states - 1)
        model.zone_state = pyo.Var(model.K, model.ZONE_STATES)
        model.q_space_w = pyo.Var(model.K, bounds=(0.0, max_heat_w))
        model.zone_slack = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        model.zone_constraints = pyo.ConstraintList()
        model.zone_constraints.add(
            model.zone_state[0, 0] == float(data.zone_temperature)
        )

        if needs_mass:
            model.zone_constraints.add(
                model.zone_state[0, 1] == float(data.zone_mass_temperature)
            )

        # Evaluated at a representative floor-circuit supply temperature and
        # the outdoor temperature of each step, so a cold day costs what a cold
        # day costs. Falls back to the flat assumption while no COP model has
        # been calibrated, exactly as the tank does.
        model.space_cop = [
            self._space_cop(float(outdoor_c[k]) if outdoor_c else None)
            for k in range(num_steps)
        ]

        discretized: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        # At most one tank block per compressor run. The three-way valve may
        # hand over to the zone and back within a run, but once the tank stops
        # being heated it is because it reached its target, so it will not need
        # heating again before the compressor stops. Without this the plan
        # flapped the valve - five steps of tank, six of zone, then a single
        # step of tank - which delivers almost nothing: a DHW start spends an
        # estimated 0.4-0.5 kWh reheating the loop and coil before the tank
        # gains at all (see MPCConfig.heat_pump_min_off_steps).
        #
        # dhw_done[k] latches once the tank stops and is only released when the
        # compressor does.
        model.dhw_done = pyo.Var(model.K, domain=pyo.Binary)
        model.dhw_done[0].fix(0)

        for k in range(1, num_steps):
            # Set when the tank stops being heated.
            model.zone_constraints.add(
                model.dhw_done[k] >= model.boiler_on[k - 1] - model.boiler_on[k]
            )
            # Carried while the compressor keeps running.
            model.zone_constraints.add(
                model.dhw_done[k] >= model.dhw_done[k - 1] + compressor_of(model, k) - 1
            )
            # And it cannot be set for no reason, which would otherwise be a
            # free way to rule the tank out of a run entirely.
            model.zone_constraints.add(
                model.dhw_done[k] <= model.dhw_done[k - 1] + model.boiler_on[k - 1]
            )
            # Having had its turn, the tank does not get a second one.
            model.zone_constraints.add(
                model.boiler_on[k]
                <= 1 - model.dhw_done[k - 1] + (1 - compressor_of(model, k - 1))
            )

        for k in range(num_steps):
            # Heat only flows to the zone while the valve points at it.
            model.zone_constraints.add(
                model.q_space_w[k] <= max_heat_w * model.space_on[k]
            )

            # One compressor, one demand at a time. Sequential within a run is
            # what the hardware does; simultaneous is not.
            model.zone_constraints.add(model.boiler_on[k] + model.space_on[k] <= 1)

            if k + 1 >= num_steps:
                continue

            dt_hours = plan.dt_hours[k]

            if dt_hours not in discretized:
                discretized[dt_hours] = discretize_zoh(a, b, dt_hours * 3600.0)

            a_d, b_d = discretized[dt_hours]
            outdoor = (
                float(outdoor_c[k]) if outdoor_c else float(data.ambient_temperature)
            )

            # u = [T_outdoor, Q_internal, Q_solar, Q_floor], the same input
            # vector both structures are identified with. Each state advances
            # by its own row of the discrete matrices, so where a gain lands -
            # the air node or the mass node - is the model's to decide, not
            # this block's.
            for i in model.ZONE_STATES:
                model.zone_constraints.add(
                    model.zone_state[k + 1, i]
                    == sum(
                        a_d[i, j] * model.zone_state[k, j] for j in model.ZONE_STATES
                    )
                    + b_d[i, 0] * outdoor
                    + b_d[i, 1] * float(internal_gain_w[k])
                    + b_d[i, 2] * float(solar_gain_w[k])
                    + b_d[i, 3] * model.q_space_w[k]
                )

        # What a thermostat reads of the plan: the air alone for one node, an
        # operative temperature for two (see zone_observation). Comfort is
        # judged on it because that is the quantity the setpoint is set in and
        # the occupant feels, and because it is what the model was identified
        # against.
        def measured(k):
            return sum(
                float(observation[i]) * model.zone_state[k, i]
                for i in model.ZONE_STATES
            )

        model.zone_measured = measured

        # Comfort at both ends of a step, for the same reason the tank's target
        # is checked at both: T[k] is the value at a step's START, and a coarse
        # look-ahead block is an hour long.
        for k in range(num_steps):
            model.zone_constraints.add(
                measured(k) + model.zone_slack[k] >= float(target_c[k])
            )

            if k + 1 < num_steps:
                model.zone_constraints.add(
                    measured(k + 1) + model.zone_slack[k] >= float(target_c[k])
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

            objective += self.config.weight_switching * model.compressor_start[k]

            objective += self.config.weight_temperature_slack * model.slack[k]

            # Comfort in the zone is weighed the same as the tank's target:
            # both are promises the plan made, and neither is worth trading for
            # a few cents of electricity.
            if hasattr(model, "zone_slack"):
                objective += self.config.weight_temperature_slack * model.zone_slack[k]

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
        # "On" whenever the boiler is actually heated, which for a coarse block
        # is the part of it the run uses rather than the whole block (see
        # fine_heat_w below). Reported any other way, the run would end where
        # the block does and the setpoint published for it would be the tank's
        # temperature at a moment the heat pump has long since stopped.
        planned_on = [
            round(float(pyo.value(model.tank_heating[m]))) for m in plan.fine_to_model
        ]

        # Replayed at the input's own fine resolution using the
        # *un-aggregated* ambient/tap data - only the on/off decision is
        # coarse for the far, look-ahead-only portion of the horizon (see
        # _build_step_plan); the physics used to report the resulting
        # trajectory stays exactly as fine-grained as the input.
        a, b = lumped_tank_state_space(
            self.thermal_model.volume_l,
            self.thermal_model.ua_top_w_per_k + self.thermal_model.ua_bottom_w_per_k,
        )
        a_d, b_d = discretize_zoh(a, b, self.config.step_hours * 3600.0)
        tap_forecast_w = data.tap_forecast_w or (0.0,) * horizon

        initial_temperature = (data.current_temp_top + data.current_temp_bottom) / 2.0
        temperatures = [float(initial_temperature)]

        # A coarse block the plan only partly uses is a run that stops inside
        # it, not an hour at half power: each source runs at its own output
        # until its setpoint and then stops (the heat pump's 6.6 kW or nothing),
        # the heat pump first and the booster after it. Spreading the block's
        # energy evenly instead reported a heat and an electrical draw the
        # machine cannot produce, and gave the tank a slow drift where it really
        # has a rise and then a coast. The energy over the block is the same
        # either way - only where it lands inside it changes. Laid from the
        # block's start, a run stays one piece: a block a run continues after is
        # full (see heat_source_constraints). A single fine step keeps its
        # average: there the part-used step IS the run's last step, which is how
        # it was identified.
        # The booster's thermostat also acts inside the block: a tap later in
        # it is met by the booster switching on again, not by heat that was
        # already put in before the tap and overshot the maximum.
        identified_max_c = self.thermal_model.max_tank_temperature_c
        max_tank_c = (
            identified_max_c
            if identified_max_c is not None
            else DEFAULT_MAX_TANK_TEMPERATURE_C
        )
        heat_pump_w = [0.0] * horizon
        booster_w = [0.0] * horizon

        for m in range(plan.num_steps):
            slots = [i for i in range(horizon) if plan.fine_to_model[i] == m]
            heat_pump_wh = float(pyo.value(model.q_heat_pump_w[m])) * plan.dt_hours[m]
            booster_wh = float(pyo.value(model.q_booster_w[m])) * plan.dt_hours[m]
            starts = round(float(pyo.value(model.compressor_start[m])))

            for n, i in enumerate(slots):
                passive = (
                    a_d[0, 0] * temperatures[i]
                    + b_d[0, 0] * float(data.ambient_temperature)
                    + b_d[0, 2] * float(tap_forecast_w[i])
                )

                if len(slots) == 1:
                    heat_pump_w[i] = heat_pump_wh / plan.dt_hours[m]
                    booster_w[i] = booster_wh / plan.dt_hours[m]
                else:
                    # A block the compressor starts in loses its ramp where the
                    # ramp happens, in the run's first quarter hour - the same
                    # energy the block's own step lost to it (see
                    # _ramp_fraction).
                    rate_w = self._heat_w * (
                        self._ramp_fraction(self.config.step_hours)
                        if starts and n == 0
                        else 1.0
                    )
                    hours = min(self.config.step_hours, heat_pump_wh / rate_w)
                    heat_pump_w[i] = rate_w * hours / self.config.step_hours
                    heat_pump_wh -= rate_w * hours

                    booster_w[i] = (
                        min(
                            booster_heat_w * (self.config.step_hours - hours),
                            booster_wh,
                            max(max_tank_c - passive - b_d[0, 1] * heat_pump_w[i], 0.0)
                            / b_d[0, 1]
                            * self.config.step_hours,
                        )
                        / self.config.step_hours
                    )
                    booster_wh -= booster_w[i] * self.config.step_hours

                if i + 1 < horizon:
                    temperatures.append(
                        float(passive + b_d[0, 1] * (heat_pump_w[i] + booster_w[i]))
                    )

        fine_heat_w = [hp + bo for hp, bo in zip(heat_pump_w, booster_w, strict=True)]

        # Only inside a coarse block: there the plan's own step spans hours and
        # the run stops inside it. A fine step is already the resolution the
        # decision was made at, and its last one may legitimately carry no heat
        # at all (the heat pump reached its setpoint), which is still part of
        # the run the minimum runtime counts.
        schedule = tuple(
            int(
                on
                and (
                    fine_heat_w[i] > 0.0
                    or plan.dt_hours[plan.fine_to_model[i]] <= self.config.step_hours
                )
            )
            for i, on in enumerate(planned_on)
        )

        temperatures = tuple(temperatures)

        # Same formula the objective itself is built from (see
        # _power_line_coefficients), evaluated at each fine step's own
        # (un-aggregated) outdoor forecast and the replayed T[i] above - the
        # reported curve reflects exactly what the optimizer costed with,
        # not a separate display-only estimate.
        overall_target_max = max(data.target_temperature_top)
        electrical_power_w = []

        for i in range(horizon):
            T_outdoor = (
                data.outdoor_temperature_forecast[i]
                if data.outdoor_temperature_forecast
                else None
            )
            alpha, beta = self._power_line_coefficients(T_outdoor, overall_target_max)
            # For the part of the step the heat pump runs (see active_power_w).
            # The booster is resistive: electrical power equals its heat (COP 1).
            used = heat_pump_w[i] / self._heat_w
            electrical_power_w.append(
                (alpha + beta * temperatures[i]) * used + booster_w[i]
            )

        # Empty when no space heating was planned, so a caller can tell "the
        # zone was left alone" from "the zone was planned to coast".
        space_schedule: tuple[int, ...] = ()
        space_heat_w: tuple[float, ...] = ()
        zone_temperatures: tuple[float, ...] = ()

        if hasattr(model, "q_space_w"):
            space_on_model = [
                round(float(pyo.value(model.space_on[k])))
                for k in range(plan.num_steps)
            ]
            space_heat_model = [
                float(pyo.value(model.q_space_w[k])) for k in range(plan.num_steps)
            ]
            # What the thermostats would read, which is what comfort was
            # judged on (see zone_observation).
            zone_model = [
                float(pyo.value(model.zone_measured(k))) for k in range(plan.num_steps)
            ]

            space_schedule = tuple(space_on_model[m] for m in plan.fine_to_model)
            space_heat_w = tuple(space_heat_model[m] for m in plan.fine_to_model)
            zone_temperatures = tuple(zone_model[m] for m in plan.fine_to_model)

        return MPCResult(
            schedule=schedule,
            temperatures=temperatures,
            electrical_power_w=tuple(electrical_power_w),
            heat_w=tuple(fine_heat_w),
            objective_value=objective_value,
            solver_status=str(termination_condition),
            termination_condition=str(termination_condition),
            space_schedule=space_schedule,
            space_heat_w=space_heat_w,
            zone_temperatures=zone_temperatures,
        )
