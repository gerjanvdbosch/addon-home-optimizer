import logging
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import highspy
import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from domain.dynamics import discretize_zoh
from domain.models import (
    BoilerThermalModel,
    BuildingThermalModel,
    FloorCircuitModel,
    HeatPumpCOPModel,
)
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.physics import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    ZONE_SCREED_STATES,
    ZONE_SLAB_STATES,
    lumped_tank_state_space,
    zone_floor,
    zone_observation,
    zone_state_space,
)

logger = logging.getLogger(__name__)

# The tank's maximum water temperature (deg C) until a booster run has shown the
# real one (see BoilerThermalIdentifier._identify_booster) - a typical limit for a
# domestic hot water tank, chosen for this installation.
DEFAULT_MAX_TANK_TEMPERATURE_C = 65.0

# How close to optimal a plan must be proven (EUR). The solver's default
# tolerance is relative to the objective instead, which an unavoidable
# temperature shortfall inflates into the thousands - at 0.01% of that, a plan was
# accepted on real data that kept the heat pump 'on' without heating for 1.5
# hours. A tenth of a cent, not a whole one: heat the sun almost covers costs
# less than a cent, so at one cent a plan with two needless full steps - the
# tank to 57 instead of 45 degC - counted as optimal.
MIP_ABSOLUTE_GAP_EUR = 0.001

# HiGHS scales the objective by 2 to this power inside the solve (and so its
# gap and bound with it). The plan's costs run down to ~1e-6 EUR per W of a
# step, a tenth of a dual tolerance above HiGHS's own (1e-7): its reduced costs
# were that inexact, and fixing variables on them at the root cut off the
# optimal plan. On real data (a cold tank, the house 1 K under its target) one
# of six random seeds proved 1.3987 EUR optimal against a feasible 1.3108 - its
# root bound already 1.3653 - and the cold solve in the app did the same with
# one more tank plan to choose from. Scaled by 2^10, all six found 1.3108, in
# the same time; no setting of cuts, symmetry, restarts or probing did.
OBJECTIVE_SCALE_EXPONENT = 10

# How far apart (K) the tangents lie that bound the square of the tank's excess
# over its unheated trajectory from below (see MPCOptimizer._build_model). They
# miss it by at most a quarter of the spacing squared, 1 K^2 at 2 K: on this
# installation's tank (200 l) and power line (some 0.006 per K of heat) about
# a Wh over a 40-hour horizon, a tenth of a cent at most - under
# MIP_ABSOLUTE_GAP_EUR, the tolerance the plan is proven to anyway.
EXCESS_TANGENT_SPACING_K = 2.0

# The same for the square of the zone's distance from its wanted temperature
# (see MPCConfig.weight_comfort_eur_per_k2h): missed by at most a quarter of
# the spacing squared, 0.01 K^2 at 0.2 K - at 0.5 EUR/K^2h half a cent an
# hour, far finer than that price is known. 0.1 K made the same plans on real
# data (13 Oct, a zone started 1-1.5 K under its target) in three times the
# time: 8 s against 2.6.
COMFORT_TANGENT_SPACING_K = 0.2

# HiGHS's presolve, off. On this model it cut off the optimal plan and proved
# a dearer one optimal, three times on real data with the zone: 2.076 EUR
# against 1.802 on a 46-hour plan, 1.7095 against 1.7085 on another, and a
# cooling day's plan held to 8757 EUR of comfort it did not have to miss
# (1.049 without presolve). Leaving out single rules (parallel rows, doubleton
# equations) mended the one day and not the next; without presolve each of
# them was proven right, in about the time it took with it. Not the same cause
# as OBJECTIVE_SCALE_EXPONENT's: with the objective scaled, presolve still
# proved a plan optimal 0.003 EUR above the optimum (real data, 16 Oct, a cold
# tank: 0.7170 against 0.7141), though up to a third sooner.
PRESOLVE = "off"

# HiGHS's primal heuristics, off once the previous plan has given the solve its
# start: they look for good plans, and that one is all but optimal already, so
# what is left is proving it - on real data they took a quarter of the solve's
# LP iterations and 21 sub-MIPs (heating with the zone 12.8 -> 9.1 s, cooling
# 6.3 -> 1.7 s, and a better plan at the time limit over 48 hours). A solve
# without a start keeps them: cold, they found the plan sooner.
WARM_START_HIGHS_OPTIONS = {
    "mip_heuristic_effort": 0.0,
    "mip_heuristic_run_feasibility_jump": False,
    "mip_heuristic_run_rens": False,
    "mip_heuristic_run_rins": False,
    "mip_heuristic_run_root_reduced_cost": False,
    "mip_heuristic_run_shifting": False,
    "mip_heuristic_run_zi_round": False,
}

# Swanson's rule: the standard three-point weights for a distribution's
# expectation from its P10/P50/P90 (exact for a symmetric distribution, close
# for moderately skewed ones). Grid import is costed as this expectation over
# outcomes of the solar surplus, solar less baseload: import cost is convex in
# it (max(0, P_el - surplus)), so costing at P50 alone systematically
# understates the expected cost of relying on uncertain sun and load.
SOLAR_SCENARIO_WEIGHTS = (0.3, 0.4, 0.3)


# The tank planned on its own: the heat pump and the booster heating one
# lumped tank. The tank is one state, so its plan is a shortest path through
# time over that state, the heat pump's mode and how long it has been in it -
# dynamic programming, exact up to the temperature grid, where the same plan as
# a mixed integer program took up to half a minute (real data, a 60 degC day:
# 24.6 s against 0.2 s, the same plan). TankSetting carries the physics and the
# rules from MPCOptimizer to it.

# The spacing of the temperature grid the plan is solved over (K). Each step's
# grid is the previous one's image under the tank's coasting, with the
# temperatures that coast onto a target exactly added, so a tank left alone
# and a target met are never interpolated: what remains is a full step of
# heat landing between two points, its value drawn straight between theirs.
# 0.3 K found the program's plan for the tank on its own too, a fifth sooner,
# but beside the zone a tank plan that made the whole 0.16% dearer; 0.5 K
# missed it on its own already (a 60 degC day 2% dearer).
GRID_K = 0.2

INFEASIBLE = 1e12


@dataclass
class TankSetting:
    """The tank over the plan's steps as physics and the rules fix it before
    anything is decided (see MPCOptimizer._tank_setting)."""

    dt_hours: list[float]
    # The tank's own continuous dynamics (see physics.lumped_tank_state_space).
    a_continuous: np.ndarray
    b_continuous: np.ndarray
    ambient_c: float
    tap_w: tuple[float, ...]
    # The heat a running heat pump puts in per step (W).
    heat_w: list[float]
    initial_c: float
    # The tank left alone, and heated flat out to what can heat it, per step.
    floor_c: list[float]
    ceiling_c: list[float]
    heat_pump_max_c: float | None
    max_tank_c: float
    heat_pump_limit_c: float
    booster_heat_w: float | None
    # Whether the booster can be of use at all, and in which steps.
    booster_possible: bool
    booster_allowed: list[bool]
    # The target per step (deg C), and what a run ending before it adds (K).
    required_c: list[float]
    overshoot_c: list[float]
    # Electrical power alpha + beta * T for the share of a step the heat pump
    # runs (see MPCOptimizer._power_line_coefficients).
    power_lines: list[tuple[float, float]]
    # The share of a step's heat a run starting in it delivers.
    ramp: list[float]
    start_power_w: float
    booster_draw_per_heat: float
    import_price: tuple[float, ...]
    export_price: tuple[float, ...]
    # (weight, surplus W per step) per solar scenario.
    scenarios: list[tuple[float, np.ndarray]]
    weight_switching: float
    weight_slack: float
    min_runtime_hours: float
    min_off_steps: int
    # Steps from now the heat pump stays off, after a run that just ended.
    idle_steps: int
    # A run under way, and the steps it still has to go on for.
    running: bool
    running_steps: int
    # Whether the compressor runs now for anything else (the zone): a run of
    # the tank cannot start in the first step then.
    compressor_busy: bool
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def num_steps(self) -> int:
        return len(self.dt_hours)

    @property
    def step_start_h(self) -> np.ndarray:
        return np.cumsum([0.0, *self.dt_hours[:-1]])

    def steps_within(self, hours: float, first: int = 0) -> range:
        """The steps from `first` on that begin within `hours` of it: what a
        duration covers counted in the plan's own steps, the first of which
        may be the rest of a quarter."""

        start_h = self.step_start_h
        end = first

        while end < self.num_steps and start_h[end] - start_h[first] < hours - 1e-9:
            end += 1

        return range(first, end)

    def discretized(self, hours: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact zero-order-hold dynamics over a step of `hours` - the same for
        every step of one length, so computed once per length."""

        if hours not in self._cache:
            self._cache[hours] = discretize_zoh(
                self.a_continuous, self.b_continuous, hours * 3600.0
            )

        return self._cache[hours]

    def step(self, k: int) -> tuple[float, float, float]:
        """Step k's decay, heat gain (K/W) and what ambient and taps do (K):
        T[k + 1] = decay T[k] + passive + gain (heat pump + booster)."""

        if ("step", k) not in self._cache:
            a_d, b_d = self.discretized(self.dt_hours[k])
            self._cache["step", k] = (
                float(a_d[0, 0]),
                float(b_d[0, 1]),
                float(b_d[0, 0] * self.ambient_c + b_d[0, 2] * float(self.tap_w[k])),
            )

        return self._cache["step", k]

    def unheated(self, start_c: float, first: int = 0) -> list[float]:
        """The tank temperature at each step from `first` on with no heating
        from start_c on."""

        temperatures = [start_c]

        for k in range(first, self.num_steps - 1):
            decay, _, passive = self.step(k)
            temperatures.append(decay * temperatures[-1] + passive)

        return temperatures


@dataclass(frozen=True)
class TankPlan:
    """Per step: the heat pump's and the booster's heat (W), the tank at the
    step's start (deg C), whether a run starts in it, and what the step costs
    (EUR) - electricity, start and any shortfall against its target."""

    heat_pump_w: tuple[float, ...]
    booster_w: tuple[float, ...]
    temperatures: tuple[float, ...]
    starts: tuple[int, ...]
    cost_eur: tuple[float, ...]


def draw_w(tank: TankSetting, k: int, tank_c, heat_pump_w, booster_w, start):
    """The electrical power step k draws (W) for this much heat from either
    source at tank_c: the power line for the heat pump's share of the step -
    a run's first quarter at its measured power instead (see
    MPCOptimizer._start_power_correction_w); a hand-over from the zone, the
    compressor at speed, on the power line throughout - and the booster's own
    draw."""

    alpha, beta = tank.power_lines[k]
    heat_pump = (alpha + beta * tank_c) * heat_pump_w / tank.heat_w[k]
    unheated_w = alpha + beta * tank.floor_c[k]

    if tank.start_power_w:
        heat_pump = heat_pump + start * (tank.start_power_w - tank.ramp[k] * unheated_w)

    return heat_pump + tank.booster_draw_per_heat * booster_w


def grid_w(tank: TankSetting, k: int, power_w, heat_pump_w, booster_w):
    """What of that power comes from the grid (W), expected over the solar
    scenarios: the sun only for the part of the step a source runs."""

    running = heat_pump_w / tank.heat_w[k]

    if tank.booster_heat_w:
        running = running + booster_w / tank.booster_heat_w

    return sum(
        weight * np.maximum(power_w - surplus_w[k] * running, 0.0)
        for weight, surplus_w in tank.scenarios
    )


def step_cost(tank: TankSetting, k: int, tank_c, heat_pump_w, booster_w, start):
    """What step k costs (EUR) for this heat: the grid power at the import
    price, the own sun at the export it forgoes, and a start."""

    power_w = draw_w(tank, k, tank_c, heat_pump_w, booster_w, start)
    hours_kwh = tank.dt_hours[k] / 1000.0
    export = tank.export_price[k]

    return (
        (tank.import_price[k] - export)
        * grid_w(tank, k, power_w, heat_pump_w, booster_w)
        * hours_kwh
        + export * np.maximum(power_w, 0.0) * hours_kwh
        + tank.weight_switching * start
    )


def shortfall_cost(tank: TankSetting, k: int, tank_c, next_c, ran):
    """What missing step k's target at its start or end costs (EUR)."""

    required = tank.required_c[k] + tank.overshoot_c[k] * ran
    short = np.maximum(required - tank_c, 0.0)

    if k + 1 < tank.num_steps:
        short = np.maximum(short, required - next_c)

    return tank.weight_slack * short


def _runs(plan: TankPlan) -> tuple[tuple[int, int], ...]:
    """When the plan's heat pump and booster are on, per step."""

    return tuple(
        (int(heat_pump_w > 0.0), int(booster_w > 0.0))
        for heat_pump_w, booster_w in zip(plan.heat_pump_w, plan.booster_w, strict=True)
    )


def _picked(quantity, next_c, best) -> float:
    """quantity of the one tank's action as chosen: of a full step, or at the
    landing point `best` of a run's last."""

    value = np.broadcast_to(np.asarray(quantity, dtype=float), np.shape(next_c))

    return float(value[0] if best is None else value[0, best[0]])


def plan_tank(
    tank: TankSetting,
    zone_on: Sequence[int] = (),
    cooling: bool = False,
    local_day: Sequence[int] = (),
) -> TankPlan:
    """The cheapest plan for the tank, on its own or around the zone's runs
    (zone_on per step, cooling or heating, by local day): the same machine
    serves one demand at a time, and the rules for sharing it are
    _build_model's - so the plan holds there with the zone's runs as given.

    States per step: the tank temperature, the mode - off since how many
    steps, resting after its block until the target it heated for comes,
    the heat pump or the booster on with how many steps of the minimum runtime
    to go - and whether the heat pump has run yet in the plan (a target after
    a run carries its overshoot). The heat pump heats a step
    in full unless it is a run's last, which it uses in part; it starts a run
    with its ramp - one taken over from the zone without - never above its
    limit, and may hand over to the booster
    there - in the step it reaches the limit, the booster taking the rest of
    it - which then heats full steps but its last, up to the boiler's maximum.
    """

    num_steps = tank.num_steps
    min_off = tank.min_off_steps
    min_runs = [
        len(tank.steps_within(tank.min_runtime_hours, k)) for k in range(num_steps)
    ]
    forced_max = max(min_runs + [tank.running_steps, 1])
    booster_w = tank.booster_heat_w or 0.0

    # Where the zone runs, the tank waits: not in the zone's steps. A heating
    # run of the zone and the tank hand the compressor over to each other
    # either way: the run that follows needs no start of its own, though the
    # tank's first step after the zone still brings the loop up to the
    # tank's temperature (see draw_w). Cooling shares no run with the tank at
    # all, so not in the step after or before one either, and a day's cooling
    # comes before its hot water: no tank before a later cooling run that day.
    zone = (list(zone_on) + [0] * num_steps)[:num_steps]
    # Nor while a tank run under way has its minimum runtime to go (see
    # _build_model's running_run): the zone's runs of a plan made before the
    # tank started have no place there (real data, 18 Oct 2026 16:22: a tank
    # run 4 minutes in, the previous plan's zone in its next hour - no plan).
    zone[: tank.running_steps] = [0] * min(tank.running_steps, num_steps)
    later_cooling = [
        cooling
        and bool(local_day)
        and any(
            zone[j] and local_day[j] == local_day[k] for j in range(k + 1, num_steps)
        )
        for k in range(num_steps)
    ]
    booster_free = [not zone[k] and not later_cooling[k] for k in range(num_steps)]
    heat_pump_free = [
        booster_free[k]
        and not (cooling and k > 0 and zone[k - 1])
        and not (cooling and k + 1 < num_steps and zone[k + 1])
        for k in range(num_steps)
    ]
    from_zone = [
        not cooling and (zone[k - 1] if k > 0 else tank.compressor_busy)
        for k in range(num_steps)
    ]
    # Hot water in one block per target: once a block has heated the tank, it
    # rests until the target it heated for comes - a step the tank left alone
    # misses its target in. Not to the end of that target: taps may draw a
    # tank below a long one, and heating again there is a new block's.
    targets = [tank.required_c[k] > tank.floor_c[k] for k in range(num_steps)]
    handover_eur = [
        tank.weight_switching
        if not cooling and k + 1 < num_steps and zone[k + 1] and not zone[k]
        else 0.0
        for k in range(num_steps)
    ]

    # Each step's grid is the previous one's image under coasting, cut below
    # the floor and extended above to what the tank can reach - plus, per
    # target, the temperatures that coast onto it exactly.
    top = max(max(tank.ceiling_c), tank.max_tank_c) + 2 * GRID_K
    grids = [np.arange(tank.initial_c, top, GRID_K)]

    for k in range(num_steps - 1):
        decay, _, passive = tank.step(k)
        spacing = decay * (grids[-1][1] - grids[-1][0])
        grid = decay * grids[-1] + passive
        grid = grid[grid >= tank.floor_c[k + 1] - 1e-6]

        if grid[-1] < top:
            extra = math.ceil((top - grid[-1]) / spacing)
            grid = np.concatenate([grid, grid[-1] + spacing * np.arange(1, extra + 1)])

        grids.append(grid)

    exact: list[list[float]] = [[] for _ in range(num_steps)]

    for j in range(num_steps):
        for required in {tank.required_c[j], tank.required_c[j] + tank.overshoot_c[j]}:
            if required <= tank.floor_c[j]:
                continue

            for end in (j, j + 1):
                temperature = required

                for k in range(min(end, num_steps - 1), -1, -1):
                    if tank.floor_c[k] - 1e-6 <= temperature <= top:
                        exact[k].append(temperature)

                    if k > 0:
                        decay, _, passive = tank.step(k - 1)
                        temperature = (temperature - passive) / decay

    grids = [
        np.unique(np.concatenate([grid, np.array(points)])) if points else grid
        for grid, points in zip(grids, exact, strict=True)
    ]

    modes = (
        [("off", count) for count in range(min_off + 1)]
        + [("rest", count) for count in range(min_off + 1)]
        + [("heat_pump", forced) for forced in range(forced_max + 1)]
        + [("booster", forced) for forced in range(forced_max + 1)]
    )
    # Whether the heat pump has run yet only differs while the tank is off and
    # has been since the plan began: any run sets it. The booster only where
    # it may run at all.
    states = [(("off", min_off), 0)] + [
        (mode, 1)
        for mode in modes
        if mode[0] != "booster" or (booster_w and any(tank.booster_allowed))
    ]

    def menu(k, tank_c, values):
        """What each kind of action does at step k from each tank_c, whatever
        the state it is taken in: its heat, next tank and energy cost, worked
        out once and shared by every state that may take it. Per kind: one
        next tank per tank_c (a full step), or a run's last step landing on
        the next grid's own points, the choice of which is the state's."""

        last = k + 1 == num_steps
        decay, gain, passive = tank.step(k)
        heat_w = tank.heat_w[k]
        grid = None if last else grids[k + 1]
        at_limit = k + 1 < num_steps and tank.ceiling_c[k + 1] > tank.heat_pump_limit_c
        coast = decay * tank_c + passive
        # No further than the tank can be heated at all (the boiler's maximum,
        # or the heat pump's limit without a booster).
        ceiling = tank.ceiling_c[k + 1] + 1e-6 if not last else math.inf
        everywhere = np.ones_like(tank_c, dtype=bool)
        kinds = {}

        def step_to(ok, heat_pump, booster, start, next_c, heats=True):
            """A full step: where it lands between the next grid's points -
            and nothing drawn where nothing heats."""

            kind = {
                "ok": ok & (next_c <= ceiling),
                "heat_pump": heat_pump,
                "booster": booster,
                "start": start,
                "next": next_c,
                "energy": step_cost(tank, k, tank_c, heat_pump, booster, start)
                if heats
                else 0.0,
            }

            if not last:
                j = np.clip(
                    np.searchsorted(grid, next_c, side="right") - 1, 0, len(grid) - 2
                )
                kind["between"] = (
                    j,
                    np.clip((next_c - grid[j]) / (grid[j + 1] - grid[j]), 0.0, 1.0),
                    next_c > grid[-1] + 1e-9,
                )

            return kind

        def land(low, high, ok, heat_pump_of, booster_of, start, saved=0.0):
            """A run's last step, landing on the next grid's own points between
            low and high - less what ending there saves."""

            high = np.minimum(high, ceiling)

            if last:
                points = low[:, None] + (high - low)[:, None] * np.linspace(0, 1, 41)
                valid = np.ones_like(points, dtype=bool)
                index = None
            else:
                first = np.searchsorted(grid, low - 1e-9)
                span = int(np.max(np.searchsorted(grid, high + 1e-9) - first)) + 2
                index = np.clip(first[:, None] + np.arange(span), 0, len(grid) - 1)
                points = grid[index]
                valid = (points >= low[:, None] - 1e-9) & (
                    points <= high[:, None] + 1e-9
                )

            heat_pump, booster = heat_pump_of(points), booster_of(points)
            at = np.broadcast_to(tank_c[:, None], points.shape)

            return {
                "ok": valid & ok[:, None],
                "heat_pump": heat_pump,
                "booster": booster,
                "start": start,
                "next": points,
                "index": index,
                "energy": step_cost(tank, k, at, heat_pump, booster, start) - saved,
            }

        kinds["coast"] = step_to(everywhere, 0.0, 0.0, 0, coast, heats=False)

        for start in (0, 1):
            # A block's first step: the compressor starting, with its ramp, or
            # taking over from the zone at speed (see _build_model).
            ramped = start and not from_zone[k]
            available = heat_w * (tank.ramp[k] if ramped else 1.0)
            full = coast + gain * available
            ok = everywhere if not at_limit else full <= tank.heat_pump_limit_c + 1e-9
            kinds["heat_pump", start] = step_to(ok, available, 0.0, ramped, full)
            high = np.minimum(full, tank.heat_pump_limit_c) if at_limit else full
            kinds["heat_pump_last", start] = land(
                coast,
                np.maximum(high, coast),
                everywhere,
                lambda points, coast=coast: (points - coast[:, None]) / gain,
                lambda points: np.zeros_like(points),
                ramped,
                handover_eur[k],
            )

            if tank.booster_allowed[k] and booster_w and booster_free[k] and not last:
                to_limit = (tank.heat_pump_limit_c - coast) / gain
                reaches = (to_limit >= 0.0) & (to_limit <= available)
                to_limit = np.clip(to_limit, 0.0, available)
                rest = np.maximum(
                    1.0 - to_limit / heat_w - (1.0 - tank.ramp[k]) * ramped, 0.0
                )
                full_booster = tank.heat_pump_limit_c + gain * booster_w * rest
                kinds["handover", start] = step_to(
                    reaches & (full_booster <= tank.max_tank_c + 1e-9),
                    to_limit,
                    booster_w * rest,
                    ramped,
                    full_booster,
                )
                low = np.full_like(tank_c, tank.heat_pump_limit_c)
                kinds["handover_last", start] = land(
                    low,
                    np.maximum(np.minimum(full_booster, tank.max_tank_c), low),
                    reaches,
                    lambda points, to_limit=to_limit: np.broadcast_to(
                        to_limit[:, None], points.shape
                    ),
                    lambda points: (points - tank.heat_pump_limit_c) / gain,
                    ramped,
                )

        if tank.booster_allowed[k] and booster_w and booster_free[k]:
            # Only above the heat pump's limit, up to the boiler's maximum.
            above = tank_c >= tank.heat_pump_max_c - 1e-9
            full = coast + gain * booster_w
            kinds["booster"] = step_to(
                above & (full <= tank.max_tank_c + 1e-9), 0.0, booster_w, 0, full
            )
            kinds["booster_last"] = land(
                coast,
                np.maximum(np.minimum(full, tank.max_tank_c), coast),
                above,
                lambda points: np.zeros_like(points),
                lambda points, coast=coast: (points - coast[:, None]) / gain,
                0,
            )

        return kinds

    def options(k, mode, ran, tank_c, values, kinds):
        """Per action this state may take: its cost (EUR, with what follows)
        at each tank_c, and (kind, next state)."""

        last = k + 1 == num_steps
        kind, count = mode
        taken = []

        def take(name, state):
            if name not in kinds:
                return

            action = kinds[name]
            full_step = np.ndim(action["next"]) == 1
            short = shortfall_cost(
                tank, k, tank_c if full_step else tank_c[:, None], action["next"], ran
            )

            if full_step:
                if last:
                    following = 0.0
                else:
                    j, w, beyond = action["between"]
                    value = values[state]
                    low, high = value[j], value[j + 1]
                    following = np.where(
                        beyond,
                        INFEASIBLE,
                        np.where(
                            low >= INFEASIBLE / 2,
                            high,
                            np.where(
                                high >= INFEASIBLE / 2, low, low + w * (high - low)
                            ),
                        ),
                    )

                cost = np.where(
                    action["ok"], action["energy"] + short + following, INFEASIBLE
                )
                taken.append((cost, (name, state, None)))
            else:
                following = (
                    0.0 if action["index"] is None else values[state][action["index"]]
                )
                cost = np.where(
                    action["ok"], action["energy"] + short + following, INFEASIBLE
                )
                best = np.argmin(cost, axis=1)
                rows = np.arange(len(tank_c))
                taken.append((cost[rows, best], (name, state, best)))

        # A block ends resting, unless its target has come.
        ended = "off" if targets[k] else "rest"

        def heat_pump(start, forced):
            take(("heat_pump", start), (("heat_pump", max(forced - 1, 0)), 1))
            take(("handover", start), (("booster", max(forced - 1, 0)), 1))

            if forced <= 0:
                take(("heat_pump_last", start), ((ended, 0), 1))
                take(("handover_last", start), ((ended, 0), 1))

        if kind == "rest":
            take("coast", ((ended, min(count + 1, min_off)), ran))
        elif kind == "off":
            take("coast", (("off", min(count + 1, min_off)), ran))

            if (
                count >= min_off
                and k >= tank.idle_steps
                and not (k == 0 and tank.compressor_busy and cooling)
                and heat_pump_free[k]
            ):
                heat_pump(1, min_runs[k] - 1)
        elif kind == "heat_pump" and heat_pump_free[k]:
            heat_pump(0, count)
        elif kind == "booster":
            take("booster", (("booster", max(count - 1, 0)), ran))

            if count <= 0:
                take("booster_last", ((ended, 0), ran))

        return taken

    values: list[dict] = [{} for _ in range(num_steps + 1)]

    for k in range(num_steps - 1, -1, -1):
        kinds = menu(k, grids[k], values[k + 1])

        for mode, ran in states:
            taken = options(k, mode, ran, grids[k], values[k + 1], kinds)
            values[k][mode, ran] = (
                np.min(np.vstack([cost for cost, _ in taken]), axis=0)
                if taken
                else np.full(len(grids[k]), INFEASIBLE)
            )

    # Forward from the tank as it is, exactly: in each step the action that
    # is cheapest together with the next step's values.
    if tank.running:
        now = [("heat_pump", max(tank.running_steps - 1, 0))]
        now += [("booster", 0)] if booster_w else []
        now += [("off", min_off)] if tank.running_steps == 0 else []
    else:
        now = [("off", min_off)]

    tank_c = np.array([tank.initial_c])
    ran = int(tank.running)
    heat_pump_w, booster_heat, temperatures, starts, costs = [], [], [], [], []

    for k in range(num_steps):
        kinds = menu(k, tank_c, values[k + 1])
        _, (name, (mode, ran_next), best) = min(
            (
                (float(cost[0]), choice)
                for state_mode in now
                for cost, choice in options(
                    k, state_mode, ran, tank_c, values[k + 1], kinds
                )
            ),
            key=lambda option: option[0],
        )
        action = kinds[name]
        heat, booster, next_c = (
            _picked(quantity, action["next"], best)
            for quantity in (action["heat_pump"], action["booster"], action["next"])
        )
        next_c = np.array([next_c])
        start = action["start"]
        costs.append(
            float(np.ravel(step_cost(tank, k, tank_c, heat, booster, start))[0])
            + float(np.ravel(shortfall_cost(tank, k, tank_c, next_c, ran))[0])
        )
        heat_pump_w.append(heat)
        booster_heat.append(booster)
        temperatures.append(float(tank_c[0]))
        starts.append(int(start))
        now, ran, tank_c = [mode], ran_next, next_c

    return TankPlan(
        heat_pump_w=tuple(heat_pump_w),
        booster_w=tuple(booster_heat),
        temperatures=tuple(temperatures),
        starts=tuple(starts),
        cost_eur=tuple(costs),
    )


def surplus_scenarios(data: MPCInput) -> list[tuple[float, np.ndarray]]:
    """The solar surplus the heat pump may use per step (W), as weighted
    scenarios: one, or three with SOLAR_SCENARIO_WEIGHTS where there is a band.
    """

    # Only solar beyond the rest of the house's own draw (the baseload) is
    # available to the heat pump, so the scenarios are of that surplus, solar
    # less baseload: its low end where little sun meets much load. The two
    # forecast errors are taken as independent, so per side the spreads add
    # in quadrature - exact for normal errors, and kept per side because
    # both are skewed.
    solar_p50 = np.asarray(data.solar_forecast_w, dtype=float)
    baseload_p50 = np.asarray(
        data.baseload_forecast_w or (0.0,) * len(data.solar_forecast_w), dtype=float
    )
    surplus_p50 = solar_p50 - baseload_p50

    if data.solar_p10_w or data.baseload_p10_w:
        solar_p10, solar_p90 = (
            np.asarray(band, dtype=float) if band else solar_p50
            for band in (data.solar_p10_w, data.solar_p90_w)
        )
        baseload_p10, baseload_p90 = (
            np.asarray(band, dtype=float) if band else baseload_p50
            for band in (data.baseload_p10_w, data.baseload_p90_w)
        )
        scenarios = zip(
            SOLAR_SCENARIO_WEIGHTS,
            (
                surplus_p50
                - np.hypot(solar_p50 - solar_p10, baseload_p90 - baseload_p50),
                surplus_p50,
                surplus_p50
                + np.hypot(solar_p90 - solar_p50, baseload_p50 - baseload_p10),
            ),
            strict=True,
        )
    else:
        scenarios = [(1.0, surplus_p50)]

    return [(weight, np.clip(surplus, 0.0, None)) for weight, surplus in scenarios]


class MPCOptimizer:
    def __init__(
        self,
        thermal_model: BoilerThermalModel,
        config: MPCConfig,
        cop_model: HeatPumpCOPModel | None = None,
        building_model: BuildingThermalModel | None = None,
        floor_circuit_model: FloorCircuitModel | None = None,
        space_cop_model: HeatPumpCOPModel | None = None,
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
        # None until runs in the zone's mode have shown how the heat pump runs
        # the floor (see features.floor). Until then a plan may put any
        # heat up to the compressor's into the zone, in runs of any length.
        self.floor_circuit_model = floor_circuit_model
        # The COP fitted on the heat pump's own runs in the zone's mode
        # (cop_heating or cop_cooling), None until there are any; heating falls
        # back to cop_model - the same Carnot model, fitted on hot water runs -
        # until then, cooling to the flat assumption, since a heating COP says
        # nothing about the cycle run backwards.
        self.space_cop_model = space_cop_model
        # None until a cop_dhw model has actually been calibrated (see
        # HeatPumpCOPIdentifier) - _power_line_coefficients() falls back to
        # the flat boiler_electrical_power_w assumption until then.
        self.cop_model = cop_model

    def solve(self, data: MPCInput) -> MPCResult:
        """The optimal plan, or the best one found within
        MPCConfig.solve_time_limit_s."""

        self._validate_input(data)

        # The tank on its own is one state over time: planned exactly, and in a
        # fraction of the time, by dynamic programming (see plan_tank). The
        # zone shares its compressor, so with it the plan stays one program.
        tank = self._tank_setting(data)

        if not self._zone_planned(data):
            return self._tank_result(data, tank)

        result, chosen = self._solve_with_zone(data, tank)

        # The tank's plans were built around the zone's runs as estimated
        # beforehand, and the zone the program settles on may hand the
        # compressor to a tank block planned as a start: that block's first
        # step then heats in full rather than with the ramp, and the steps
        # its pattern holds on overheat the tank (real data, 17 Oct 2026, a
        # solve without a previous plan: 50.7 degC against 47.9, 4.8 ct
        # dearer). Once more around the zone as settled, that plan both a
        # tank plan and the start, and the cheaper of the two kept.
        settled = tuple(int(on) for on in result.space_schedule)
        around = plan_tank(tank, settled, data.zone_cooling, data.zone_local_day)

        if settled != tuple(data.previous_space_on) and _runs(around) != chosen:
            again, _ = self._solve_with_zone(
                replace(data, previous_space_on=settled), tank
            )

            if again.objective_value < result.objective_value:
                return again

        return result

    def _solve_with_zone(
        self, data: MPCInput, tank: TankSetting
    ) -> tuple[MPCResult, tuple[tuple[int, int], ...]]:
        """The plan with the zone, and the tank's pattern in it (see _runs)."""

        model = self._build_model(data, tank)

        time_limit_s = self.config.solve_time_limit_s
        started = time.monotonic()
        solver = self._solver()
        # Pyomo's translation of the model into HiGHS took as long as the warm
        # start's own solve and more: once, then.
        solver.set_instance(model)

        # With the zone, the program decides the zone and which of a few of the
        # tank's own plans goes with it (see plan_tank): the tank's alone, and
        # around the zone's runs as the previous plan and the program's
        # relaxation have them - so the zone's need for the compressor is seen
        # by both. The program then searches the zone only: where it decided
        # both, a day with the zone took 15 s to over a minute (real data: the
        # plan within 1% of the optimum, in 1.5-8 s from the previous plan).
        # HiGHS lets go of the interpreter while it solves, so the relaxation
        # solves beside the plans that do not need it: on real data, as long as
        # either of them alone.
        def around(runs) -> TankPlan:
            return plan_tank(tank, runs, data.zone_cooling, data.zone_local_day)

        with ThreadPoolExecutor(max_workers=1) as pool:
            relaxing = pool.submit(self._relaxed_zone, model, solver)
            previous = tuple(data.previous_space_on)
            plans = [plan_tank(tank)] + ([around(previous)] if any(previous) else [])
            on = relaxing.result()

        relaxed, partly = (tuple(int(x >= share) for x in on) for share in (0.75, 0.25))

        for runs in dict.fromkeys((relaxed, partly)):
            if any(runs) and runs != previous:
                plans.append(around(runs))

        self._choose_tank(model, plans)
        solver.update()
        # The tank is one of those: the previous plan starts the zone's only -
        # or without one, the relaxation's, so the search starts from a plan
        # rather than spending most of its time finding one (real data: 70%
        # of its LP iterations in its own heuristics).
        solver.config.warmstart = self._warm_start(
            model,
            replace(
                data,
                previous_tank_on=(),
                previous_space_on=previous or relaxed,
            ),
            solver,
        )

        if solver.config.warmstart:
            solver.highs_options = {**solver.highs_options, **WARM_START_HIGHS_OPTIONS}

        # A start found is part of the solve and shares its time; one that
        # failed takes none of it - the cold solve then needs all of it.
        solver.config.time_limit = (
            time_limit_s - (time.monotonic() - started)
            if solver.config.warmstart
            else time_limit_s
        )
        results = solver.solve(model)
        stopped_early = (
            results.termination_condition == TerminationCondition.maxTimeLimit
            and results.best_feasible_objective is not None
        )

        if (
            results.termination_condition != TerminationCondition.optimal
            and not stopped_early
        ):
            raise RuntimeError(
                "MPC optimization failed. Termination condition: "
                f"{results.termination_condition}"
            )

        results.solution_loader.load_vars()

        if stopped_early:
            logger.info(
                "MPC stopped at its %.0f s time limit: best plan %.3f EUR, at most "
                "%.3f EUR from optimal",
                time_limit_s,
                results.best_feasible_objective,
                # The bound comes back in HiGHS's scaled units, the plan not.
                results.best_feasible_objective
                - results.best_objective_bound / 2.0**OBJECTIVE_SCALE_EXPONENT,
            )

        chosen = tuple(
            (
                round(pyo.value(model.boiler_on[k])),
                round(pyo.value(model.booster_on[k])),
            )
            for k in model.K
        )

        return self._extract_result(model, data, results.termination_condition), chosen

    @staticmethod
    def _relaxed_zone(model: pyo.ConcreteModel, solver: Highs) -> list[float]:
        """How far the program's relaxation - every decision allowed a
        fraction - runs the zone per step: on a copy of the instance the
        solver holds (see _warm_start). The tank is planned around where it
        runs at least three quarters on, which made a tank plan the program
        chose that no other did (real data, eight days), and around where it
        runs a quarter: a later zone run the relaxation only partly plans,
        which a tank block placed around the first run alone had taken (a
        cold-tank test day: 112 EUR of comfort missed against 18)."""

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        # By interior point, without the crossover to a vertex: only the
        # zone's values are read, not a basis. The simplex took 25.6 s on one
        # real day (196 000 degenerate iterations), the interior point 0.4 s,
        # and the same zone on all eight inputs compared.
        highs.setOptionValue("solver", "ipm")
        highs.setOptionValue("run_crossover", "off")
        relaxation = solver._solver_model.getLp()
        relaxation.integrality_ = []
        highs.passModel(relaxation)
        highs.run()
        values = highs.getSolution().col_value
        columns = solver._pyomo_var_to_solver_var_map

        return [
            values[columns[id(model.space_on[k])]]
            if id(model.space_on[k]) in columns
            else 0.0
            for k in model.K
        ]

    @staticmethod
    def _choose_tank(model: pyo.ConcreteModel, plans: list[TankPlan]) -> None:
        """The tank's heat pump and booster on or off as in exactly one of
        `plans`, the program to choose which."""

        patterns = list(dict.fromkeys(_runs(plan) for plan in plans))
        model.tank_plan = pyo.Var(range(len(patterns)), domain=pyo.Binary)
        model.tank_plan_constraint = pyo.ConstraintList()
        model.tank_plan_constraint.add(
            sum(model.tank_plan[i] for i in range(len(patterns))) == 1
        )

        for k in model.K:
            for variable, source in ((model.boiler_on, 0), (model.booster_on, 1)):
                if not variable[k].fixed:
                    model.tank_plan_constraint.add(
                        variable[k]
                        == sum(
                            model.tank_plan[i] * pattern[k][source]
                            for i, pattern in enumerate(patterns)
                        )
                    )

    def _zone_planned(self, data: MPCInput) -> bool:
        """Whether the zone is planned beside the tank: with a building model
        and a zone state and target to plan it from (see _add_space_heating)."""

        return (
            self.building_model is not None
            and bool(data.zone_state)
            and bool(data.zone_target_temperature)
        )

    @staticmethod
    def _tank_result(data: MPCInput, tank: TankSetting) -> MPCResult:
        """The tank's own plan (see plan_tank) as the plan's result: its
        draw per step as it was costed, its grid share expected over the solar
        scenarios."""

        plan = plan_tank(tank)
        power_w, electricity_kwh, grid_kwh = [], [], []

        for k in range(tank.num_steps):
            heat_pump_w, booster_w = plan.heat_pump_w[k], plan.booster_w[k]
            drawn = float(
                draw_w(
                    tank,
                    k,
                    plan.temperatures[k],
                    heat_pump_w,
                    booster_w,
                    plan.starts[k],
                )
            )
            power_w.append(drawn)
            electricity_kwh.append(max(drawn, 0.0) * tank.dt_hours[k] / 1000.0)
            grid_kwh.append(
                float(grid_w(tank, k, drawn, heat_pump_w, booster_w))
                * tank.dt_hours[k]
                / 1000.0
            )

        status = str(TerminationCondition.optimal)

        return MPCResult(
            schedule=tuple(
                int(heat_pump_w > 0.0 or booster_w > 0.0)
                for heat_pump_w, booster_w in zip(
                    plan.heat_pump_w, plan.booster_w, strict=True
                )
            ),
            temperatures=plan.temperatures,
            electrical_power_w=tuple(power_w),
            heat_w=tuple(
                heat_pump_w + booster_w
                for heat_pump_w, booster_w in zip(
                    plan.heat_pump_w, plan.booster_w, strict=True
                )
            ),
            objective_value=sum(plan.cost_eur),
            solver_status=status,
            termination_condition=status,
            electricity_kwh=sum(electricity_kwh),
            grid_kwh=sum(grid_kwh),
            electricity_step_kwh=tuple(electricity_kwh),
            grid_step_kwh=tuple(grid_kwh),
        )

    @staticmethod
    def _solver() -> Highs:
        solver = Highs()
        solver.highs_options = {
            "user_objective_scale": OBJECTIVE_SCALE_EXPONENT,
            "mip_abs_gap": MIP_ABSOLUTE_GAP_EUR * 2.0**OBJECTIVE_SCALE_EXPONENT,
            "mip_rel_gap": 0.0,
            "presolve": PRESOLVE,
        }
        solver.config.load_solution = False
        # HiGHS's own progress log, at INFO by default, buried the plan's own
        # lines under dozens per solve: kept, but for debugging.
        solver.config.log_level = logging.DEBUG

        return solver

    def _warm_start(
        self, model: pyo.ConcreteModel, data: MPCInput, solver: Highs
    ) -> bool:
        """Load the previous plan, one quarter on, into the model as the
        solve's starting point; whether there was one that holds.

        A solve stopped at its time limit publishes the best plan found by
        then, and on the Home Assistant host that was at times far from
        optimal: on 2 Oct two runs where one would do and a run before
        sunrise, each 0.23 EUR dearer than the plan the solve before had
        found. Started from that plan, the solve can only improve on it.

        The previous plan's decisions are fixed beyond the first step - the
        one that changed since - and the rest solved for, a small problem,
        so the start meets this plan's constraints and current state. Solved
        on a copy of the HiGHS model the solver holds, not on that one: on an
        instance a warm start had just run on - its previous plan no longer
        held - HiGHS proved a plan optimal at the root that was not, on the
        very same model (real data: 0.4674 EUR against an optimum of 0.4656).
        Through the private side of Pyomo's interface (its HiGHS model and
        column map), which has no public way to either."""

        if not data.previous_tank_on and not data.previous_space_on:
            return False

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("time_limit", self.config.warm_start_time_limit_s)
        highs.setOptionValue("presolve", PRESOLVE)
        highs.passModel(solver._solver_model.getModel())
        columns = solver._pyomo_var_to_solver_var_map

        for variable, previous in (
            (model.tank_heating, data.previous_tank_on),
            (model.space_on, data.previous_space_on),
        ):
            for k, on in enumerate(previous[: len(model.K)]):
                if k > 0 and not variable[k].fixed:
                    highs.changeColBounds(columns[id(variable[k])], on, on)

        highs.run()

        if (
            highs.getInfo().primal_solution_status
            != highspy.SolutionStatus.kSolutionStatusFeasible
        ):
            logger.info("Previous plan does not hold: solving cold")
            return False

        start = highs.getSolution().col_value

        for variable_id, column in columns.items():
            solver._vars[variable_id][0].set_value(start[column], skip_validation=True)

        return True

    def _validate_input(self, data: MPCInput) -> None:
        horizon = len(data.solar_forecast_w)

        if horizon < 2:
            raise ValueError("MPC horizon must contain at least 2 steps.")

        for name in ("import_price_eur_per_kwh", "export_price_eur_per_kwh"):
            if getattr(data, name) and len(getattr(data, name)) != horizon:
                raise ValueError(
                    f"{name} must be empty or have the same length as "
                    f"solar_forecast_w ({horizon}), got {len(getattr(data, name))}."
                )

        if any(
            export > import_ for import_, export in zip(*self.prices(data), strict=True)
        ):
            raise ValueError(
                "export_price_eur_per_kwh may not exceed import_price_eur_per_kwh."
            )

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

        if data.first_step_hours is not None and not (
            0.0 < data.first_step_hours <= self.config.step_hours
        ):
            raise ValueError(
                "first_step_hours must lie within one step, got "
                f"{data.first_step_hours}."
            )

        if data.baseload_forecast_w and len(data.baseload_forecast_w) != horizon:
            raise ValueError(
                "baseload_forecast_w must be empty or have the same length as "
                f"solar_forecast_w ({horizon}), got {len(data.baseload_forecast_w)}."
            )

        if len(data.baseload_p10_w) != len(data.baseload_p90_w) or (
            data.baseload_p10_w and len(data.baseload_p10_w) != horizon
        ):
            raise ValueError(
                "baseload_p10_w and baseload_p90_w must both be empty or both have "
                f"the same length as solar_forecast_w ({horizon}), got "
                f"{len(data.baseload_p10_w)} and {len(data.baseload_p90_w)}."
            )

        if self.config.step_hours <= 0:
            raise ValueError("step_hours must be greater than zero.")

        if self.config.boiler_electrical_power_w < 0:
            raise ValueError("boiler_electrical_power_w cannot be negative.")

        if self.config.compressor_min_runtime_steps < 1:
            raise ValueError("compressor_min_runtime_steps must be at least 1.")

    def prices(self, data: MPCInput) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The import and export price per step (EUR/kWh): the input's
        own, or the config's flat ones where it carries none."""

        horizon = len(data.solar_forecast_w)

        return (
            data.import_price_eur_per_kwh
            or (self.config.import_price_eur_per_kwh,) * horizon,
            data.export_price_eur_per_kwh
            or (self.config.export_price_eur_per_kwh,) * horizon,
        )

    def _tank_setting(self, data: MPCInput) -> TankSetting:
        """The tank over the plan's steps as physics and the rules fix it, for
        the plan as a whole (_build_model) and the tank's own (plan_tank)."""

        num_steps = len(data.solar_forecast_w)
        # Every step a quarter hour, the resolution the heat pump is driven at,
        # but the first: the plan starts now, partway into it.
        dt_hours = [self.config.step_hours] * num_steps

        if data.first_step_hours is not None:
            dt_hours[0] = data.first_step_hours

        # When each step begins, in hours from now.
        step_start_h = np.cumsum([0.0, *dt_hours[:-1]])

        def steps_within(hours: float, first: int = 0) -> range:
            """The steps from `first` on that begin within `hours` of it: what
            a duration covers counted in the plan's own steps, the first of
            which may be the rest of a quarter."""

            end = first

            while (
                end < num_steps
                and step_start_h[end] - step_start_h[first] < hours - 1e-9
            ):
                end += 1

            return range(first, end)

        solar_scenarios = surplus_scenarios(data)
        tap_w = data.tap_forecast_w or (0.0,) * num_steps
        outdoor_c = list(data.outdoor_temperature_forecast)
        heat_w = [
            self._heat_at(outdoor_c[k] if outdoor_c else None) for k in range(num_steps)
        ]
        target_c = data.target_temperature_top
        overall_target_max = max(data.target_temperature_top)
        heat_pump_max_c = self.thermal_model.heat_pump_max_tank_temperature_c
        identified_max_c = self.thermal_model.max_tank_temperature_c
        max_tank_c = (
            identified_max_c
            if identified_max_c is not None
            else DEFAULT_MAX_TANK_TEMPERATURE_C
        )
        booster_heat_w = self.thermal_model.booster_heat_w

        initial_temperature = self._initial_temperature(data)

        # Exact zero-order-hold dynamics for the lumped tank node. Unlike the
        # two-node calibration model, this simplified model has no on/off
        # mixing-regime switch (only the heat input, not the loss
        # coefficient, depends on boiler_on), so (A_d, B_d) depends only on
        # each step's own duration - cached per distinct duration (the first
        # step and the rest) rather than recomputed per step.
        a, b = lumped_tank_state_space(
            self.thermal_model.volume_l,
            self.thermal_model.ua_top_w_per_k + self.thermal_model.ua_bottom_w_per_k,
        )
        discretization_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        def discretized(hours: float) -> tuple[np.ndarray, np.ndarray]:
            if hours not in discretization_cache:
                discretization_cache[hours] = discretize_zoh(a, b, hours * 3600.0)

            return discretization_cache[hours]

        ambient_c = float(data.ambient_temperature)

        # A draw replaces hot water with cold from the mains, so taps cannot
        # take the tank below the mains' temperature, however much the forecast
        # expects: held there along the tank left unheated, where it would get
        # that cold first - a heated tank has heat to give. The forecast is the
        # heat draws took from the tanks it was learned on (real plan, 16 Oct
        # 2026: an emptied tank at 25 degC at 12.3 degC by the next morning's
        # run, the mains at 15.5 and the tank measured at 19.5).
        # A target the tank can reach by then is heated to, so the tank left
        # unheated starts again from it: held to a tank never heated again, the
        # next day's evening showers were cut to nothing after a run at noon
        # (real plan, 18 Oct 2026: 1.07 of 4.47 kWh drawn).
        max_heat_w = max(max(heat_w), booster_heat_w or 0.0)
        heat_pump_limit_c = (
            min(heat_pump_max_c, max_tank_c)
            if heat_pump_max_c is not None
            else max_tank_c
        )

        if data.mains_water_temperature is not None:
            tap_w = list(tap_w)
            tank_c = ceiling_c = initial_temperature
            hottest_c = max(
                initial_temperature,
                max_tank_c if booster_heat_w is not None else heat_pump_limit_c,
            )

            for k in range(num_steps):
                tank_c = max(tank_c, min(float(target_c[k]), ceiling_c))
                a_d, b_d = discretized(dt_hours[k])
                coast_c = float(a_d[0, 0] * tank_c + b_d[0, 0] * ambient_c)
                floor_c = min(data.mains_water_temperature, coast_c)
                tap_w[k] = min(float(tap_w[k]), (coast_c - floor_c) / -b_d[0, 2])
                tank_c = coast_c + b_d[0, 2] * tap_w[k]
                ceiling_c = min(
                    hottest_c,
                    float(
                        a_d[0, 0] * ceiling_c
                        + b_d[0, 0] * ambient_c
                        + b_d[0, 1] * max_heat_w
                        + b_d[0, 2] * tap_w[k]
                    ),
                )

        def unheated(start_c: float, first: int = 0) -> list[float]:
            """The tank temperature at each step from `first` on with no heating
            from start_c on."""

            temperatures = [start_c]

            for k in range(first, num_steps - 1):
                a_d, b_d = discretized(dt_hours[k])
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

        # And only in the steps it can serve such a target from: where a tank at
        # the heat pump's limit would have cooled below it by then, but one at
        # the boiler's maximum not yet - and the steps before, that the booster
        # needs to lift the tank from the one to the other. Earlier, whatever it
        # heats is lost again before the target; after it, the tank only cools.
        # A 60 degC disinfection target otherwise gave the booster a decision in
        # every step of the horizon and doubled the solve time. A tank already
        # above the limit keeps it everywhere: a booster run may be under way.
        booster_useful = [False] * num_steps

        if booster_possible and heat_pump_max_c is not None:
            if initial_temperature > heat_pump_max_c:
                booster_useful = [True] * num_steps
            else:
                holds = []

                for k in range(num_steps):
                    from_limit = unheated(heat_pump_max_c, k)
                    from_maximum = unheated(max_tank_c, k)
                    holds.append(
                        any(
                            from_limit[j - k] < float(target_c[j])
                            and (j == k + 1 or from_maximum[j - k] >= target_c[j])
                            for j in range(k + 1, num_steps)
                        )
                    )

                lift_j = (
                    self.thermal_model.volume_l
                    * RHO_WATER_KG_PER_L
                    * CP_WATER_J_PER_KG_K
                    * max(max_tank_c - heat_pump_max_c, 0.0)
                )
                lift_steps = math.ceil(
                    lift_j / (booster_heat_w * self.config.step_hours * 3600.0)
                )
                booster_useful = [
                    any(holds[k : k + lift_steps + 1]) for k in range(num_steps)
                ]

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
        t_floor = unheated(initial_temperature)
        t_ceiling = [initial_temperature]

        for k in range(num_steps - 1):
            a_d, b_d = discretized(dt_hours[k])
            passive = b_d[0, 0] * ambient_c + b_d[0, 2] * float(tap_w[k])
            heated = a_d[0, 0] * t_ceiling[-1] + passive + b_d[0, 1] * max_heat_w
            t_ceiling.append(min(t_upper, float(heated)))

        # A heat pump run does not end where the plan stops it: it settles
        # setpoint_overshoot_k above its setpoint (see the target rule in
        # _build_model), where the target and that overshoot fit under the
        # step's ceiling and the heat pump's own limit.
        overshoot_k = self.thermal_model.setpoint_overshoot_k or 0.0
        required_c = [min(float(target_c[k]), t_ceiling[k]) for k in range(num_steps)]
        overshoot_c = [
            0.0
            if required_c[k] + overshoot_k
            > min(t_ceiling[k], heat_pump_max_c or math.inf)
            else overshoot_k
            for k in range(num_steps)
        ]
        # The booster where it can be of use (see booster_useful), and where
        # the tank can be at the heat pump's limit by the step's end at all.
        booster_allowed = [
            booster_useful[k]
            and heat_pump_max_c is not None
            and t_ceiling[min(k + 1, num_steps - 1)] >= heat_pump_max_c
            for k in range(num_steps)
        ]
        min_runtime_hours = (
            self.config.compressor_min_runtime_steps * self.config.step_hours
        )
        # A run under way below the heat pump's limit keeps heating until its
        # minimum runtime has passed (see the running_run rule).
        running_steps = (
            len(steps_within(min_runtime_hours - data.compressor_elapsed_hours))
            if data.boiler_on_current
            and data.compressor_elapsed_hours > 0.0
            and not (
                heat_pump_max_c is not None and initial_temperature >= heat_pump_max_c
            )
            else 0
        )
        import_price, export_price = self.prices(data)

        return TankSetting(
            dt_hours=dt_hours,
            a_continuous=a,
            b_continuous=b,
            ambient_c=ambient_c,
            tap_w=tuple(tap_w),
            heat_w=heat_w,
            initial_c=initial_temperature,
            floor_c=t_floor,
            ceiling_c=t_ceiling,
            heat_pump_max_c=heat_pump_max_c,
            max_tank_c=max_tank_c,
            heat_pump_limit_c=heat_pump_limit_c,
            booster_heat_w=booster_heat_w,
            booster_possible=booster_possible,
            booster_allowed=booster_allowed,
            required_c=required_c,
            overshoot_c=overshoot_c,
            power_lines=[
                self._power_line_coefficients(
                    outdoor_c[k] if outdoor_c else None, overall_target_max
                )
                for k in range(num_steps)
            ],
            ramp=[self._ramp_fraction(hours) for hours in dt_hours],
            start_power_w=(
                self.cop_model.start_step_power_w if self.cop_model else 0.0
            ),
            booster_draw_per_heat=self._booster_draw_per_heat,
            import_price=tuple(import_price),
            export_price=tuple(export_price),
            scenarios=solar_scenarios,
            weight_switching=self.config.weight_switching,
            weight_slack=self.config.weight_temperature_slack,
            min_runtime_hours=min_runtime_hours,
            min_off_steps=self.config.heat_pump_min_off_steps,
            idle_steps=(
                0
                if data.boiler_on_current
                else len(
                    steps_within(
                        self.config.heat_pump_min_off_steps * self.config.step_hours
                        - data.idle_elapsed_hours
                    )
                )
            ),
            running=bool(data.boiler_on_current),
            running_steps=running_steps,
            compressor_busy=bool(data.space_on_current and not data.boiler_on_current),
        )

    def _build_model(
        self, data: MPCInput, tank: TankSetting | None = None
    ) -> pyo.ConcreteModel:
        tank = tank or self._tank_setting(data)
        num_steps = tank.num_steps
        dt_hours = tank.dt_hours
        steps_within = tank.steps_within
        discretized = tank.discretized
        unheated = tank.unheated
        solar_scenarios = tank.scenarios
        tap_w = tank.tap_w
        outdoor_c = list(data.outdoor_temperature_forecast)
        heat_w = tank.heat_w
        target_c = data.target_temperature_top
        heat_pump_max_c = tank.heat_pump_max_c
        max_tank_c = tank.max_tank_c
        booster_heat_w = tank.booster_heat_w
        initial_temperature = tank.initial_c
        ambient_c = tank.ambient_c
        booster_possible = tank.booster_possible
        t_floor = tank.floor_c
        t_ceiling = tank.ceiling_c

        model = pyo.ConcreteModel()
        model.K = pyo.RangeSet(0, num_steps - 1)

        model.T = pyo.Var(model.K, bounds=lambda m, k: (t_floor[k], t_ceiling[k]))

        model.boiler_on = pyo.Var(model.K, domain=pyo.Binary)

        model.compressor_start = pyo.Var(model.K, domain=pyo.Binary)

        # Heat pump heat (W): continuous up to q_in_nominal_w while on. Below
        # that, a step is only partly used: the heat pump stops by itself once
        # the tank reaches its setpoint (published per run - see
        # app.optimization), or modulates down near its tank limit (real data:
        # ~7 kW at 36-47 degC, ~3 kW at 55 degC).
        model.q_heat_pump_w = pyo.Var(model.K, bounds=lambda m, k: (0.0, heat_w[k]))

        # The booster heater is a resistive element: it cannot modulate, so while
        # it runs it heats at its identified rating. Its heat per step is still
        # continuous, for the same reason as the heat pump's: the tank's
        # thermostat cuts it out at the boiler's maximum partway through a step,
        # and it takes over from the heat pump partway through one (see
        # heat_source_constraints). As a whole-step decision it could do
        # neither: it overshot the maximum, and a heat pump reaching its limit
        # early in a step left the rest of that step empty before the booster
        # started - a gap the real run never has.
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

        # initial_temperature (above) is the tank mixed: the heat it holds, cold
        # layer below the sensors included (see
        # BoilerThermalModel.mixed_temperature). The sensors' plain average
        # overstated it by up to 11 K after tapping, and runs were planned
        # accordingly short - half the real electricity on a warm tank.
        model.initial_temperature = pyo.Constraint(
            expr=model.T[0] == float(initial_temperature)
        )

        # A target above a step's ceiling cannot be met by any plan; that part of
        # the shortfall is the same constant whatever is decided, so it is left
        # out and the slack only measures what planning can still change. The
        # optimum is the same plan - but an unreachable legionella target had
        # inflated the objective into the thousands, and proving a plan to the
        # cent against that took the solver half a minute.
        # A heat pump run does not end where the plan stops it: it stops on its
        # setpoint and then settles setpoint_overshoot_k above it (see
        # BoilerThermalModel), and that heat is part of what the run costs and
        # how long it takes. So a target after a planned run is met with that
        # much to spare - the setpoint published is the run's planned end less
        # it (see app.optimization) - while heat already in the tank counts in
        # full: a later replan that finds the tank a fraction of a degree
        # warmer than needed sees a buffer, not a reason for another run. ran[k]
        # is whether the heat pump heats the tank at or before step k's start:
        # the bounds leave it 0 or 1 once the runs are decided. Not where the
        # booster finishes the tank: its thermostat, not a setpoint, ends it.
        model.ran = pyo.Var(model.K, bounds=(0.0, 1.0))
        model.ran_constraint = pyo.ConstraintList()
        model.ran_constraint.add(model.ran[0] >= int(data.boiler_on_current))

        for k in range(1, num_steps):
            model.ran_constraint.add(model.ran[k] >= model.ran[k - 1])
            model.ran_constraint.add(model.ran[k] >= model.boiler_on[k - 1])

        def required(m: pyo.ConcreteModel, k: int):
            return tank.required_c[k] + tank.overshoot_c[k] * m.ran[k]

        def temperature_rule(m: pyo.ConcreteModel, k: int):
            return m.T[k] + m.slack[k] >= required(m, k)

        model.temperature_constraint = pyo.Constraint(
            model.K,
            rule=temperature_rule,
        )

        # T[k] is the temperature at the START of step k, so the rule above only
        # checks a step's warmest moment while the tank is coasting: the plan
        # could meet a target at a step's start while the tank sags below it by
        # the step's end (seen on this installation with one-hour steps, 0.25 K
        # by 18:45 against an 18:00 target; some 0.03 K over a quarter hour).
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

            return m.T[k + 1] + m.slack[k] >= required(m, k)

        model.end_temperature_constraint = pyo.Constraint(
            model.K,
            rule=end_temperature_rule,
        )

        # The compressor runs now if it serves either demand: a plan that goes
        # on heating the zone it is heating already makes no start.
        initial_compressor_on = int(data.boiler_on_current or data.space_on_current)

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
                return m.compressor_start[k] >= (
                    compressor(m, k) - initial_compressor_on
                )

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
            previous = initial_compressor_on if k == 0 else compressor(model, k - 1)
            model.start_exact.add(model.compressor_start[k] <= compressor(model, k))
            model.start_exact.add(model.compressor_start[k] <= 1 - previous)

        # A tank block begins (tank_start): with a start of the compressor, or
        # handed over from a heating run of the zone in the step before
        # (tank_handover, the exact product of the two binaries). The ramp
        # (see _ramp_fraction) hangs on the blocks that start the compressor,
        # tank_start - tank_handover, not on every start - a run the zone opens
        # heats no tank, and charged the ramp regardless, the tank's heat in
        # that step was bounded below zero. A hand-over carries none, as a run
        # the tank hands to the zone does not (see _add_space_heating): the
        # ramp is the compressor's own rise to full output, timed on the heat
        # it delivers into the water, and handed over it is at speed already.
        # The loop coming up to the tank's temperature is no more a loss there
        # than after a start, where the ramp and the tank were identified with
        # it inside: on this installation it rests at 24 degC (120 starts,
        # Jun-Oct 2026), the floor's supply, and takes some 0.45 kWh either way.
        # Costed as a start's ramp at full output instead, a hand-over came out
        # some 6 ct dearer than a start, and the tank went before the zone
        # rather than after it, hours ahead of its target (17 Oct 2026). Only a
        # start draws the start's measured power (start_step_power_w). Cooling
        # hands over to no tank (see _add_space_heating).
        model.tank_start = pyo.Var(model.K, bounds=(0.0, 1.0))
        model.tank_handover = pyo.Var(model.K, bounds=(0.0, 1.0))

        for k in range(num_steps):
            previous_on = (
                int(data.boiler_on_current) if k == 0 else model.boiler_on[k - 1]
            )
            zone_before = (
                int(data.space_on_current) if k == 0 else model.space_on[k - 1]
            )

            if data.zone_cooling:
                model.tank_handover[k].fix(0.0)
            else:
                for bound in (
                    model.tank_handover[k] <= model.boiler_on[k],
                    model.tank_handover[k] <= zone_before,
                    model.tank_handover[k] >= model.boiler_on[k] + zone_before - 1,
                ):
                    model.start_exact.add(bound)

            model.start_exact.add(
                model.tank_start[k]
                <= model.compressor_start[k] + model.tank_handover[k]
            )
            model.start_exact.add(model.tank_start[k] <= model.boiler_on[k])
            model.start_exact.add(model.tank_start[k] >= model.tank_handover[k])
            model.start_exact.add(
                model.tank_start[k]
                >= model.compressor_start[k] + model.boiler_on[k] - 1
            )
            # Each step the tank comes on in is such a block - the same plans,
            # but in the solver's relaxation the product above let a tenth of a
            # run come on with no start at all.
            model.start_exact.add(
                model.tank_start[k] >= model.boiler_on[k] - previous_on
            )

        self._add_space_heating(model, data, dt_hours, outdoor_c)

        model.minimum_runtime = pyo.ConstraintList()

        min_runtime = self.config.compressor_min_runtime_steps

        for start in range(num_steps):
            for k in steps_within(min_runtime * self.config.step_hours, start):
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

            # The tank's own part of a run lasts as long, also where the zone
            # makes up the rest of the compressor's time: turning the valve to
            # the tank first reheats the loops and plate exchanger to tank
            # temperature (an estimated 0.4-0.5 kWh), a hand-over the plan does
            # not price.
            # Without this, one quarter of hot water tacked onto a zone run met
            # the rule above. The booster counts for the same reason as there.
            previous_tank = (
                int(data.boiler_on_current)
                if start == 0
                else model.boiler_on[start - 1]
            )

            for k in steps_within(min_runtime * self.config.step_hours, start):
                model.minimum_runtime.add(
                    heating(model, k) >= model.boiler_on[start] - previous_tank
                )

        # A run already heating keeps heating until its minimum runtime has
        # passed. Otherwise the next replan, minutes after the start, can plan it
        # off at step 0 while the heat pump finishes the run anyway - the plan,
        # and anything acting on it, would flip for nothing. Only a compressor
        # run below the heat pump's limit: once the booster has taken over (no
        # compressor time, see MPCInput.compressor_elapsed_hours), or the tank
        # is at that limit - a setpoint set by hand above it - the compressor
        # has nothing left to give and the booster's thermostat ends the run.
        # Forcing it on for a whole minimum runtime pushed a tank near its
        # maximum past the cut-out: no plan at all (real data, 5 Oct: a run set
        # to 60 degC by hand, at 59.5 degC).
        remaining = steps_within(
            min_runtime * self.config.step_hours - data.compressor_elapsed_hours
        )
        at_limit = (
            heat_pump_max_c is not None and initial_temperature >= heat_pump_max_c
        )
        model.running_run = pyo.ConstraintList()

        if (
            data.boiler_on_current
            and data.compressor_elapsed_hours > 0.0
            and not at_limit
        ):
            for k in remaining:
                model.running_run.add(heating(model, k) >= 1)

        # The heat pump stays off for heat_pump_min_off_steps after a run ends,
        # and does not start at all until that long after the last real run
        # ended.
        model.minimum_off_time = pyo.ConstraintList()
        min_off = self.config.heat_pump_min_off_steps

        for stop in range(num_steps):
            for offset in range(1, min_off + 1):
                k = stop + offset

                if k >= num_steps:
                    continue

                model.minimum_off_time.add(
                    heating(model, k)
                    <= 1 - (heating(model, stop) - heating(model, stop + 1))
                )

        if not data.boiler_on_current:
            for k in steps_within(
                min_off * self.config.step_hours - data.idle_elapsed_hours
            ):
                model.minimum_off_time.add(heating(model, k) == 0)

        model.thermal_dynamics = pyo.ConstraintList()

        for k in range(num_steps - 1):
            a_d, b_d = discretized(dt_hours[k])

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
        heat_pump_limit_c = tank.heat_pump_limit_c

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

        # Where a heat pump run ends, in step k (on[k] - on[k + 1] +
        # tank_start[k + 1], 0 or 1 in a real plan, see the rule for a run's
        # full steps below), the booster may take over in that step or from the
        # next: once.
        model.handover_in = pyo.Var(model.K, bounds=(0.0, 1.0))
        model.handover_next = pyo.Var(model.K, bounds=(0.0, 1.0))

        for k in range(num_steps):
            ends = (
                model.boiler_on[k] - model.boiler_on[k + 1] + model.tank_start[k + 1]
                if k + 1 < num_steps
                else model.boiler_on[k]
            )
            model.heat_source_constraints.add(
                model.handover_in[k] + model.handover_next[k] <= ends
            )

        def busy(k: int):
            """The part of step k the tank's sources run. A heat pump starting
            in it runs longer than its heat alone says: the ramp's shortfall
            (see _ramp_fraction) is time, not heat. Only used where the booster
            may run."""

            return (
                model.q_heat_pump_w[k] / heat_w[k]
                + (1.0 - self._ramp_fraction(dt_hours[k]))
                * (model.tank_start[k] - model.tank_handover[k])
                + model.q_booster_w[k] / booster_heat_w
            )

        for k in range(num_steps):
            on = model.boiler_on[k]

            # What this step can deliver: full output, less the part of it the
            # compressor spends coming up to speed if it starts here (see
            # _ramp_fraction).
            available = heat_w[k] * (
                on
                - (1.0 - self._ramp_fraction(dt_hours[k]))
                * (model.tank_start[k] - model.tank_handover[k])
            )

            model.heat_source_constraints.add(model.q_heat_pump_w[k] <= available)
            add_product(model.heat_pump_temperature[k], on, k)

            # The heat pump runs at its own power until the setpoint and then
            # stops, so a step it still runs after is a full one: only a run's
            # last step is partly used. Otherwise the plan could spread a run's
            # heat over part-used steps to follow the sun, which the heat pump
            # never does. With the minimum runtime this also keeps a run from
            # delivering less than one full step.
            #
            # A run stops after step k where on[k] - on[k + 1] + tank_start[k + 1]
            # is 1, and nowhere else: tank_start marks every block's first
            # step. The same plans as 1 - on[k + 1], but in the
            # solver's relaxation a fraction of 'on' held over many steps then
            # stops only as far as it also starts, and pays for that start
            # (unit commitment's start/stop formulation). On 1 - on[k + 1] the
            # relaxation kept the tank at 0.175 'on' for 15 hours without
            # heating or starting: a bound 0.29 EUR under the plan, and 40 s to
            # prove it.
            if k + 1 < num_steps:
                model.heat_source_constraints.add(
                    model.q_heat_pump_w[k]
                    >= available
                    - heat_w[k]
                    * (
                        model.boiler_on[k]
                        - model.boiler_on[k + 1]
                        + model.tank_start[k + 1]
                    )
                )

            # The heat pump modulates, so it stops exactly at its limit: the tank
            # temperature after a step it runs in - the dynamics below, times on -
            # stays under it. Skipped where the step's ceiling already rules that
            # out.
            if k + 1 < num_steps and t_ceiling[k + 1] > heat_pump_limit_c:
                a_d, b_d = discretized(dt_hours[k])
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
            if not tank.booster_allowed[k]:
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
            # it never starts a DHW run by itself. As a flow: a booster run goes
            # on from itself or begins where a heat pump run ends - in that run's
            # last step or the one after - and each run hands over once. Stated
            # as following any heating step, the booster counted as its own
            # predecessor, and in the solver's relaxation a heat pump run 0.19
            # 'on' grew into a booster 0.78 'on' without a start of its own.
            if k == 0:
                previous = float(data.boiler_on_current)
            else:
                previous = model.booster_on[k - 1] + model.handover_next[k - 1]
            model.heat_source_constraints.add(
                booster <= previous + model.handover_in[k]
            )

            # Both sources heat until their setpoint, so a step the booster runs
            # in on its own follows one with no idle time left - the run is one
            # piece, as it really is.
            if k > 0:
                model.heat_source_constraints.add(busy(k - 1) >= booster - on)

            if last:
                continue

            a_d, b_d = discretized(dt_hours[k])
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

        # A run's steps before its last are full, so a run cannot reach a step
        # its own heat would lift the tank past that step's ceiling at, even
        # from the tank unheated: the heat pump heats the tank at step k only
        # if a run that can still be on there started since (-1 being the run
        # under way). Every real plan meets this already; it is here for the
        # solver's relaxation, where a tenth of 'on' held for five hours - a
        # run the tank could never take - otherwise came with a tenth of one
        # start (unit commitment's maximum up time).
        run_starts = ([-1] if data.boiler_on_current else []) + list(range(num_steps))
        reaching: list[list[int]] = [[] for _ in range(num_steps)]

        for j in run_starts:
            rise_k = 0.0

            for k in range(max(j, 0), num_steps):
                if rise_k > t_ceiling[k] - t_floor[k] + 1e-6:
                    break

                reaching[k].append(j)
                a_d, b_d = discretized(dt_hours[k])
                ramp = self._ramp_fraction(dt_hours[k]) if k == j else 1.0
                rise_k = float(a_d[0, 0] * rise_k + b_d[0, 1] * heat_w[k] * ramp)

        model.run_length = pyo.ConstraintList()

        for k in range(num_steps):
            model.run_length.add(
                model.boiler_on[k]
                <= sum(1 if j < 0 else model.tank_start[j] for j in reaching[k])
            )

        # Heat above the heat pump's limit can only come from the booster: the
        # tank's excess over that limit at any step is at most what it held at
        # the start plus the booster heat put in since - losses and taps only
        # lower it. Every real plan meets this already; it is here for the
        # solver's relaxation, where a heat pump at a small fraction of 'on' in
        # every step could otherwise heat past its limit a little at a time
        # and never need the booster at all (real data: a lower bound of 0.39
        # EUR against a 0.81 EUR legionella plan, and 30 s without proving it).
        if booster_possible:
            heat_capacity_j_per_k = (
                self.thermal_model.volume_l * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K
            )
            # In kelvin of the tank's: in joules the row's coefficients ran to
            # 1e6 and its bound to 5e7, far from the rest of the model's.
            booster_k = 0.0

            for k in range(num_steps - 1):
                booster_k = (
                    booster_k
                    + model.q_booster_w[k]
                    * dt_hours[k]
                    * 3600.0
                    / heat_capacity_j_per_k
                )

                if t_ceiling[k + 1] > heat_pump_limit_c:
                    model.heat_source_constraints.add(
                        model.T[k + 1] - heat_pump_limit_c
                        <= max(initial_temperature - heat_pump_limit_c, 0.0) + booster_k
                    )

        # A target the tank cannot hold without heating needs a run: if even the
        # hottest the tank can be at step m, left unheated, has cooled below
        # the target by step k, something heats in between - the tank already
        # heating at m, or a tank block that begins after it (tank_start, on a
        # start or handed over from the zone): the booster only takes one
        # over. Stated for the latest such m, the
        # strongest. Every real plan meets this already; in the solver's
        # relaxation a heat pump a little 'on' in every step otherwise heated
        # without ever paying for a start (real data: 0.25 starts against the 2
        # a legionella plan really makes), and with any start or the zone
        # counted, a zone run met it in place of the tank's own (a lower bound
        # one start short on a day with both).
        model.run_needed = pyo.ConstraintList()
        # The tank is linear, so its excess over the unheated trajectory only
        # decays: left alone from x at step m, it is t_floor[k] + (x -
        # t_floor[m]) * kept[k] / kept[m] at step k.
        kept = np.cumprod([1.0] + [tank.step(j)[0] for j in range(num_steps - 1)])

        for k in range(1, num_steps):
            target = min(float(target_c[k]), t_ceiling[k])

            if target <= t_floor[k]:
                continue

            for m in range(k - 1, -1, -1):
                cooled_c = t_floor[k] + (t_ceiling[m] - t_floor[m]) * kept[k] / kept[m]

                if cooled_c < target:
                    # Or the plan misses the target: unheated, by at least this.
                    model.run_needed.add(
                        sum(model.tank_start[j] for j in range(m + 1, k))
                        + model.tank_heating[m]
                        + model.slack[k] / (target - cooled_c)
                        >= 1
                    )
                    break

        # The same from the tank as it is now, the strongest of them: a tank
        # that cannot hold the first target it would miss from its present
        # temperature needs a tank block that begins before it, unless the
        # tank heats now (the booster only takes over a run). The rule
        # above starts
        # from the hottest the tank could be, so it only asks for heating
        # shortly before the target, and the relaxation met it with a little of
        # 'on' spread over many steps (real data, 8 Oct 2026: 0.34 starts
        # against the 1 the plan makes, a lower bound of 0.48 EUR against 0.70).
        if not data.boiler_on_current:
            cooled = unheated(initial_temperature)

            for k in range(1, num_steps):
                target = min(float(target_c[k]), t_ceiling[k])

                if target > t_floor[k] and cooled[k] < target:
                    model.run_needed.add(
                        sum(model.tank_start[j] for j in range(k))
                        + model.slack[k] / (target - cooled[k])
                        >= 1
                    )
                    break

            # Every target the tank misses unheated comes after a run, and so
            # with the run's overshoot (see required). In the solver's
            # relaxation a run a fraction 'on' owed only that fraction of it: a
            # tank heated 0.3 kWh short of any real plan's on real data.
            for k in range(1, num_steps):
                target = min(float(target_c[k]), t_ceiling[k])

                if target > t_floor[k] and cooled[k] < target:
                    model.ran_constraint.add(
                        model.ran[k] + model.slack[k] / (target - cooled[k]) >= 1
                    )

        # active_power_w[k, s] is the grid draw at step k if solar scenario s
        # comes true: max(0, electrical power - solar). One schedule is shared by
        # all scenarios (the plan cannot know which one will happen; replanning
        # every step corrects course once it does), and only the cost differs
        # between them.
        #
        # The heat pump draws alpha + beta * T (see _power_line_coefficients)
        # for the share u = q / heat_w of the step it runs, so a step draws
        # alpha * u + beta * T * u: beta * T_floor * u for the tank unheated,
        # and beta * (T - T_floor) * u for how much warmer it is.
        # heat_pump_excess_share, (T - T_floor) * u, is held between its
        # McCormick envelopes over the step's range, with T * on the exact
        # product heat_pump_temperature: exact for a full step (u = on) and an
        # idle one (u = 0), loose for a run's partly used first and last steps.
        #
        # What those envelopes cannot see is how much warmer than unheated the
        # tank is, and in the solver's relaxation a fraction of 'on' was
        # priced at the unheated tank: 2.6 against the 3.3 kWh of any real
        # plan, and 40 s to prove one. A balance of the tank itself pins it.
        # Its excess over the unheated trajectory, x = T - T_floor (K), grows
        # only with the heat put in - taps, ambient and the tank now are in
        # that trajectory - as x[k + 1] = a x[k] + b q[k]. So the heat times
        # the step's mean excess is
        #     q[k] (x[k] + b q[k] / 2a) = (x[k + 1]^2 - a^2 x[k]^2) / 2ab,
        # the discrete form of x q = d(C x^2 / 2)/dt + UA x^2: heat put into
        # a tank already warmer grows its excess's C x^2 / 2 and makes up what
        # the excess loses. Weighted by the line's slope per unit heat and
        # summed from now (x = 0) it telescopes into x^2 at weights that are
        # not negative - a convex function, held by its tangents - which the
        # steps' beta * (T - T_floor) * u must add up to at every step:
        # exact over the plan, only the split between a run's partly used
        # steps left to the envelopes. With it the relaxation prices heat at
        # the tank it heats: a real day's plan proven in 4 s instead of 44.
        model.heat_pump_excess_share = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.heat_pump_power_constraint = pyo.ConstraintList()

        model.S = pyo.RangeSet(0, len(solar_scenarios) - 1)
        model.active_power_w = pyo.Var(model.K, model.S, domain=pyo.NonNegativeReals)

        model.active_power_constraint = pyo.ConstraintList()

        power_lines = tank.power_lines

        # The heat pump's whole draw per step, sun or grid - see _build_objective.
        # Never below zero: the power line is a linear fit, and on inputs far
        # outside what it was fitted on it can dip below zero at a cold tank,
        # which no compressor does.
        model.draw_w = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        start_power_w = self.cop_model.start_step_power_w if self.cop_model else 0.0

        for k in range(num_steps):
            share = model.q_heat_pump_w[k] / heat_w[k]
            excess_share = model.heat_pump_excess_share[k]
            # (T - T_floor) * on, exact like heat_pump_temperature.
            excess_on = model.heat_pump_temperature[k] - t_floor[k] * model.boiler_on[k]
            range_k = t_ceiling[k] - t_floor[k]

            for bound in (
                excess_share >= excess_on - range_k * (model.boiler_on[k] - share),
                excess_share <= range_k * share,
                excess_share <= excess_on,
            ):
                model.heat_pump_power_constraint.add(bound)

        # The tangents of x^2 (EXCESS_TANGENT_SPACING_K apart) and, per step,
        # how far the steps so far book more than the balance above asks of
        # them: never less than nothing. In Wh, where its coefficients stay
        # near 1. The weights are beta / heat_w per step, held where they rise
        # faster than the excess decays (an evening cooling down) to a rate it
        # does, the convex bound's condition - a weaker bound there, never a
        # wrong one, and none where the line does not rise with T at all - a
        # flat fallback without a COP model. x is T - T_floor itself, not a
        # balance of its own: a second chain of dynamics beside T's, driven by
        # the same heat, had the solver's presolve cut off the optimal plan.
        if any(beta > 0.0 for _, beta in power_lines):
            model.excess_squared_k2 = pyo.Var(model.K, domain=pyo.NonNegativeReals)
            model.excess_surplus_wh = pyo.Var(model.K, domain=pyo.NonNegativeReals)
            model.excess_constraint = pyo.ConstraintList()

            # Nothing heated yet: no excess - and its square fixed too, which the
            # tangents only bound from below and the first step's balance takes
            # away, so a free one booked a surplus out of nothing.
            model.excess_squared_k2[0].fix(0.0)
            model.excess_surplus_wh[0].fix(0.0)
            weight_wh_per_k2 = math.inf

            for k in range(1, num_steps):
                for point_k in np.arange(
                    0.0,
                    t_ceiling[k] - t_floor[k] + EXCESS_TANGENT_SPACING_K,
                    EXCESS_TANGENT_SPACING_K,
                ):
                    model.excess_constraint.add(
                        model.excess_squared_k2[k]
                        >= 2.0 * point_k * (model.T[k] - t_floor[k]) - point_k**2
                    )

            for k in range(num_steps - 1):
                a_d, b_d = discretized(dt_hours[k])
                decay, rise_k_per_w = float(a_d[0, 0]), float(b_d[0, 1])
                weight_wh_per_k2 = min(
                    max(power_lines[k][1], 0.0)
                    / heat_w[k]
                    * dt_hours[k]
                    / (2.0 * decay * rise_k_per_w),
                    weight_wh_per_k2 / decay**2,
                )
                # The step's heat times its mean excess: the heat pump's as the
                # envelopes book it (heat_w * (T - T_floor) * u, its own rise in
                # full: q^2 <= heat_w * q), the booster's - which raises x too -
                # at the most it can be, the step's ceiling; against the
                # balance's x[k + 1]^2 - a^2 x[k]^2.
                booster_excess_k = t_ceiling[k] - t_floor[k]
                booster_excess_k += (
                    rise_k_per_w * (booster_heat_w or 0.0) / (2.0 * decay)
                )
                booked_wh = (
                    weight_wh_per_k2
                    * 2.0
                    * decay
                    * rise_k_per_w
                    * (
                        heat_w[k] * model.heat_pump_excess_share[k]
                        + rise_k_per_w
                        * heat_w[k]
                        / (2.0 * decay)
                        * model.q_heat_pump_w[k]
                        + booster_excess_k * model.q_booster_w[k]
                    )
                )
                model.excess_constraint.add(
                    model.excess_surplus_wh[k + 1]
                    == model.excess_surplus_wh[k]
                    + booked_wh
                    - weight_wh_per_k2
                    * (
                        model.excess_squared_k2[k + 1]
                        - decay**2 * model.excess_squared_k2[k]
                    )
                )

        for k in range(num_steps):
            alpha, beta = power_lines[k]
            share = model.q_heat_pump_w[k] / heat_w[k]
            heat_pump_power_w = alpha * share + beta * (
                t_floor[k] * share + model.heat_pump_excess_share[k]
            )

            # A run's first step draws its measured power (see
            # HeatPumpCOPModel.start_step_power_w) in place of the power line
            # over the heat it delivers, the ramp's share of the step: taken
            # as the power at the tank unheated, so its excess over that is
            # priced as in any step - exact for the horizon's first run, which
            # has none. A hand-over from the zone runs on the power line.
            ramp = self._ramp_fraction(dt_hours[k])
            unheated_w = alpha + beta * t_floor[k]

            if start_power_w:
                heat_pump_power_w += (start_power_w - ramp * unheated_w) * (
                    model.tank_start[k] - model.tank_handover[k]
                )

            # The booster draws more than it heats the tank (see
            # BoilerThermalModel.booster_electrical_w).
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
                heat_pump_power_w + self._booster_draw_per_heat * model.q_booster_w[k]
            )
            running = model.q_heat_pump_w[k] / heat_w[k]
            if booster_heat_w:
                running = running + model.q_booster_w[k] / booster_heat_w

            # Space heating draws through the same compressor. Its efficiency
            # is a per-step constant rather than a line in the zone's
            # temperature: the supply temperature a floor circuit runs at is
            # set by the weather compensation curve, not by the room, so there
            # is no tank-like relationship to follow here - and fitting one
            # would need the heating COP model a heating season has to produce.
            #
            # Cooling moves heat the other way, so its magnitude is what draws.
            if hasattr(model, "q_space_w"):
                moved_w = model.space_sign * model.q_space_w[k]
                electrical_w = electrical_w + (
                    model.space_power_w[k]
                    if hasattr(model, "space_power_w")
                    else moved_w / model.space_cop[k]
                )
                running = running + moved_w / self._heat_w

            model.active_power_constraint.add(model.draw_w[k] >= electrical_w)

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
        # surplus sun is not stored for demand beyond it.
        import_price, export_price = self.prices(data)
        model.objective = pyo.Objective(
            expr=self._build_objective(
                model,
                dt_hours,
                [weight for weight, _ in solar_scenarios],
                import_price,
                export_price,
            ),
            sense=pyo.minimize,
        )

        model.dt_hours = dt_hours
        model.t_floor = t_floor
        model.mpc_scenario_weights = [weight for weight, _ in solar_scenarios]

        return model

    @property
    def _booster_draw_per_heat(self) -> float:
        """Electrical W drawn per W of booster heat: its identified draw over
        its heat (see BoilerThermalModel.booster_electrical_w), 1 without it."""

        model = self.thermal_model

        if not model.booster_electrical_w or not model.booster_heat_w:
            return 1.0

        return model.booster_electrical_w / model.booster_heat_w

    @property
    def _heat_w(self) -> float:
        """The heat the compressor delivers once it is up to speed, as its
        calorimetry measures it (W; the tank takes up _heat_at). The
        calorimetric measurement where runs have shown it, the ODE's own fitted
        constant otherwise - that one is an average over whole runs, including
        their ramp, so it understates a running compressor.
        """

        return self.thermal_model.q_in_steady_w or self.thermal_model.q_in_nominal_w

    def _heat_at(self, outdoor_c: float | None) -> float:
        """The heat a running compressor puts into the tank at this outdoor
        temperature (W): the calorimetric line of the COP model (see
        HeatPumpCOPModel.q_in_at_zero_outdoor_w), or _heat_w without it or
        without a forecast - as the tank takes it up (see
        BoilerThermalModel.calorimetric_gain)."""

        cop = self.cop_model
        gain = self.thermal_model.calorimetric_gain or 1.0

        if cop is None or outdoor_c is None or not cop.q_in_at_zero_outdoor_w:
            return self._heat_w * gain

        return (
            cop.q_in_at_zero_outdoor_w + cop.q_in_per_outdoor_w_per_k * outdoor_c
        ) * gain

    def _initial_temperature(self, data: MPCInput) -> float:
        """The tank the plan starts from now (deg C) - its first step runs from
        now (see MPCInput.first_step_hours): mixed from its sensors (see
        BoilerThermalModel.mixed_temperature), or while a DHW run is under way
        the tank mixed just before it began plus the heat the run has put in
        since, less the standing loss - as its calorimetry measured it (see
        MPCInput.run_heat_j), or without that the same ramp and heat input the
        plan runs on. The sensors lag that heat while the tank is heating (real
        data, 28 Sep: 42.0/41.5 degC half an hour into a run, 41.0 mixed,
        against 44 degC by the run's own heat balance), and a plan started
        from them saw the run go on a quarter hour longer than it did. Taps
        during the run are not known and left out. At most the heat pump's
        own limit: beyond it the booster, not this estimate, takes over."""

        heated_c = self.run_temperature(
            data, data.compressor_elapsed_hours, data.run_heat_j
        )

        if heated_c is not None:
            return heated_c

        return self.thermal_model.mixed_temperature(
            data.current_temp_top,
            data.current_temp_bottom,
            data.ambient_temperature,
            data.current_stratification_k,
        )

    def run_temperature(
        self, data: MPCInput, elapsed_hours: float, heat_j: float | None = None
    ) -> float | None:
        """The tank elapsed_hours into the DHW run under way (deg C) by the
        run's heat balance (see _initial_temperature), None without such a
        run or the tank before it. heat_j is the heat the run's calorimetry
        measured going in by then (J); None takes the plan's own ramp and heat
        input. The plan's ran 2-4 K warmer than the measured heat over a run
        whose compressor came up slower than its ramp (21 Oct 2026)."""

        if (
            not data.boiler_on_current
            or data.run_start_temp_top is None
            or data.run_start_temp_bottom is None
            or data.compressor_elapsed_hours <= 0.0
        ):
            return None

        start_c = self.thermal_model.mixed_temperature(
            data.run_start_temp_top,
            data.run_start_temp_bottom,
            data.run_start_ambient_c
            if data.run_start_ambient_c is not None
            else data.ambient_temperature,
            data.run_start_stratification_k,
        )
        outdoor_c = (
            data.outdoor_temperature_forecast[0]
            if data.outdoor_temperature_forecast
            else None
        )
        # The mean heat over the time run so far, its ramp included.
        if heat_j is None:
            heat_w = self._heat_at(outdoor_c) * self._ramp_fraction(elapsed_hours)
        else:
            heat_w = (
                heat_j
                * (self.thermal_model.calorimetric_gain or 1.0)
                / max(elapsed_hours * 3600.0, 1.0)
            )
        a_d, b_d = discretize_zoh(
            *lumped_tank_state_space(
                self.thermal_model.volume_l,
                self.thermal_model.ua_top_w_per_k
                + self.thermal_model.ua_bottom_w_per_k,
            ),
            elapsed_hours * 3600.0,
        )
        estimate_c = float(
            a_d[0, 0] * start_c
            + b_d[0, 0] * float(data.ambient_temperature)
            + b_d[0, 1] * heat_w
        )
        limit_c = self.thermal_model.heat_pump_max_tank_temperature_c

        return min(estimate_c, limit_c) if limit_c is not None else estimate_c

    def _ramp_fraction(self, dt_hours: float) -> float:
        """Share of _heat_w a step delivers when the compressor starts in it.

        A compressor rising linearly to full output over its ramp delivers half
        of it while still rising, so a step shorter than the ramp carries
        dt / 2T of it and a longer one all but T / 2T of its own length. 1
        without an identified ramp, which is the constant-output model this
        replaced.
        """

        # Timed from the run's own start where the COP model has it (see
        # HeatPumpCOPModel.start_ramp_seconds).
        return self._ramp_share(
            dt_hours,
            (self.cop_model and self.cop_model.start_ramp_seconds)
            or self.thermal_model.q_in_ramp_seconds,
        )

    @staticmethod
    def _ramp_share(dt_hours: float, ramp_seconds: float | None) -> float:
        """See _ramp_fraction: the share of a step's heat a compressor
        starting in it delivers, over a ramp of ramp_seconds."""

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
        temperature is the plate exchanger's approach measured on the boiler's own runs
        (see _supply_margin_c). POWER_FIT_T_LOW_C/HIGH_C are
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

        margin = self._supply_margin_c(overall_target_max) + self._half_step_rise_k

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

    @property
    def _half_step_rise_k(self) -> float:
        """Half the rise of the tank over a step the heat pump runs through
        (K). T[k] is the tank at the start of the step, but the supply follows
        the tank as it warms, so the step draws as at its middle - the point
        the supply margin is measured against (see
        BoilerThermalIdentifier._identify_supply_margin). 0 without a measured
        margin: the fallback margin is not measured against the tank's
        middle."""

        if self.thermal_model.supply_margin_k is None:
            return 0.0

        return (
            self._heat_w
            * (self.thermal_model.calorimetric_gain or 1.0)
            * self.config.step_hours
            * 3600.0
            / (
                2.0
                * RHO_WATER_KG_PER_L
                * self.thermal_model.volume_l
                * CP_WATER_J_PER_KG_K
            )
        )

    def _supply_margin_c(self, overall_target_max: float) -> float:
        """How much hotter the heat pump's supply runs than the tank it charges
        (see _power_line_coefficients); 0 without a calibrated COP model.

        The plate exchanger's measured approach (BoilerThermalModel.supply_margin_k)
        where calibrated: a property of the exchanger, whatever the targets are. A model
        calibrated before it existed falls back to the reference supply's margin
        above the highest target until its next calibration - which a target
        near that supply, such as a legionella run, drove to 0, costing every
        run in the horizon as if the supply were no hotter than the tank.
        """

        if self.cop_model is None:
            return 0.0

        if self.thermal_model.supply_margin_k is not None:
            return self.thermal_model.supply_margin_k

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

    def _space_cop(self, outdoor_c: float | None, cooling: bool = False) -> float:
        """Coefficient of performance for the zone at one step: the heat moved
        per unit of electricity, taken from the water while cooling."""

        cop_model = self.space_cop_model or (None if cooling else self.cop_model)

        if cop_model is None or outdoor_c is None:
            # Same fallback the tank uses: a flat electrical assumption over
            # its nominal heat output.
            return self._heat_w / max(self.config.boiler_electrical_power_w, 1.0)

        # At the supply the heat pump's own heating curve runs at, once known.
        supply_c = (
            self.floor_circuit_model.supply_c(outdoor_c)
            if self.floor_circuit_model is not None
            else self.SPACE_HEATING_SUPPLY_C
        )

        return float(cop_model.clamped_cop(outdoor_c, supply_c))

    def _add_space_heating(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
        dt_hours: list[float],
        outdoor_c: Sequence[float] | None,
    ) -> None:
        """Adds the zone as a second demand on the same compressor, heated or,
        with data.zone_cooling, cooled: the same floor and the same balance,
        with the heat into it negative.

        Without a building model, or without a zone temperature to start from,
        space_on stays fixed at zero and the model is exactly the
        domestic-hot-water one it was before. That is deliberate: the zone's
        dynamics are the one thing here that cannot be guessed at, and planning
        floor runs against a model that cannot predict their effect would be
        worse than not planning them at all (see features/building.py for the
        metrics that decide when it can).
        """

        num_steps = len(dt_hours)
        building = self.building_model

        # The unmeasured slabs are as much a required initial condition as the
        # measured rooms: a plan started with a cold screed is a different plan
        # from one started with a charged one. Missing them means the filter
        # has not run, which is not something to paper over with a guess - the
        # zone is simply not planned, exactly as when no model has been
        # calibrated.
        if not self._zone_planned(data):
            for k in range(num_steps):
                model.space_on[k].fix(0)

            return

        target_c = data.zone_target_temperature
        minimum_c = data.zone_minimum_temperature or None
        maximum_c = data.zone_maximum_temperature or None
        none_w = (0.0,) * num_steps
        internal_gain_w = data.zone_internal_gain_w or none_w
        solar_gain_w = data.zone_solar_gain_w or none_w
        solar_gain_rest_w = data.zone_solar_gain_rest_w or none_w

        # The same compressor serves both, so its output is the tank's nominal
        # heat - but the zone no more than its runs have moved (see
        # FloorCircuitModel.max_heat_w), the range its COP model was fitted in.
        # Beyond it that model is extrapolated, and wrongly: at the tank's
        # 6.5 kW and a 29 degC supply it gave a COP of 5.9 where the tank's
        # runs at that output and supply measured 4.0-4.2 (real data, Oct
        # 2026; the floor's runs at 3.4-4.4 kW were within 10% of it), so the
        # plan ran the floor flat out at a high supply on a cheap-looking COP.
        operating_range = self.floor_circuit_model
        max_heat_w = (
            min(self._heat_w, operating_range.max_heat_w)
            if operating_range is not None and operating_range.max_heat_w
            else self._heat_w
        )
        cooling = bool(data.zone_cooling)
        # The sign of the heat the zone's mode brings into the floor.
        sign = -1.0 if cooling else 1.0
        model.space_sign = sign

        a, b = zone_state_space(building)
        observation = zone_observation(building)
        num_states = a.shape[0]
        initial = np.array([float(t) for t in data.zone_state])

        if len(initial) != num_states:
            raise ValueError(
                f"The zone's state has {len(initial)} values, the building "
                f"model {num_states}: estimate it with this model."
            )

        # The rooms - air, furnishings, internal walls - and the slabs the
        # floor's heat lands in (see BuildingThermalModel).
        model.ZONE_STATES = pyo.RangeSet(0, num_states - 1)
        model.zone_state = pyo.Var(model.K, model.ZONE_STATES)
        model.q_space_w = pyo.Var(
            model.K, bounds=(-max_heat_w, 0.0) if cooling else (0.0, max_heat_w)
        )
        model.zone_slack = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.zone_excess = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.zone_discomfort_k = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.zone_discomfort_k2 = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        model.zone_constraints = pyo.ConstraintList()

        for i in model.ZONE_STATES:
            model.zone_constraints.add(model.zone_state[0, i] == float(initial[i]))

        # Evaluated at a representative floor-circuit supply temperature and
        # the outdoor temperature of each step, so a cold day costs what a cold
        # day costs. Falls back to the flat assumption while no COP model has
        # been calibrated, exactly as the tank does.
        model.space_cop = [
            self._space_cop(float(outdoor_c[k]) if outdoor_c else None, cooling)
            for k in range(num_steps)
        ]

        # A run the zone opens starts the compressor, and the compressor does
        # not move its heat at once: as for the tank (see _ramp_fraction), the
        # step it starts in delivers its heat less the ramp's shortfall (real
        # cooling data, 30 runs: 68% over the first quarter hour, a 530 s
        # linear ramp's 71%). q_space_w stays the heat the heat pump runs at -
        # what its supply, its least heat and its draw follow - so the start
        # step is priced at running power: the ramp costs heat, hardly power
        # (real data: 79% of running power over that quarter). A run the tank
        # hands over carries no ramp: the compressor is already at speed.
        ramp_seconds = (
            self.space_cop_model.start_ramp_seconds
            if self.space_cop_model is not None
            and self.space_cop_model.cooling == cooling
            else 0.0
        )

        def floor_w(k: int):
            """The heat the floor takes in step k (W)."""

            if not ramp_seconds:
                return model.q_space_w[k]

            share = self._ramp_share(dt_hours[k], ramp_seconds)

            return model.q_space_w[k] - sign * (1.0 - share) * model.space_ramp_w[k]

        if ramp_seconds:
            # |Q| * (the zone starts the compressor), exact: the start is 0 or 1
            # wherever the binaries are, and the heat bounded by the compressor.
            model.space_ramp_w = pyo.Var(model.K, bounds=(0.0, max_heat_w))

            for k in range(num_steps):
                zone_start = (
                    model.compressor_start[k]
                    - model.tank_start[k]
                    + model.tank_handover[k]
                )
                moved_w = sign * model.q_space_w[k]

                for bound in (
                    model.space_ramp_w[k] <= max_heat_w * zone_start,
                    model.space_ramp_w[k] <= moved_w,
                    model.space_ramp_w[k] >= moved_w - max_heat_w * (1 - zone_start),
                ):
                    model.zone_constraints.add(bound)

        discretized: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        zone_outdoor_c = [
            float(outdoor_c[k]) if outdoor_c else float(data.ambient_temperature)
            for k in range(num_steps)
        ]

        # The three-way valve hands a heating run over either way: from the
        # tank to the zone - the loop is then hotter than the floor needs, and
        # that heat flows on into it - and from the zone to the tank, the loop
        # first brought up from the floor's supply to above the tank's (see
        # tank_handover). Either keeps the compressor running, where a stop in
        # between pays a start and the same loop to bring up after it. That
        # the tank is heated in one block per target, and so not twice in a
        # run, is the tank's own plan's (see plan_tank).
        #
        # Cooling rules out both. The heat pump reverses its refrigerant cycle
        # between cooling and the tank, and does so with the compressor
        # stopped (real data: all six hand-overs from cooling to hot water
        # stood at 0 Hz for about three minutes), so neither shares a run with
        # the other. Planned a step apart, which the real three-minute stop
        # fits within.
        if cooling:
            model.zone_constraints.add(
                model.boiler_on[0] <= 1 - int(data.space_on_current)
            )
            model.zone_constraints.add(
                model.space_on[0] <= 1 - int(data.boiler_on_current)
            )

            for k in range(1, num_steps):
                model.zone_constraints.add(
                    model.boiler_on[k] <= 1 - model.space_on[k - 1]
                )
                model.zone_constraints.add(
                    model.space_on[k] <= 1 - model.boiler_on[k - 1]
                )

        # A day's cooling comes before its hot water: a hot water run warms
        # the attic the tank stands in, and cooling after it gives more chance
        # of condensation - so once the tank has been heated on a day, that
        # day is not cooled again (booster included: it warms the attic too).
        # A latch per day, reset at local midnight: no more than "the tank has
        # been heated today", which the cooling bound below keeps at its least.
        if cooling and data.zone_local_day:
            day = data.zone_local_day
            model.dhw_today = pyo.Var(model.K, bounds=(0.0, 1.0))

            for k in range(num_steps):
                model.zone_constraints.add(model.dhw_today[k] >= model.tank_heating[k])

                if k == 0:
                    earlier = int(data.dhw_earlier_today)
                elif day[k] == day[k - 1]:
                    earlier = model.dhw_today[k - 1]
                else:
                    continue

                model.zone_constraints.add(model.dhw_today[k] >= earlier)
                model.zone_constraints.add(model.space_on[k] <= 1 - earlier)

        for k in range(num_steps):
            # Heat only flows to the zone while the valve points at it.
            model.zone_constraints.add(
                sign * model.q_space_w[k] <= max_heat_w * model.space_on[k]
            )

            # One compressor, one demand at a time. Sequential within a run is
            # what the hardware does; simultaneous is not. The booster counts
            # too: it finishes the tank with the heat pump in its hot water
            # mode (see app.state), which serves nothing else meanwhile -
            # counted apart, a plan heated the floor beside it (0.9 kW on top
            # of the booster's 2.2).
            model.zone_constraints.add(model.tank_heating[k] + model.space_on[k] <= 1)

            if k + 1 >= num_steps:
                continue

            hours = dt_hours[k]

            if hours not in discretized:
                discretized[hours] = discretize_zoh(a, b, hours * 3600.0)

            a_d, b_d = discretized[hours]
            outdoor = zone_outdoor_c[k]

            # u = [T_outdoor, Q_internal, Q_solar_living, Q_solar_rest,
            # Q_floor], the same input vector the zone is identified with. Each
            # state advances by its own row of the discrete matrices, so where
            # a gain lands - which room, or the slabs - is the model's to
            # decide, not this block's.
            for i in model.ZONE_STATES:
                model.zone_constraints.add(
                    model.zone_state[k + 1, i]
                    == sum(
                        a_d[i, j] * model.zone_state[k, j] for j in model.ZONE_STATES
                    )
                    + b_d[i, 0] * outdoor
                    + b_d[i, 1] * float(internal_gain_w[k])
                    + b_d[i, 2] * float(solar_gain_w[k])
                    + b_d[i, 3] * float(solar_gain_rest_w[k])
                    + b_d[i, 4] * floor_w(k)
                )

        # What a thermostat reads of the plan: an operative temperature, part
        # air and part mass (see zone_observation). Comfort is
        # judged on it because that is the quantity the setpoint is set in and
        # the occupant feels, and because it is what the model was identified
        # against.
        def measured(k):
            return sum(
                float(observation[i]) * model.zone_state[k, i]
                for i in model.ZONE_STATES
            )

        model.zone_measured = measured

        # What a thermostat would read with the zone heated (or cooled) flat out
        # every step, and with it left alone. A target above the warmer of the
        # two cannot be met by any plan, and sun alone may carry the zone past
        # a ceiling below the cooler; those parts are the same whatever is
        # decided, so they are left out and the slacks measure only what
        # planning can still change - as for the tank's target. Priced at
        # weight_temperature_slack, an unavoidable shortfall of a few
        # hundredths of a degree had the solver proving plans to a millionth
        # of one: over a minute for a plan it had found in seconds.
        def states_with(heat_w: float) -> list[np.ndarray]:
            state = initial
            states = [state]

            for k in range(num_steps - 1):
                a_d, b_d = discretized[dt_hours[k]]
                inputs = np.array(
                    [
                        zone_outdoor_c[k],
                        float(internal_gain_w[k]),
                        float(solar_gain_w[k]),
                        float(solar_gain_rest_w[k]),
                        heat_w,
                    ]
                )
                state = a_d @ state + b_d @ inputs
                states.append(state)

            return states

        # Every state lies between these two for any plan: the zone's response
        # is monotone in the heat, which runs from 0 to flat out either way.
        flat_out = states_with(sign * max_heat_w)
        unheated = states_with(0.0)
        warmest = [np.maximum(a, b) for a, b in zip(flat_out, unheated, strict=True)]
        coolest = [np.minimum(a, b) for a, b in zip(flat_out, unheated, strict=True)]
        reachable_c = [float(observation @ state) for state in warmest]
        unheated_c = [float(observation @ state) for state in coolest]

        # Comfort at both ends of a step, for the same reason the tank's target
        # is checked at both: T[k] is the value at a step's START.
        for k in range(num_steps):
            ends = (k, k + 1) if k + 1 < num_steps else (k,)
            span_k = 0.0

            # How far the zone is from its wanted temperature towards
            # discomfort, a target it cannot reach taken as far as it can.
            for j in ends:
                if cooling:
                    wanted_c = max(float(target_c[k]), unheated_c[j])
                    away = measured(j) - wanted_c
                    edge_c = float(maximum_c[k]) if maximum_c else reachable_c[j]
                    span_k = max(span_k, min(edge_c, reachable_c[j]) - wanted_c)
                else:
                    wanted_c = min(float(target_c[k]), reachable_c[j])
                    away = wanted_c - measured(j)
                    edge_c = float(minimum_c[k]) if minimum_c else unheated_c[j]
                    span_k = max(span_k, wanted_c - max(edge_c, unheated_c[j]))

                model.zone_constraints.add(model.zone_discomfort_k[k] >= away)

            # Its square, bounded from below by tangents as far as the band
            # reaches: beyond its edge the slacks below take over.
            for point_k in np.arange(
                COMFORT_TANGENT_SPACING_K,
                span_k + COMFORT_TANGENT_SPACING_K,
                COMFORT_TANGENT_SPACING_K,
            ):
                model.zone_constraints.add(
                    model.zone_discomfort_k2[k]
                    >= 2.0 * point_k * model.zone_discomfort_k[k] - point_k**2
                )

            # The band's edges, soft only so that a plan must still exist where
            # sun alone pushes the zone past the ceiling, or a cold night below
            # the floor: only what the plan can change is its to answer for,
            # priced as dearly as the tank's target.
            for j in ends:
                if minimum_c is not None:
                    model.zone_constraints.add(
                        measured(j) + model.zone_slack[k]
                        >= min(float(minimum_c[k]), reachable_c[j])
                    )

                if maximum_c is not None:
                    model.zone_constraints.add(
                        measured(j) - model.zone_excess[k]
                        <= max(float(maximum_c[k]), unheated_c[j])
                    )

        # A zone that leaves its comfort band by step k when left alone needs
        # the compressor before then: a start, unless it runs now - the zone's
        # counterpart of the tank's run_needed. Every real plan meets this
        # already; in the solver's relaxation a third of 'on' held for 14 hours
        # otherwise heated the zone below the compressor's least output, at a
        # third of one start against the two the plan makes.
        if not (data.boiler_on_current or data.space_on_current):
            for k in range(1, num_steps):
                left_c = float(observation @ unheated[k])

                if (maximum_c if cooling else minimum_c) is None:
                    break

                if cooling:
                    missed_c = left_c - max(float(maximum_c[k]), unheated_c[k])
                    slack = model.zone_excess[k]
                else:
                    missed_c = min(float(minimum_c[k]), reachable_c[k]) - left_c
                    slack = model.zone_slack[k]

                if missed_c > 0.0:
                    model.zone_constraints.add(
                        sum(model.compressor_start[j] for j in range(k))
                        + slack / missed_c
                        >= 1
                    )
                    break

        # Water condenses on a floor below the dew point of the air above it.
        # The floor's surface lies between a slab and its room, so while
        # cooling no slab may be cooled below the dew point (plus a margin, see
        # BuildingConfig.dew_point_margin). Hard, unlike comfort: a wet floor
        # is not traded against a warm room. Always feasible, since cooling
        # only lowers the mass - where the zone alone would sit below the
        # bound, that is the bound.
        if cooling and data.zone_mass_minimum_c:
            mass_minimum_c = data.zone_mass_minimum_c

            # The screeds too, where the floor has them: its surface.
            layers = ZONE_SLAB_STATES + (ZONE_SCREED_STATES if num_states > 4 else ())

            for k in range(num_steps):
                for i in layers:
                    model.zone_constraints.add(
                        model.zone_state[k, i]
                        >= min(mass_minimum_c[k], float(unheated[k][i]))
                    )

        # The temperature the floor's water delivers against: the screeds it
        # heats first where the floor has them, else the slabs' mean, one
        # circuit running through both (see zone_floor).
        against = zone_floor(building)

        def sink_c(state: np.ndarray) -> float:
            return float(against @ state)

        def sink(k: int):
            return sum(
                float(against[i]) * model.zone_state[k, i] for i in model.ZONE_STATES
            )

        # How the heat pump runs the floor by itself, once heating runs have
        # shown it (see features.floor). On its curve it picks its supply
        # temperature itself, so the heat a run delivers is not the plan's to
        # choose: Q = G * (supply - T_mass), with T_mass the node the floor's
        # heat lands in. The plan decides when; the physics decides how much -
        # and a quarter hour at full power into a floor at room temperature is
        # no longer on offer. Needs an outdoor forecast, which the curve is a
        # function of.
        operating = self.floor_circuit_model if outdoor_c else None

        if operating is None:
            return

        # The product T_sink * space_on, exact for a binary decision within the
        # sink's range, which the unheated and flat-out rollouts bound - the
        # slabs' mean being monotone in the heat like each slab.
        model.space_sink_on = pyo.Var(model.K)
        conductance = operating.conductance_w_per_k
        # The mode's own COP (EER) model, where runs have shown one. The heat
        # pump holds the supply it is given (its flow mode), so the plan
        # chooses it - heating only with that model: on a flat COP a warmer
        # supply would cost the plan nothing, while it costs the heat pump
        # lift. Without it heating keeps to the curve the heating runs showed,
        # and cooling is priced at the flat per-step COP above.
        efficiency = (
            self.space_cop_model
            if self.space_cop_model is not None
            and self.space_cop_model.cooling == cooling
            else None
        )
        # Whether the supply is the plan's to choose rather than the curve's.
        chosen = cooling or efficiency is not None

        if efficiency is not None:
            model.space_power_w = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        def draw_w(moved_w: float, outdoor: float, sink_c: float) -> float:
            """The electricity (W) to move this heat into (or, cooling, out of)
            a mass at sink_c, at the supply that takes: sink_c +- Q / G."""

            supply_c = sink_c + sign * moved_w / conductance

            return moved_w / float(efficiency.clamped_cop(outdoor, supply_c))

        supply_minimum_c = data.zone_supply_minimum_c or None
        model.space_supply_c = []

        for k in range(num_steps):
            low = sink_c(coolest[k])
            high = sink_c(warmest[k])
            on = model.space_on[k]
            sink_on = model.space_sink_on[k]

            for bound in (
                sink_on <= high * on,
                sink_on >= low * on,
                sink_on <= sink(k) - low * (1 - on),
                sink_on >= sink(k) - high * (1 - on),
            ):
                model.zone_constraints.add(bound)

            if chosen:
                # The supply is the plan's to choose - the setpoint is given to
                # the heat pump - so the heat is too, and the supply follows
                # from it: T_supply = T_sink + Q / G.
                model.space_supply_c.append(sink(k) + model.q_space_w[k] / conductance)

                # No less than the compressor at its lowest speed: asked for
                # less it overshoots the setpoint rather than turn down.
                model.zone_constraints.add(
                    sign * model.q_space_w[k] >= operating.min_heat_w * on
                )

                # What that heat draws: the heat over the COP (EER) at the
                # supply it takes, T_sink +- Q / G - so more heat moved means a
                # higher lift and a lower COP (real cooling data: an EER of 5.3
                # at the least cooling, 5.0 at 4.5 kW, from a 21 degC floor on
                # a 25 degC day). Convex in the heat for a given mass, heating
                # as cooling: Q / COP grows as Q times a lift term that grows
                # with Q, its curvature only turning at a lift of the absolute
                # condensing temperature. So the tangents at a few points bound
                # it from below and the plan, paying for it, sits on the curve.
                # The mass the heat goes into enters to first order around the
                # unheated one, on the exact product T_sink * on (so nothing
                # while off): across mass and heat together the draw is not
                # convex, and priced against the unheated mass alone it read
                # the cooling supply too warm by the run's own cooling of the
                # mass - some 3.5% per kelvin, 5.6% on a hot day's run - where
                # planning twice, the second time against the mass the first
                # planned, doubled the solve.
                if efficiency is not None:
                    reference_c = sink_c(unheated[k])

                    for point_w in np.linspace(0.0, max_heat_w, 7):
                        at_w = draw_w(point_w, zone_outdoor_c[k], reference_c)
                        per_w = (
                            draw_w(point_w + 1.0, zone_outdoor_c[k], reference_c) - at_w
                        )
                        per_k = (
                            draw_w(point_w, zone_outdoor_c[k], reference_c + 0.1) - at_w
                        ) / 0.1
                        # Each tangent in its perspective, scaled by on: the
                        # same bound on or off, but a tighter relaxation for a
                        # step partly on (a heating day's solve 8.3 -> 6.1 s).
                        model.zone_constraints.add(
                            model.space_power_w[k]
                            >= at_w * on
                            + per_w * (sign * model.q_space_w[k] - point_w * on)
                            + per_k * (sink_on - reference_c * on)
                        )

                # A supply no colder than uninsulated pipes allow, so no more
                # than the floor takes from water at that temperature. Together
                # with the bound above, a humid day may leave no supply that
                # does both: the zone is then not cooled.
                if cooling and supply_minimum_c is not None:
                    model.zone_constraints.add(
                        -model.q_space_w[k]
                        <= conductance * (sink_on - supply_minimum_c[k] * on)
                    )

                continue

            model.space_supply_c.append(operating.supply_c(zone_outdoor_c[k]))
            floor_w = conductance * (
                operating.supply_c(zone_outdoor_c[k]) * on - sink_on
            )
            model.zone_constraints.add(model.q_space_w[k] <= floor_w)

            # Held to exactly that wherever it is within the compressor's own
            # output. Where it may not be - a curve extrapolated to a colder day
            # than its runs covered - the heat pump gives what it can, and the
            # floor's uptake only caps it.
            supply = operating.supply_c(zone_outdoor_c[k])
            if conductance * (supply - low) <= max_heat_w:
                model.zone_constraints.add(model.q_space_w[k] >= floor_w)

        # Within a cooling run the supply only falls, within a heating run it
        # only rises. Set past the water already in the loop - warmer while
        # cooling, colder while heating - the setpoint asks for less than the
        # compressor gives at its lowest speed: it stops, and starts again later
        # - a start the plan would not see (real cooling data; the Ecodan
        # firmware keeps a heating setpoint within 1 K under the supply for the
        # same reason). The other way is always possible, and a cooling run
        # whose dew point bound rises past it has to stop.
        #
        # Stated on supply * on = sink * on + Q / G, exact through the product
        # above and 0 while off: a step that stops is then free, and one after
        # a stop free. Stated on the supply itself, with a big-M on both steps,
        # the solver ran into its time limit on a single day's plan.
        #
        # With each step's supply while on between lo and hi (the sink's range
        # and the compressor's least and most heat), the bound for a pair of
        # steps is the convex hull of its four on/off cases, heating
        #   s[k+1] >= s[k] - (hi[k] - lo[k+1]) - lo[k+1] on[k] + hi[k] on[k+1],
        # tight in each case: s[k+1] >= s[k] when both run, nothing beyond lo
        # and hi otherwise - cooling its mirror. A single big-M from the warmest
        # supply bounded the same integer plans more loosely: a heating day's
        # solve 6.1 -> 5.2 s.
        if chosen:
            least_c = operating.min_heat_w / conductance
            most_c = max_heat_w / conductance

            def supply_range(k: int) -> tuple[float, float]:
                low_c, high_c = sink_c(coolest[k]), sink_c(warmest[k])

                if cooling:
                    return low_c - most_c, high_c - least_c

                return low_c + least_c, high_c + most_c

            def supply_on(k: int):
                return model.space_sink_on[k] + model.q_space_w[k] / conductance

            for k in range(num_steps - 1):
                lo_k, hi_k = supply_range(k)
                lo_next, hi_next = supply_range(k + 1)
                on_k, on_next = model.space_on[k], model.space_on[k + 1]

                # Nor does a run's heat rise: held at its setpoint the floor
                # takes less as its screed warms, and a setpoint raised within
                # the run holds the heat, not more of it (real data, 11 Oct
                # 2026: 4.6 kW falling to 3.0 kW at 24.8-25.5 degC). Per step
                # the plan sat at the compressor's least heat and ended a run
                # 50% higher - within a tenth of a cent of a steady run, the
                # solver's own tolerance; on eight days this held runs steady
                # for at most 0.23 cent, every solve as quick. Exact in the
                # binaries' four cases: no bound beyond the most heat unless
                # both steps run.
                model.zone_constraints.add(
                    sign * model.q_space_w[k + 1]
                    <= sign * model.q_space_w[k] + max_heat_w * (2 - on_k - on_next)
                )

                # Where the ranges do not overlap the order holds by itself.
                if cooling and hi_next >= lo_k:
                    model.zone_constraints.add(
                        supply_on(k + 1)
                        <= supply_on(k)
                        + (hi_next - lo_k)
                        - hi_next * on_k
                        + lo_k * on_next
                    )
                elif not cooling and hi_k >= lo_next:
                    model.zone_constraints.add(
                        supply_on(k + 1)
                        >= supply_on(k)
                        - (hi_k - lo_next)
                        - lo_next * on_k
                        + hi_k * on_next
                    )

    def _build_objective(
        self,
        model: pyo.ConcreteModel,
        dt_hours: list[float],
        scenario_weights: list[float],
        import_price_eur_per_kwh: Sequence[float],
        export_price_eur_per_kwh: Sequence[float],
    ):
        objective = 0.0

        for k in model.K:
            expected_grid_power_w = sum(
                weight * model.active_power_w[k, s]
                for s, weight in enumerate(scenario_weights)
            )
            grid_energy_kwh = expected_grid_power_w * dt_hours[k] / 1000.0
            energy_kwh = model.draw_w[k] * dt_hours[k] / 1000.0

            # What the house pays for the heat pump against not running it at
            # all: the grid energy at the price, and the own solar it uses at
            # the export that solar would otherwise have earned.
            export = export_price_eur_per_kwh[k]
            objective += (
                import_price_eur_per_kwh[k] - export
            ) * grid_energy_kwh + export * energy_kwh

            objective += self.config.weight_switching * model.compressor_start[k]

            objective += self.config.weight_temperature_slack * model.slack[k]

            # The zone's band is weighed the same as the tank's target: both
            # are promises the plan made, and neither is worth trading for a
            # few cents of electricity. Within it, comfort is traded.
            if hasattr(model, "zone_slack"):
                objective += self.config.weight_temperature_slack * (
                    model.zone_slack[k] + model.zone_excess[k]
                )
                objective += (
                    self.config.weight_comfort_eur_per_k2h
                    * dt_hours[k]
                    * model.zone_discomfort_k2[k]
                )

        return objective

    def _extract_result(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
        termination_condition: TerminationCondition,
    ) -> MPCResult:
        num_steps = len(data.solar_forecast_w)
        dt_hours: list[float] = model.dt_hours

        def values(variable) -> list[float]:
            return [float(pyo.value(variable[k])) for k in range(num_steps)]

        objective_value = float(pyo.value(model.objective))
        # "On" whenever the boiler is heated, by either source.
        schedule = tuple(round(on) for on in values(model.tank_heating))
        temperatures = tuple(values(model.T))
        heat_pump_w = values(model.q_heat_pump_w)
        booster_w = values(model.q_booster_w)
        tank_start = [round(start) for start in values(model.tank_start)]
        handover = [round(on) for on in values(model.tank_handover)]

        # Same formula the objective itself is built from (see
        # _power_line_coefficients), evaluated at each step's own outdoor
        # forecast and planned tank temperature - the reported curve reflects
        # exactly what the optimizer costed with, not a separate display-only
        # estimate.
        overall_target_max = max(data.target_temperature_top)
        start_power_w = self.cop_model.start_step_power_w if self.cop_model else 0.0
        electrical_power_w = []

        for k in range(num_steps):
            T_outdoor = (
                data.outdoor_temperature_forecast[k]
                if data.outdoor_temperature_forecast
                else None
            )
            alpha, beta = self._power_line_coefficients(T_outdoor, overall_target_max)
            # For the part of the step the heat pump runs (see active_power_w).
            used = heat_pump_w[k] / self._heat_at(T_outdoor)
            heat_pump_power_w = (
                start_power_w + beta * (temperatures[k] - model.t_floor[k]) * used
                if tank_start[k] and not handover[k] and used and start_power_w
                else (alpha + beta * temperatures[k]) * used
            )

            electrical_power_w.append(
                heat_pump_power_w + self._booster_draw_per_heat * booster_w[k]
            )

        # Empty when no space heating was planned, so a caller can tell "the
        # zone was left alone" from "the zone was planned to coast".
        space_schedule: tuple[int, ...] = ()
        space_heat_w: tuple[float, ...] = ()
        zone_temperatures: tuple[float, ...] = ()
        zone_mass_temperatures: tuple[float, ...] = ()
        zone_screed_temperatures: tuple[float, ...] = ()
        space_supply_c: tuple[float, ...] = ()
        space_electrical_w: tuple[float, ...] = ()

        if hasattr(model, "q_space_w"):
            space_schedule = tuple(round(on) for on in values(model.space_on))
            space_heat_w = tuple(values(model.q_space_w))
            # What the thermostats would read, which is what comfort was
            # judged on (see zone_observation).
            zone_temperatures = tuple(
                float(pyo.value(model.zone_measured(k))) for k in range(num_steps)
            )
            zone_mass_temperatures = tuple(
                float(pyo.value(model.zone_state[k, ZONE_SLAB_STATES[0]]))
                for k in range(num_steps)
            )

            if ZONE_SCREED_STATES[0] in model.ZONE_STATES:
                zone_screed_temperatures = tuple(
                    float(pyo.value(model.zone_state[k, ZONE_SCREED_STATES[0]]))
                    for k in range(num_steps)
                )
            # The electricity that heat costs, as the objective prices it:
            # the heat moved, either way, over the step's COP.
            space_electrical_w = tuple(
                values(model.space_power_w)
                if hasattr(model, "space_power_w")
                else (
                    model.space_sign * heat_w / cop
                    for heat_w, cop in zip(space_heat_w, model.space_cop, strict=True)
                )
            )

            if getattr(model, "space_supply_c", None):
                space_supply_c = tuple(
                    float(pyo.value(model.space_supply_c[k])) if on else math.nan
                    for k, on in enumerate(space_schedule)
                )

        # The objective's own energy terms (see _build_objective), so what is
        # reported of a plan's cost is what the plan was chosen on.
        electricity_step_kwh = tuple(
            draw_w * hours / 1000.0
            for draw_w, hours in zip(values(model.draw_w), dt_hours, strict=True)
        )
        grid_step_kwh = tuple(
            sum(
                weight * pyo.value(model.active_power_w[k, s])
                for s, weight in enumerate(model.mpc_scenario_weights)
            )
            * dt_hours[k]
            / 1000.0
            for k in range(num_steps)
        )

        return MPCResult(
            schedule=schedule,
            temperatures=temperatures,
            electrical_power_w=tuple(electrical_power_w),
            heat_w=tuple(
                hp + bo for hp, bo in zip(heat_pump_w, booster_w, strict=True)
            ),
            objective_value=objective_value,
            solver_status=str(termination_condition),
            termination_condition=str(termination_condition),
            space_schedule=space_schedule,
            space_heat_w=space_heat_w,
            zone_temperatures=zone_temperatures,
            zone_mass_temperatures=zone_mass_temperatures,
            zone_screed_temperatures=zone_screed_temperatures,
            space_supply_c=space_supply_c,
            space_electrical_w=space_electrical_w,
            electricity_kwh=sum(electricity_step_kwh),
            grid_kwh=sum(grid_step_kwh),
            electricity_step_kwh=electricity_step_kwh,
            grid_step_kwh=grid_step_kwh,
        )
