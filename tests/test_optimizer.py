import dataclasses
from dataclasses import replace

import highspy
import numpy as np
import pytest
from pyomo.contrib.appsi.solvers.highs import Highs

from domain.dynamics import discretize_zoh
from domain.models import (
    TANK_SENSOR_RESOLUTION_K,
    BoilerThermalModel,
    BuildingThermalModel,
    FloorCircuitModel,
    HeatPumpCOPModel,
)
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.physics import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    floor_circuit_conductance_w_per_k,
    lumped_tank_state_space,
    mains_water_temperature_c,
    tank_sensor_temperatures,
    zone_observation,
    zone_size,
    zone_state_space,
)
from features import optimizer as optimizer_module
from features.optimizer import (
    MIP_ABSOLUTE_GAP_EUR,
    MPCOptimizer,
    plan_tank,
    surplus_scenarios,
)

THERMAL_MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=0.15,
    ua_bottom_w_per_k=0.20,
    ua_mix_idle_w_per_k=0.03,
    ua_mix_active_w_per_k=9638.6,
    q_in_nominal_w=3700.0,
)

COP_MODEL = HeatPumpCOPModel(
    eta_carnot=0.5,
    delta_t_cond=5.0,
    delta_t_evap=10.0,
    reference_supply_temperature_c=45.0,
    q_th_at_power_fit_low_w=3000.0,
    q_th_at_power_fit_high_w=5000.0,
)

# Solar rising to a midday peak then falling, 24 steps of 15 minutes (6 hours).
_SOLAR_MIDDAY_W = [500, 1000, 2000, 3000, 3500, 3000, 2000, 1000, 500, 0.0]
# Own sun is only cheaper than the grid where exporting it earns less than
# importing costs; the tests about following the sun are about that case.
NO_EXPORT = MPCConfig(export_price_eur_per_kwh=0.0)


SOLAR_FORECAST_W = [0.0] * 8 + _SOLAR_MIDDAY_W + [0.0] * 6


def _make_input(**overrides) -> MPCInput:
    defaults = dict(
        solar_forecast_w=SOLAR_FORECAST_W,
        ambient_temperature=20.0,
        current_temp_top=30.0,
        current_temp_bottom=28.0,
        boiler_on_current=False,
        target_temperature_top=(10.0,) * len(SOLAR_FORECAST_W),
    )
    defaults.update(overrides)
    return MPCInput(**defaults)


def test_schedules_heating_during_solar_peak_when_sufficient():
    """Regression test for the stated goal: with no draw-off requirement except a
    45 degC deadline that solar alone can satisfy, the optimizer must run the
    boiler during the solar peak (not before, not after) and must not run once
    the target is met - a real bug (see test_boiler_can_turn_off_after_target)
    once made it impossible to ever stop heating once started.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0

    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    on_steps = [k for k, v in enumerate(result.schedule) if v == 1]

    assert len(on_steps) > 0
    # The solar peak is at index 12; heating must overlap it, not happen only
    # long before or after it.
    assert min(on_steps) <= 14
    assert max(on_steps) < 18  # done heating before the deadline check

    # The requirement is actually met.
    assert result.temperatures[18] >= 45.0 - 1e-6

    # It must not keep heating past the point the target is satisfied and
    # exceeded (that would just waste money for no benefit).
    assert result.schedule[-1] == 0


def test_boiler_can_turn_off_after_reaching_target():
    """Regression test for a real bug: an equality startup constraint
    (compressor_start[k] == boiler_on[k] - boiler_on[k-1]) forces compressor_start to -1
    on every stop event, which is infeasible against its own binary domain -
    making any schedule that ever turns the boiler back off unsolvable, so the
    optimizer was forced to keep it on forever once started. A schedule that
    turns on, then off again, must be achievable.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0  # an early, modest requirement well before the horizon ends

    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    # Must turn off again at some point after turning on - not stay on for the
    # rest of the horizon once started.
    assert 0 in result.schedule
    assert 1 in result.schedule

    first_on = result.schedule.index(1)
    later_off = any(v == 0 for v in result.schedule[first_on:])
    assert later_off


def test_tap_forecast_adds_an_additional_heat_sink_to_the_temperature_trajectory():
    """The tap-draw forecast (see MPCInput.tap_forecast_w) must genuinely affect
    the predicted temperature trajectory as an additional heat sink on top of
    passive UA loss, not just be accepted and ignored.
    """

    low_target = (10.0,) * len(SOLAR_FORECAST_W)
    no_sun = [0.0] * len(SOLAR_FORECAST_W)
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    baseline = optimizer.solve(
        _make_input(solar_forecast_w=no_sun, target_temperature_top=low_target)
    )

    with_draws = optimizer.solve(
        _make_input(
            solar_forecast_w=no_sun,
            target_temperature_top=low_target,
            # Small enough that the resulting drop stays well above the 10 degC
            # floor (starting ~29 degC) - isolates the tap term's effect on the
            # trajectory without also triggering a heating-decision difference.
            tap_forecast_w=(300.0,) * len(SOLAR_FORECAST_W),
        )
    )

    # The target is already below the starting temperature, so the boiler need
    # not run at all either way - this isolates the tap term's effect on the
    # simulated trajectory from any heating-decision difference.
    assert baseline.schedule == with_draws.schedule
    assert all(v == 0 for v in baseline.schedule)
    assert all(
        with_draws.temperatures[k] < baseline.temperatures[k] - 1e-6
        for k in range(1, len(SOLAR_FORECAST_W))
    )


def test_tap_forecast_can_force_additional_heating_before_a_deadline():
    """A large forecasted draw between the last heating opportunity and a
    deadline must force the optimizer to heat more than it would without it -
    the entire point of making tap usage visible to planning.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    baseline = optimizer.solve(_make_input(target_temperature_top=tuple(target)))

    tap_forecast = [0.0] * len(SOLAR_FORECAST_W)
    for k in range(13, 18):
        tap_forecast[k] = 5000.0

    with_draw = optimizer.solve(
        _make_input(
            target_temperature_top=tuple(target),
            tap_forecast_w=tuple(tap_forecast),
        )
    )

    assert with_draw.temperatures[18] >= 45.0 - 1e-6
    assert sum(with_draw.schedule) > sum(baseline.schedule)


def test_thermal_dynamics_matches_manual_discretization():
    """The pyomo thermal_dynamics constraints must reproduce exactly the same
    trajectory as directly calling discretize_zoh with the lumped state-space -
    verifies the MPC's own dynamics formulation against the shared physics
    utility, independent of the solver's optimization behavior.
    """

    data = _make_input(boiler_on_current=True)

    # Minimum runtime of 1 so an arbitrary short on/off pattern doesn't conflict
    # with the (unrelated) scheduling constraint this test isn't exercising.
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(compressor_min_runtime_steps=1))
    model = optimizer._build_model(data)

    # Force a known, arbitrary on/off pattern and re-derive T by hand - one that
    # stays below the tank's (default) maximum, which is not what is tested here.
    pattern = [1, 1, 0, 0, 0, 0] * 4
    pattern = pattern[: len(SOLAR_FORECAST_W)]

    for k, v in enumerate(pattern):
        model.boiler_on[k].fix(v)
        # Heat input is continuous (the compressor modulates), so fix it to the
        # nominal output this test re-derives by hand.
        model.q_heat_pump_w[k].fix(THERMAL_MODEL.q_in_nominal_w if v else 0.0)

    from pyomo.contrib.appsi.solvers.highs import Highs

    results = Highs().solve(model)

    from pyomo.contrib.appsi.base import TerminationCondition

    assert results.termination_condition == TerminationCondition.optimal

    import pyomo.environ as pyo

    solved_T = [pyo.value(model.T[k]) for k in range(len(pattern))]

    a, b = lumped_tank_state_space(
        THERMAL_MODEL.volume_l,
        THERMAL_MODEL.ua_top_w_per_k + THERMAL_MODEL.ua_bottom_w_per_k,
    )
    a_d, b_d = discretize_zoh(a, b, MPCConfig().step_hours * 3600.0)

    expected_T = [(data.current_temp_top + data.current_temp_bottom) / 2.0]
    for k in range(len(pattern) - 1):
        q_in = THERMAL_MODEL.q_in_nominal_w if pattern[k] else 0.0
        next_t = (
            a_d[0, 0] * expected_T[-1]
            + b_d[0, 0] * data.ambient_temperature
            + b_d[0, 1] * q_in
        )
        expected_T.append(float(next_t))

    assert np.allclose(solved_T, expected_T, atol=1e-6)


def test_minimum_runtime_is_respected():
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0

    config = MPCConfig(compressor_min_runtime_steps=4)
    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, config)
    result = optimizer.solve(data)

    schedule = result.schedule
    k = 0
    while k < len(schedule):
        if schedule[k] == 1 and (k == 0 or schedule[k - 1] == 0):
            run_length = 0
            j = k
            while j < len(schedule) and schedule[j] == 1:
                run_length += 1
                j += 1
            assert run_length >= config.compressor_min_runtime_steps or j == len(
                schedule
            )
            k = j
        else:
            k += 1


def test_the_heat_pump_stays_off_for_a_while_after_a_run():
    """Two targets a few steps apart: the second must be served by the same run
    or a later one, never by a second run within the pause (see
    MPCConfig.heat_pump_min_off_steps)."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[8] = 40.0
    target[12] = 45.0

    config = MPCConfig(heat_pump_min_off_steps=4, compressor_min_runtime_steps=1)
    result = MPCOptimizer(THERMAL_MODEL, config).solve(
        _make_input(target_temperature_top=tuple(target))
    )

    schedule = result.schedule
    off_steps = 0

    for k, on in enumerate(schedule):
        if on and k > 0 and schedule[k - 1] == 0:
            assert off_steps >= config.heat_pump_min_off_steps
        off_steps = 0 if on else off_steps + 1


def test_no_run_starts_before_the_pause_since_the_last_one_has_passed():
    """The pause counts from the last real run, not from the horizon's start:
    with half of it gone, only the remaining steps stay off."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[6] = 45.0

    config = MPCConfig(heat_pump_min_off_steps=4, compressor_min_runtime_steps=1)
    optimizer = MPCOptimizer(THERMAL_MODEL, config)
    data = _make_input(
        target_temperature_top=tuple(target),
        idle_elapsed_hours=0.5,  # two of the four steps have passed
    )

    schedule = optimizer.solve(data).schedule

    assert schedule[:2] == (0, 0)
    assert any(schedule[2:])


def test_identical_solar_band_matches_planning_on_p50_alone():
    """With p10 = p50 = p90 the expected-cost objective must reduce exactly to
    the single-scenario one (the weights sum to 1) - the band only changes the
    plan when it actually carries uncertainty.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    on_p50 = optimizer.solve(_make_input(target_temperature_top=tuple(target)))
    with_band = optimizer.solve(
        _make_input(
            target_temperature_top=tuple(target),
            solar_p10_w=tuple(SOLAR_FORECAST_W),
            solar_p90_w=tuple(SOLAR_FORECAST_W),
        )
    )

    # The solver stops within MIP_ABSOLUTE_GAP_EUR of the optimum, so where
    # plans lie that close together either may come back.
    assert with_band.objective_value == pytest.approx(
        on_p50.objective_value, abs=MIP_ABSOLUTE_GAP_EUR
    )


def test_uncertain_solar_window_loses_to_a_certain_one_with_less_p50():
    """The reason for costing grid import as an expectation: a window whose
    p50 covers the heat pump but whose p10 does not is, on average, more
    expensive than a window with a bit less but certain sun. Planning on p50
    alone picks the uncertain window; the scenario objective must pick the
    certain one.
    """

    horizon = 24
    certain = range(4, 9)
    uncertain = range(12, 17)

    p10, p50, p90 = ([0.0] * horizon for _ in range(3))
    for k in certain:
        p10[k] = p50[k] = p90[k] = 2500.0  # 500 W short of the 3 kW heat pump
    for k in uncertain:
        p10[k], p50[k], p90[k] = 0.0, 3500.0, 7000.0

    target = [10.0] * horizon
    target[20] = 45.0

    optimizer = MPCOptimizer(THERMAL_MODEL, NO_EXPORT)
    common = dict(solar_forecast_w=p50, target_temperature_top=tuple(target))

    on_p50 = optimizer.solve(_make_input(**common))
    with_band = optimizer.solve(
        _make_input(**common, solar_p10_w=tuple(p10), solar_p90_w=tuple(p90))
    )

    # Compare where each plan puts its heating rather than expecting one window
    # to be skipped entirely.
    def steps_in(result, window):
        return sum(result.schedule[k] for k in window)

    assert steps_in(on_p50, uncertain) > steps_in(on_p50, certain)
    assert steps_in(with_band, certain) > steps_in(with_band, uncertain)
    assert with_band.temperatures[20] >= 45.0 - 1e-6


def test_validate_input_rejects_a_solar_band_of_the_wrong_length():
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    with pytest.raises(ValueError):
        optimizer.solve(_make_input(solar_p10_w=(0.0, 0.0), solar_p90_w=(0.0, 0.0)))

    with pytest.raises(ValueError):
        optimizer.solve(_make_input(solar_p10_w=tuple(SOLAR_FORECAST_W)))


def test_baseload_takes_its_share_of_the_sun_first():
    """Only solar beyond the rest of the house's own draw is available to the
    heat pump: with a baseload as large as the solar forecast, heating costs the
    same as with no sun at all."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    no_sun = optimizer.solve(
        _make_input(
            solar_forecast_w=[0.0] * len(SOLAR_FORECAST_W),
            target_temperature_top=tuple(target),
        )
    )
    all_sun_used_by_the_house = optimizer.solve(
        _make_input(
            target_temperature_top=tuple(target),
            baseload_forecast_w=tuple(SOLAR_FORECAST_W),
        )
    )

    assert all_sun_used_by_the_house.objective_value == pytest.approx(
        no_sun.objective_value
    )


def test_validate_input_rejects_a_baseload_forecast_of_the_wrong_length():
    data = _make_input(baseload_forecast_w=(0.0,))

    with pytest.raises(ValueError, match="baseload_forecast_w"):
        MPCOptimizer(THERMAL_MODEL, MPCConfig()).solve(data)


MAX_TANK_C = 60.0
LIMITED_MODEL = dataclasses.replace(THERMAL_MODEL, max_tank_temperature_c=MAX_TANK_C)


def test_planning_never_heats_above_the_tank_maximum():
    """A target above the tank's identified maximum cannot be met by heating past
    it - that shortfall belongs in the slack, not in an impossible plan."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = MAX_TANK_C + 10.0

    result = MPCOptimizer(LIMITED_MODEL, MPCConfig()).solve(
        _make_input(target_temperature_top=tuple(target))
    )

    assert max(result.temperatures) <= MAX_TANK_C + 1e-6
    assert any(result.schedule)


def test_a_tank_hotter_than_the_maximum_is_simply_not_heated():
    """A tank above the maximum (e.g. after a manual legionella cycle) must leave
    the plan solvable."""

    result = MPCOptimizer(LIMITED_MODEL, MPCConfig()).solve(
        _make_input(current_temp_top=65.0, current_temp_bottom=65.0)
    )

    assert not any(result.schedule)


def _running_run_input(elapsed_hours: float) -> MPCInput:
    """A run in progress that is not needed (target already met, no sun), so
    stopping it would be cheapest."""

    return _make_input(
        solar_forecast_w=[0.0] * len(SOLAR_FORECAST_W),
        current_temp_top=40.0,
        current_temp_bottom=40.0,
        boiler_on_current=True,
        compressor_elapsed_hours=elapsed_hours,
    )


def test_a_run_just_started_keeps_heating_until_its_minimum_runtime():
    config = MPCConfig()
    result = MPCOptimizer(THERMAL_MODEL, config).solve(_running_run_input(0.1))

    assert result.schedule[: config.compressor_min_runtime_steps] == (1, 1)


def test_a_run_under_way_keeps_the_compressor_from_the_zone_planned_there():
    """The previous plan had the zone where the tank has since started: the
    tank's plan around those runs finishes its own run first."""

    tank = MPCOptimizer(THERMAL_MODEL, MPCConfig())._tank_setting(
        _running_run_input(0.1)
    )
    zone_on = [1] * 4 + [0] * (tank.num_steps - 4)

    plan = plan_tank(tank, zone_on)

    assert tank.running_steps > 0
    assert all(plan.heat_pump_w[: tank.running_steps])


def test_a_run_past_its_minimum_runtime_may_stop():
    result = MPCOptimizer(THERMAL_MODEL, MPCConfig()).solve(_running_run_input(1.0))

    assert result.schedule[0] == 0


def test_a_run_past_its_minimum_runtime_is_finished_not_split():
    """A run under way heats for a later target: it goes on until it has,
    rather than stopping - for the zone, or for sun later on - and heating
    the same target in a second block (real data, 25 Oct 2026)."""

    num_steps = len(SOLAR_FORECAST_W)
    target = [10.0] * (num_steps - 2) + [45.0] * 2
    tank = MPCOptimizer(THERMAL_MODEL, MPCConfig())._tank_setting(
        _make_input(
            current_temp_top=31.0,
            current_temp_bottom=30.0,
            boiler_on_current=True,
            compressor_elapsed_hours=0.6,
            target_temperature_top=tuple(target),
        )
    )
    alone = plan_tank(tank)
    around_zone = plan_tank(tank, [1, 1] + [0] * (num_steps - 2))

    assert tank.running_steps == 0
    assert _blocks([int(w > 0) for w in alone.heat_pump_w]) == 1
    assert alone.heat_pump_w[0] > 0
    assert not any(around_zone.heat_pump_w[: num_steps - 2])


def test_the_tank_does_not_take_the_compressor_back_from_the_zone():
    """A block that hands the compressor to the zone does not come back in
    that run: the next block waits until the compressor has paused."""

    num_steps = len(SOLAR_FORECAST_W)
    target = [10.0] * 2 + [36.0] * 4 + [10.0] * 2 + [45.0] * (num_steps - 8)
    tank = MPCOptimizer(THERMAL_MODEL, MPCConfig())._tank_setting(
        _make_input(
            solar_forecast_w=[0.0] * num_steps,
            target_temperature_top=tuple(target),
            idle_elapsed_hours=5.0,
        )
    )
    zone_on = [0, 0, 1, 1, 1, 1] + [0] * (num_steps - 6)
    heated = [int(w > 0) for w in plan_tank(tank, zone_on).heat_pump_w]
    min_off = MPCConfig().heat_pump_min_off_steps

    assert heated[:2] == [1, 1]
    assert not any(heated[2 : 6 + min_off])


def test_the_solve_starts_from_the_previous_plan_and_still_finds_the_best():
    """A solve stopped at its time limit publishes its best plan so far, so it
    starts from the previous one: loaded beyond the first step as it was, and
    left behind where a better plan exists - here a run before the sun."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    previous = tuple(int(k in (2, 3, 4)) for k in range(len(SOLAR_FORECAST_W)))
    data = _make_input(target_temperature_top=tuple(target), previous_tank_on=previous)
    optimizer = MPCOptimizer(THERMAL_MODEL, NO_EXPORT)

    model = optimizer._build_model(data)
    solver = optimizer._solver()
    solver.set_instance(model)
    assert optimizer._warm_start(model, data, solver)
    assert [round(model.tank_heating[k].value) for k in model.K][1:] == list(
        previous[1:]
    )

    warm = optimizer.solve(data)
    cold = optimizer.solve(replace(data, previous_tank_on=()))

    assert warm.schedule == cold.schedule
    assert not any(warm.schedule[:8])


def test_the_plan_is_solved_apart_from_its_warm_start(monkeypatch):
    """The warm start runs on a HiGHS instance of its own, and the plan on one
    that has run nothing before. On the instance a warm start had just run on
    - its previous plan no longer held - HiGHS proved a plan optimal at the
    root that was not, on a model identical to a fresh instance's (real data:
    0.4674 EUR against an optimum of 0.4656). Not reproduced on a small model,
    so what is guarded is that the instance is never shared again. With the
    zone: the tank alone is not solved by HiGHS at all (see plan_tank)."""

    runs = []
    run = highspy.Highs.run

    def recorded(highs):
        runs.append(highs)
        return run(highs)

    monkeypatch.setattr(highspy.Highs, "run", recorded)

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    # One that holds and one that does not: a single step is shorter than
    # the compressor's minimum runtime.
    for previous in ((2, 3, 4), (5,)):
        runs.clear()
        MPCOptimizer(THERMAL_MODEL, NO_EXPORT, building_model=BUILDING_MODEL).solve(
            _zone_input(
                target_temperature_top=tuple(target),
                previous_tank_on=tuple(
                    int(k in previous) for k in range(len(SOLAR_FORECAST_W))
                ),
            )
        )

        # The relaxation that proposes the tank's plans runs first (see
        # MPCOptimizer._relaxed_zone): the warm start and the plan are last.
        *_, warm_start, plan = runs
        assert warm_start is not plan


def test_validate_input_rejects_mismatched_target_length():
    data = _make_input(target_temperature_top=(10.0, 10.0))  # wrong length

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    with pytest.raises(ValueError):
        optimizer.solve(data)


def test_no_heating_scheduled_when_target_already_below_current():
    """If the whole horizon's requirement is already satisfied by the current
    temperature, the optimizer must not spend grid money heating anyway - heat
    beyond what the targets need has no value.
    """

    data = _make_input(
        solar_forecast_w=[0.0] * len(SOLAR_FORECAST_W),
        current_temp_top=50.0,
        current_temp_bottom=50.0,
        target_temperature_top=(10.0,) * len(SOLAR_FORECAST_W),
    )

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    assert all(v == 0 for v in result.schedule)


def test_grid_heat_is_never_bought_just_to_store_it_even_from_a_cold_tank():
    """Heat beyond what the targets need has no value, so even a tank colder
    than its surroundings - where grid heat is cheapest - is not heated."""

    data = _make_input(
        solar_forecast_w=[0.0] * len(SOLAR_FORECAST_W),
        current_temp_top=15.0,
        current_temp_bottom=15.0,
        outdoor_temperature_forecast=(20.0,) * len(SOLAR_FORECAST_W),
    )

    result = MPCOptimizer(THERMAL_MODEL, MPCConfig(), COP_MODEL).solve(data)

    assert all(v == 0 for v in result.schedule)


def test_nothing_is_heated_without_a_target_even_on_surplus_sun():
    """Heat left in the tank at the horizon end has no value (see
    MPCOptimizer._build_model), so without a target even free sun is not
    stored: a start still costs something."""

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(_make_input())

    assert max(SOLAR_FORECAST_W) > MPCConfig().boiler_electrical_power_w
    assert not any(result.schedule)


def test_electrical_power_matches_the_line_the_objective_was_built_from():
    """MPCResult.electrical_power_w (the reported schedule) must be computed
    from exactly the same (alpha, beta) line the objective itself uses to
    cost active_power_w[k] (see _power_line_coefficients) evaluated at the
    solved T[k] - not a separate, display-only estimate. This is the
    consistency the whole linear-in-T reformulation exists for.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0

    outdoor_forecast = (5.0,) * len(SOLAR_FORECAST_W)

    data = _make_input(
        target_temperature_top=tuple(target),
        outdoor_temperature_forecast=outdoor_forecast,
    )

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    result = optimizer.solve(data)

    alpha, beta = optimizer._power_line_coefficients(5.0, max(target))
    on_steps = [k for k, on in enumerate(result.schedule) if on]
    assert on_steps

    for k in on_steps:
        used = result.heat_w[k] / THERMAL_MODEL.q_in_nominal_w
        expected_power_w = (alpha + beta * result.temperatures[k]) * used
        assert result.electrical_power_w[k] == pytest.approx(expected_power_w)

    # Differs meaningfully from the flat fallback, proving the model is
    # actually driving the value, not coincidentally matching it.
    first = on_steps[0]
    assert result.electrical_power_w[first] != pytest.approx(
        MPCConfig().boiler_electrical_power_w
        * result.heat_w[first]
        / THERMAL_MODEL.q_in_nominal_w
    )


def test_power_line_evaluates_cop_at_the_real_supply_reference_not_margin_shifted():
    """Regression test for a real bug: q_th_at_power_fit_low_w/high_w were
    measured at rows whose *real* T_supply was near POWER_FIT_T_LOW_C/
    HIGH_C (see HeatPumpCOPIdentifier.calibrate()), so COP must be evaluated
    directly at those same values - adding the T[k]->T_supply margin a
    second time there evaluates COP at a hotter, never-actually-measured
    T_supply, understating COP and so overstating power.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0  # < COP_MODEL.reference_supply_temperature_c (45), so
    # margin = 45 - 40 = 5 - nonzero, to actually exercise the fix.

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    alpha, beta = optimizer._power_line_coefficients(5.0, max(target))

    margin = COP_MODEL.reference_supply_temperature_c - max(target)
    assert margin > 0.0  # sanity-check the setup

    t_k_high = HeatPumpCOPModel.POWER_FIT_T_HIGH_C - margin
    power_at_high = alpha + beta * t_k_high

    expected_cop = COP_MODEL.cop(5.0, HeatPumpCOPModel.POWER_FIT_T_HIGH_C)
    expected_power_at_high = COP_MODEL.q_th_at_power_fit_high_w / expected_cop

    assert power_at_high == pytest.approx(expected_power_at_high)


def test_reported_electrical_power_rises_as_tank_heats_through_a_run():
    """The whole point of reporting from T[k] instead of a fixed reference is
    that reported electrical draw should visibly rise across a multi-step
    compressor run as the tank heats up - the behavior confirmed on real
    data (P_el rising through a DHW cycle as T_supply rises) that a single
    flat reference cannot reproduce.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[16] = 45.0

    outdoor_forecast = (10.0,) * len(SOLAR_FORECAST_W)

    data = _make_input(
        target_temperature_top=tuple(target),
        outdoor_temperature_forecast=outdoor_forecast,
    )

    optimizer = MPCOptimizer(
        THERMAL_MODEL, MPCConfig(compressor_min_runtime_steps=4), cop_model=COP_MODEL
    )
    result = optimizer.solve(data)

    on_steps = [k for k, v in enumerate(result.schedule) if v == 1]
    assert len(on_steps) >= 2
    assert on_steps == list(range(on_steps[0], on_steps[-1] + 1))  # one contiguous run

    # The full steps only: the run's last one is used for just the part the
    # target needs, and draws for that part alone.
    on_steps_power = [result.electrical_power_w[k] for k in on_steps[:-1]]
    # Strictly increasing across the run: later steps have a hotter tank
    # (T[k] rises monotonically while heating), hence a lower COP and higher
    # reported electrical draw.
    assert all(
        on_steps_power[i] < on_steps_power[i + 1]
        for i in range(len(on_steps_power) - 1)
    )


def _flat_power_w(result) -> list[float]:
    return [
        MPCConfig().boiler_electrical_power_w * heat / THERMAL_MODEL.q_in_nominal_w
        for heat in result.heat_w
    ]


def test_electrical_power_falls_back_to_flat_assumption_without_cop_model():
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0
    data = _make_input(target_temperature_top=tuple(target))
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    assert any(result.schedule)
    assert list(result.electrical_power_w) == pytest.approx(_flat_power_w(result))


def test_electrical_power_falls_back_without_outdoor_forecast_even_with_cop_model():
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0
    # No outdoor_temperature_forecast.
    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    result = optimizer.solve(data)

    assert any(result.schedule)
    assert list(result.electrical_power_w) == pytest.approx(_flat_power_w(result))


def test_cop_clamped_to_sanity_range_for_implausible_inputs():
    """An outdoor/target combination outside anything the model was fit on
    must not translate into an absurd electrical-power estimate - COP is
    clamped to HeatPumpCOPModel's own [MIN_COP, MAX_COP] sanity range
    before the linear (alpha, beta) fit is built (see
    _power_line_coefficients), so the fit itself never sees an implausible
    endpoint.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0
    margin = max(COP_MODEL.reference_supply_temperature_c - max(target), 0.0)

    # An outdoor forecast this close to POWER_FIT_T_HIGH_C itself (COP is
    # evaluated directly at the real T_supply reference points, no margin -
    # see _power_line_coefficients) pushes the raw formula's COP above any
    # real compressor's achievable efficiency - implausible on purpose, to
    # exercise the clamp.
    outdoor_forecast = (60.0,) * len(SOLAR_FORECAST_W)

    data = _make_input(
        target_temperature_top=tuple(target),
        outdoor_temperature_forecast=outdoor_forecast,
    )

    raw_cop = COP_MODEL.cop(60.0, HeatPumpCOPModel.POWER_FIT_T_HIGH_C)
    assert raw_cop > HeatPumpCOPModel.MAX_COP  # sanity-check the setup

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    result = optimizer.solve(data)

    alpha, beta = optimizer._power_line_coefficients(60.0, max(target))
    on_steps = [k for k, on in enumerate(result.schedule) if on]
    assert on_steps

    for k in on_steps:
        used = result.heat_w[k] / THERMAL_MODEL.q_in_nominal_w
        expected_power_w = (alpha + beta * result.temperatures[k]) * used
        assert result.electrical_power_w[k] == pytest.approx(expected_power_w)
    # The clamp must actually have engaged for this scenario, i.e. the fit's
    # high endpoint used the clamped, not the implausible raw, COP. The line
    # is valid in T[k] terms, so the high reference point corresponds to
    # T[k] = POWER_FIT_T_HIGH_C - margin (T_supply = T[k] + margin).
    unclamped_power_at_high = COP_MODEL.q_th_at_power_fit_high_w / raw_cop
    t_k_high = HeatPumpCOPModel.POWER_FIT_T_HIGH_C - margin
    clamped_power_at_high = alpha + beta * t_k_high
    assert clamped_power_at_high == pytest.approx(
        COP_MODEL.q_th_at_power_fit_high_w / HeatPumpCOPModel.MAX_COP
    )
    assert clamped_power_at_high != pytest.approx(unclamped_power_at_high)


def test_lumped_state_space_matches_full_tank_capacity():
    c_expected = RHO_WATER_KG_PER_L * THERMAL_MODEL.volume_l * CP_WATER_J_PER_KG_K
    ua_total = THERMAL_MODEL.ua_top_w_per_k + THERMAL_MODEL.ua_bottom_w_per_k

    a, b = lumped_tank_state_space(THERMAL_MODEL.volume_l, ua_total)

    assert a.shape == (1, 1)
    assert b.shape == (1, 3)
    assert a[0, 0] == pytest.approx(-ua_total / c_expected)
    assert b[0, 0] == pytest.approx(ua_total / c_expected)
    assert b[0, 1] == pytest.approx(1.0 / c_expected)
    # Tap draw is a heat sink, opposite sign to Q_in - same magnitude since
    # both terms are a straight W/C_total conversion.
    assert b[0, 2] == pytest.approx(-1.0 / c_expected)


def test_a_long_horizon_meets_a_late_target():
    """A 48h horizon, the longest the app plans (192 quarter-hour steps), is
    planned at full resolution and still meets a target late on its second
    day."""

    horizon = 192
    target = [10.0] * horizon
    target[168] = 45.0

    data = _make_input(
        solar_forecast_w=[0.0] * horizon,
        target_temperature_top=tuple(target),
        current_temp_top=20.0,
        current_temp_bottom=20.0,
    )

    result = MPCOptimizer(THERMAL_MODEL, MPCConfig()).solve(data)

    assert result.temperatures[168] >= 45.0 - 1e-6
    assert len(result.schedule) == horizon
    assert len(result.temperatures) == horizon


def test_target_holds_through_each_step():
    """A target must hold for the whole step, not just where it starts.

    T[k] is the temperature at the START of step k, so checking only that
    leaves a coasting tank free to sag below the target before the step ends
    (a real plan with one-hour steps met its 18:00 target while dropping under
    it by 18:45).
    """

    horizon = 48
    solar = (0.0,) * horizon

    # A deadline held for a few hours so the tank has to still be above it at
    # the end of a step and not only at its start.
    # Kept clear of the very last step, whose end the model deliberately does
    # not represent - see end_temperature_rule.
    target = [10.0] * horizon
    deadline, closes = horizon - 24, horizon - 8
    for k in range(deadline, closes):
        target[k] = 45.0

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    result = optimizer.solve(
        _make_input(
            solar_forecast_w=list(solar),
            target_temperature_top=tuple(target),
        )
    )

    shortfall = [
        target[k] - result.temperatures[k]
        for k in range(deadline, closes)
        if target[k] - result.temperatures[k] > 0.0
    ]

    assert not shortfall, f"plan dips up to {max(shortfall):.3f} K below target"


# The air and, behind it, the screed and internal walls the floor heats.
BUILDING_MODEL = BuildingThermalModel(
    ua_envelope_w_per_k=130.0,
    ua_air_mass_w_per_k=500.0,
    c_air_j_per_k=2.5e6,
    c_mass_j_per_k=37.5e6,
    a_eff_m2=9.0,
    # A pure air sensor, so the plans these tests check are judged on the air
    # node exactly; the operative reading has its own test.
    sensor_mass_fraction=0.0,
    internal_gain_fraction=1.0,
    # Both rooms alike and closely coupled: these tests are about the plan,
    # not how the zone divides.
    a_eff_rest_m2=0.0,
    ua_rooms_w_per_k=500.0,
    living_envelope_fraction=0.5,
    living_area_fraction=0.5,
    closed_shutter_gain_fraction=0.0,
)


def _zone_state(room_c: float, slab_c: float) -> tuple[float, ...]:
    """Both rooms at one temperature and both slabs at another (see
    physics.zone_state_space)."""

    return (room_c, room_c, slab_c, slab_c)


def _zone_input(**overrides) -> MPCInput:
    """A zone starting below a target it has to reach later in the horizon."""

    target = [10.0] * len(SOLAR_FORECAST_W)

    for k in range(len(SOLAR_FORECAST_W) // 2, len(SOLAR_FORECAST_W)):
        target[k] = 20.0

    defaults = dict(
        zone_state=_zone_state(19.5, 19.5),
        zone_target_temperature=tuple(target),
        zone_internal_gain_w=(0.0,) * len(SOLAR_FORECAST_W),
    )
    defaults.update(overrides)
    # The target held exactly, unless a test asks for a band below it.
    defaults.setdefault("zone_minimum_temperature", defaults["zone_target_temperature"])

    return _make_input(**defaults)


def _compressor_starts(result) -> int:
    """Starts of the compressor, whichever demand it was serving."""

    on = [
        max(dhw, space)
        for dhw, space in zip(
            result.schedule,
            result.space_schedule or (0,) * len(result.schedule),
            strict=True,
        )
    ]

    return sum(1 for k, value in enumerate(on) if value and (k == 0 or not on[k - 1]))


def test_the_zone_waits_while_the_booster_finishes_the_tank():
    """The booster finishes the tank with the heat pump in its hot water mode,
    which then serves nothing else: no step both heats the tank, by either
    source, and serves the zone."""

    thermal_model = replace(
        THERMAL_MODEL,
        heat_pump_max_tank_temperature_c=55.0,
        max_tank_temperature_c=60.0,
        booster_heat_w=2000.0,
        booster_electrical_w=2400.0,
    )
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[20] = 59.0
    result = MPCOptimizer(
        thermal_model, MPCConfig(), building_model=BUILDING_MODEL
    ).solve(_zone_input(target_temperature_top=tuple(target)))

    assert max(result.temperatures) > 55.0
    assert any(result.space_schedule)
    assert not any(
        tank and space
        for tank, space in zip(result.schedule, result.space_schedule, strict=True)
    )


def test_without_a_building_model_nothing_about_the_tank_changes():
    """Space heating is opt-in: no model, no second demand, same plan."""

    data = _zone_input(target_temperature_top=(45.0,) * len(SOLAR_FORECAST_W))

    without = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL).solve(data)
    assert without.space_schedule == ()
    assert without.zone_temperatures == ()

    # And the tank plan is the one it would have made on its own.
    plain = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL).solve(
        _make_input(target_temperature_top=(45.0,) * len(SOLAR_FORECAST_W))
    )
    assert without.schedule == plain.schedule


def test_the_zone_and_the_tank_are_never_served_at_once():
    """One compressor, one three-way valve: sequential is what the hardware does."""

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input(target_temperature_top=(45.0,) * len(SOLAR_FORECAST_W)))

    assert result.space_schedule, "expected the zone to be heated at all"

    for dhw, space in zip(result.schedule, result.space_schedule, strict=True):
        assert not (dhw and space), "tank and zone served in the same step"


def test_both_demands_are_served_in_one_compressor_start():
    """The point of counting compressor starts rather than boiler starts.

    Nothing rewards chaining the two demands - it simply stops costing extra,
    because the compressor never stops while the valve switches. A plan that
    heats the tank and the zone in two separate runs pays a second start for
    nothing.
    """

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input(target_temperature_top=(45.0,) * len(SOLAR_FORECAST_W)))

    assert any(result.schedule), "expected the tank to be heated"
    assert any(result.space_schedule), "expected the zone to be heated"
    assert _compressor_starts(result) == 1


def test_a_floor_run_under_way_goes_on_without_a_start():
    """A zone that needs heat now is served by the run already under way:
    stopping it to start again pays a start for nothing (real data, 11 Oct:
    the floor planned off for half an hour in the middle of its run)."""

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    )
    data = _zone_input(zone_target_temperature=(20.0,) * len(SOLAR_FORECAST_W))
    idle = optimizer.solve(data)
    running = optimizer.solve(replace(data, space_on_current=True))

    assert running.space_schedule[0] == 1
    assert running.objective_value < idle.objective_value


def test_the_zone_reaches_its_target():
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input())

    data = _zone_input()

    for k, target in enumerate(data.zone_target_temperature):
        assert result.zone_temperatures[k] >= target - 1e-6


def test_heating_the_zone_is_not_free():
    """It draws through the same compressor, so it has to be costed."""

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    )

    heated = optimizer.solve(_zone_input())
    # The same zone with nothing asked of it.
    idle = optimizer.solve(
        _zone_input(zone_target_temperature=(10.0,) * len(SOLAR_FORECAST_W))
    )

    assert sum(heated.space_heat_w) > 0.0
    assert sum(idle.space_heat_w) == pytest.approx(0.0, abs=1e-6)
    assert heated.objective_value > idle.objective_value


def _blocks(schedule) -> int:
    return sum(
        1
        for k, value in enumerate(schedule)
        if value and (k == 0 or not schedule[k - 1])
    )


def test_the_tank_does_not_get_a_second_turn_within_one_run():
    """Once the tank stops it has reached its target, so it will not need
    heating again before the compressor stops.

    Without this the plan flapped the three-way valve - five steps of tank, six
    of zone, then a single step of tank. That single step delivers almost
    nothing: a DHW start spends an estimated 0.4-0.5 kWh reheating the loops and
    plate exchanger before the tank gains at all.
    """

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input(target_temperature_top=(45.0,) * len(SOLAR_FORECAST_W)))

    assert _blocks(result.schedule) == 1, "the tank was served in more than one block"
    assert _compressor_starts(result) == 1

    # Both demands still met, so the rule did not simply forbid one of them.
    assert max(result.temperatures) >= 45.0 - 1e-6
    assert result.zone_temperatures[-1] >= 20.0 - 1e-6


def test_heating_hands_the_compressor_from_the_zone_to_the_tank():
    """A heating run of the zone and the tank share one compressor run in
    either order: the zone first, then the tank, with no stop in between - a
    stop pays a start and brings up the same loop after it (real data, 11
    Oct: tank, zone, a stop and then the tank again)."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[16] = 45.0
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(
        _zone_input(
            zone_target_temperature=(20.0,) * len(SOLAR_FORECAST_W),
            target_temperature_top=tuple(target),
            space_on_current=True,
        )
    )
    zone = [k for k, on in enumerate(result.space_schedule) if on]
    tank = [k for k, on in enumerate(result.schedule) if on]

    assert zone and tank and max(zone) < min(tank)
    assert _compressor_starts(result) == 1, "the run under way stopped"


def test_the_tanks_start_up_follows_the_outdoor_temperature():
    """With the start-up's shortfall as a line in the outdoor temperature,
    a step starting the compressor delivers the heat line less that shortfall
    there - more short when colder; without a forecast, the single ramp."""

    cop_model = replace(
        COP_MODEL,
        start_ramp_seconds=884.0,
        q_in_at_zero_outdoor_w=5400.0,
        q_in_per_outdoor_w_per_k=50.0,
        start_shortfall_at_zero_outdoor_j=1100.0 * 3600.0,
        start_shortfall_per_outdoor_j_per_k=-17.0 * 3600.0,
    )
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=cop_model)

    for outdoor_c in (12.0, 20.0):
        heat_w = 5400.0 + 50.0 * outdoor_c
        shortfall_j = (1100.0 - 17.0 * outdoor_c) * 3600.0
        assert heat_w * 1800.0 * optimizer._ramp_fraction(0.5, outdoor_c) == (
            pytest.approx(heat_w * 1800.0 - shortfall_j)
        )

    assert optimizer._ramp_fraction(0.25, 12.0) < optimizer._ramp_fraction(0.25, 20.0)
    assert optimizer._ramp_fraction(0.25, None) == pytest.approx(1.0 - 884.0 / 1800.0)


def test_a_tank_handed_the_compressor_by_the_zone_carries_no_ramp():
    """The ramp is the compressor's own rise to full output: a tank block that
    starts it heats less in its first step, one the zone hands over to heats in
    full, at the power line - as the zone does after the tank. Costed with a
    start's ramp, a hand-over came out dearer than a start, and the plan put
    the tank before the zone, hours ahead of its target (real plan, 17 Oct
    2026: hot water from 8:30 for 18:00, the zone after it)."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 45.0
    data = _zone_input(target_temperature_top=tuple(target))
    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=replace(COP_MODEL, start_ramp_seconds=875.0),
        building_model=BUILDING_MODEL,
    )
    tank = optimizer._tank_setting(data)
    handed = 5
    after_zone = plan_tank(tank, [1] * handed + [0] * (tank.num_steps - handed))
    started = plan_tank(tank)

    first = [
        next(k for k, w in enumerate(plan.heat_pump_w) if w > 0)
        for plan in (after_zone, started)
    ]

    assert first[0] == handed
    assert after_zone.heat_pump_w[handed] == pytest.approx(tank.heat_w[handed])
    assert started.heat_pump_w[first[1]] == pytest.approx(
        tank.ramp[first[1]] * tank.heat_w[first[1]]
    )


def test_a_zone_run_handed_over_by_the_tank_takes_the_loops_heat():
    """The water in the heat pump and its pipes, at the tank's temperature as
    the valve turns, flows on into the floor: the first step of a zone run the
    tank hands over takes that heat on top of the compressor's, and no other
    step does."""

    handover_j = 0.25 * 3.6e6
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 45.0
    data = _zone_input(target_temperature_top=tuple(target))
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
        space_cop_model=replace(
            COP_MODEL, start_ramp_seconds=0.0, handover_heat_j=handover_j
        ),
    ).solve(data)
    handed = [
        k
        for k in range(1, len(result.schedule))
        if result.space_schedule[k] and result.schedule[k - 1]
    ]

    assert handed, "expected the tank to hand the run to the zone"

    a_d, b_d = discretize_zoh(*zone_state_space(BUILDING_MODEL), 900.0)
    state = np.array(data.zone_state, dtype=float)

    for k in range(len(result.schedule) - 1):
        floor_w = result.space_heat_w[k] + handover_j / 900.0 * (k in handed)
        state = a_d @ state + b_d @ np.array(
            [data.ambient_temperature, 0.0, 0.0, 0.0, floor_w, 0.0]
        )

        assert float(zone_observation(BUILDING_MODEL) @ state) == pytest.approx(
            result.zone_temperatures[k + 1], abs=1e-4
        )


def test_hot_water_is_heated_in_one_block_the_booster_finishing_it():
    """The heat pump to its limit and the booster on from there, one block
    before the target - not a block before a zone run and the booster's after
    it, with a second start (real data, 11 Oct)."""

    thermal_model = replace(
        THERMAL_MODEL,
        heat_pump_max_tank_temperature_c=55.0,
        max_tank_temperature_c=60.0,
        booster_heat_w=2000.0,
        booster_electrical_w=2400.0,
    )
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[20] = 59.0
    result = MPCOptimizer(
        thermal_model,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input(target_temperature_top=tuple(target)))

    assert max(result.temperatures) > 55.0
    assert any(result.space_schedule)
    assert _blocks(result.schedule) == 1
    assert _compressor_starts(result) == 1


def test_the_zone_may_still_be_served_before_and_after_the_tank():
    """The restriction is on the tank, not on the valve.

    The heat pump interrupts space heating for a tank that calls and returns to
    it afterwards, so the zone may be served in more than one block within a
    run - only the tank may not.
    """

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    )
    result = optimizer.solve(
        _zone_input(target_temperature_top=(45.0,) * len(SOLAR_FORECAST_W))
    )

    # Nothing in the model forbids it; this only records that the tank's rule
    # was not accidentally applied to the zone as well.
    assert not hasattr(optimizer, "_space_done")
    assert _blocks(result.space_schedule) >= 1


@pytest.mark.parametrize("tank_steps, feasible", [(1, False), (2, True)])
def test_the_tank_part_of_a_zone_run_lasts_the_minimum_runtime(tank_steps, feasible):
    """The zone made up the compressor's minimum runtime, so a single quarter of
    hot water followed by a quarter of zone was a valid plan (real data, 4 Oct:
    one quarter at 20:45 from a solve stopped at its time limit)."""

    optimizer = MPCOptimizer(
        THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL, building_model=BUILDING_MODEL
    )
    model = optimizer._build_model(_zone_input())
    first = 8

    for k in range(first - 1, first + tank_steps + 1):
        model.boiler_on[k].fix(int(first <= k < first + tank_steps))

    solver = Highs()
    solver.config.load_solution = False
    results = solver.solve(model)

    assert (results.best_feasible_objective is not None) == feasible


def test_the_two_node_zone_is_brought_to_its_target():
    """The MPC plans against whichever structure it was handed.

    The air reaches the target and holds it. It arrives slightly late, because
    the compressor's heat enters the screed and reaches the air only through
    the coupling - that lag is the structure telling the truth about floor
    heating, not the plan giving up, so the transient shortfall is bounded
    rather than forbidden.
    """

    data = _zone_input(zone_state=_zone_state(19.5, 19.5))

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(data)

    assert any(result.space_schedule), "expected the zone to be heated"

    shortfall = max(
        target - temperature
        for temperature, target in zip(
            result.zone_temperatures, data.zone_target_temperature, strict=True
        )
    )

    assert shortfall < 0.25, f"zone dips {shortfall:.3f} K below target"
    assert result.zone_temperatures[-1] >= data.zone_target_temperature[-1] - 1e-6


def test_a_two_node_zone_is_not_planned_without_its_mass_temperature():
    """Nothing measures the screed, so a plan that needs it must be given the
    filter's estimate. Guessing it would silently decide whether the floor is
    charged or cold, which is the whole question the second state answers.
    """

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input(zone_state=()))

    assert result.space_schedule == ()
    assert result.zone_temperatures == ()


def test_a_charged_screed_needs_less_heating_than_a_cold_one():
    """The effect the second state exists for.

    Same air temperature, same comfort target, same weather - only the heat
    already stored in the floor differs."""

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    )

    cold = optimizer.solve(_zone_input(zone_state=_zone_state(19.5, 18.0)))
    charged = optimizer.solve(_zone_input(zone_state=_zone_state(19.5, 24.0)))

    assert sum(charged.space_heat_w) < sum(cold.space_heat_w)


def test_solar_gain_displaces_heating_in_the_two_node_zone():
    """The sun warms the room node, and so displaces heating the compressor
    would otherwise have to deliver."""

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    )

    steps = len(SOLAR_FORECAST_W)
    dark = optimizer.solve(_zone_input(zone_state=_zone_state(19.5, 19.5)))
    sunny = optimizer.solve(
        _zone_input(
            zone_state=_zone_state(19.5, 19.5),
            zone_living_gain_w=(2000.0,) * steps,
        )
    )

    assert sum(sunny.space_heat_w) < sum(dark.space_heat_w)


def test_comfort_is_judged_on_what_the_thermostat_reads():
    """A wall thermostat reads part mass, so warm air over a cold floor reads
    colder than the air - and that reading is what the plan is held to and
    reports, not the air node.
    """

    data = _zone_input(
        zone_state=_zone_state(22.0, 16.0),
        zone_target_temperature=(20.0,) * len(SOLAR_FORECAST_W),
    )

    def solve(building):
        return MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(),
            cop_model=COP_MODEL,
            building_model=building,
        ).solve(data)

    air_only = solve(BUILDING_MODEL)
    operative = solve(replace(BUILDING_MODEL, sensor_mass_fraction=0.5))

    # Half air, half mass against the air node alone.
    assert air_only.zone_temperatures[0] == pytest.approx(22.0)
    assert operative.zone_temperatures[0] == pytest.approx(19.0)
    # A reading this far below target is worth heat, and never less of it than
    # the same zone judged on its warmer air.
    assert sum(operative.space_heat_w) >= sum(air_only.space_heat_w) - 1e-6
    assert sum(operative.space_heat_w) > 0.0


def test_a_run_delivers_less_heat_in_the_step_it_starts_in():
    """A compressor takes minutes to reach full output, so the step it starts
    in carries only part of it - and the steps after it carry all of it."""

    ramped = replace(
        THERMAL_MODEL,
        q_in_steady_w=6000.0,
        q_in_ramp_seconds=900.0,
    )
    optimizer = MPCOptimizer(ramped, MPCConfig(), cop_model=COP_MODEL)

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    result = optimizer.solve(_make_input(target_temperature_top=tuple(target)))

    heating = [k for k, on in enumerate(result.schedule) if on]
    assert heating, "expected the tank to be heated"

    first = heating[0]
    # Half the ramp is still missing over a step as long as the ramp itself.
    assert result.heat_w[first] == pytest.approx(0.5 * 6000.0, rel=0.02)
    assert result.heat_w[first + 1] == pytest.approx(6000.0, rel=0.02)


def test_without_an_identified_ramp_every_step_is_a_full_one():
    """The constant-output model this replaced: no ramp, no reduction."""

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    result = optimizer.solve(_make_input(target_temperature_top=tuple(target)))

    heating = [k for k, on in enumerate(result.schedule) if on]

    assert result.heat_w[heating[0]] == pytest.approx(
        THERMAL_MODEL.q_in_nominal_w, rel=0.02
    )


def test_a_higher_ceiling_buffers_the_sun_in_the_floor_and_the_ceiling_holds():
    """Room above the target is what a floor buffer is charged into: with it the
    plan moves heat into the sunny hours, and never past the ceiling."""

    steps = 96
    hour = (np.arange(steps) * 0.25 + 6.0) % 24.0
    sun = np.clip(4000.0 * np.sin(np.pi * (hour - 8.0) / 9.0), 0.0, None)
    sun[(hour < 8.0) | (hour > 17.0)] = 0.0
    sunny = sun > 500.0
    model = BuildingThermalModel(
        ua_envelope_w_per_k=80.0,
        ua_air_mass_w_per_k=650.0,
        c_air_j_per_k=2.6e6,
        c_mass_j_per_k=35.0e6,
        a_eff_m2=3.0,
        sensor_mass_fraction=0.5,
        internal_gain_fraction=0.6,
        a_eff_rest_m2=0.0,
        ua_rooms_w_per_k=500.0,
        living_envelope_fraction=0.5,
        living_area_fraction=0.5,
        closed_shutter_gain_fraction=0.0,
    )
    data = _make_input(
        solar_forecast_w=list(sun),
        target_temperature_top=(10.0,) * steps,
        outdoor_temperature_forecast=(5.0,) * steps,
        zone_state=_zone_state(20.1, 20.4),
        zone_target_temperature=tuple(np.where(hour < 22.0, 20.0, 18.0)),
        zone_minimum_temperature=tuple(np.where(hour < 22.0, 20.0, 18.0)),
        zone_internal_gain_w=(150.0,) * steps,
    )
    optimizer = MPCOptimizer(
        THERMAL_MODEL, NO_EXPORT, cop_model=COP_MODEL, building_model=model
    )

    def sunny_heat(ceiling_c: float):
        result = optimizer.solve(
            replace(data, zone_maximum_temperature=(ceiling_c,) * steps)
        )
        assert max(result.zone_temperatures) <= ceiling_c + 1e-6

        return float(np.asarray(result.space_heat_w)[sunny].sum())

    assert sunny_heat(22.0) > 1.5 * sunny_heat(20.3)


def _cold_day(steps: int = 96, **overrides):
    hour = (np.arange(steps) * 0.25 + 6.0) % 24.0
    defaults = dict(
        solar_forecast_w=[0.0] * steps,
        target_temperature_top=(10.0,) * steps,
        outdoor_temperature_forecast=(5.0,) * steps,
        zone_state=_zone_state(20.1, 20.2),
        zone_target_temperature=tuple(np.where(hour < 22.0, 20.0, 18.0)),
        zone_maximum_temperature=(21.5,) * steps,
        zone_internal_gain_w=(150.0,) * steps,
    )
    defaults.update(overrides)
    defaults.setdefault("zone_minimum_temperature", defaults["zone_target_temperature"])

    return _make_input(**defaults)


TWO_NODE = BuildingThermalModel(
    ua_envelope_w_per_k=80.0,
    ua_air_mass_w_per_k=650.0,
    c_air_j_per_k=2.6e6,
    c_mass_j_per_k=35.0e6,
    a_eff_m2=3.0,
    sensor_mass_fraction=0.5,
    internal_gain_fraction=0.6,
    # Both rooms alike and closely coupled: these tests are about the plan,
    # not how the zone divides.
    a_eff_rest_m2=0.0,
    ua_rooms_w_per_k=500.0,
    living_envelope_fraction=0.5,
    living_area_fraction=0.5,
    closed_shutter_gain_fraction=0.0,
)


def test_space_runs_are_as_long_and_as_strong_as_the_heat_pump_makes_them():
    """Once its heating runs are known, the plan decides when the zone is heated
    but not how much: the heat follows from the curve and the floor, and a run
    lasts at least the compressor's own minimum runtime."""

    space = FloorCircuitModel(
        supply_at_zero_outdoor_c=28.0,
        supply_per_outdoor_k=-0.4,
        conductance_w_per_k=400.0,
    )
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
        floor_circuit_model=space,
    ).solve(_cold_day())

    on = list(result.space_schedule)
    heat = np.asarray(result.space_heat_w)

    assert any(on), "expected the zone to be heated"
    # What a floor near room temperature takes from the curve's 26 degC water
    # (400 W/K over about 6 K), not the compressor's full output.
    assert heat.max() < 400.0 * (26.0 - 19.5)
    assert heat.max() < THERMAL_MODEL.q_in_nominal_w

    runs = []
    for k, value in enumerate(on):
        if value and (k == 0 or not on[k - 1]):
            runs.append(1)
        elif value:
            runs[-1] += 1

    # A run cut off by the end of the horizon may be shorter.
    assert all(
        length >= MPCConfig().compressor_min_runtime_steps for length in runs[:-1]
    )


def test_a_run_into_a_screed_falls_off_as_it_warms():
    """On its own curve the heat pump holds its supply; the water heats the
    screed first, so a run delivers less each step as that warms."""

    building = replace(
        TWO_NODE, screed_capacity_j_per_k=4.0e6, screed_to_slab_w_per_k=800.0
    )
    space = FloorCircuitModel(
        supply_at_zero_outdoor_c=28.0,
        supply_per_outdoor_k=0.0,
        conductance_w_per_k=600.0,
    )
    day = _cold_day()
    # The screeds at their slabs' temperature, nothing having heated them.
    day = replace(day, zone_state=day.zone_state + day.zone_state[2:4])
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=building,
        floor_circuit_model=space,
    ).solve(day)

    on = list(result.space_schedule)
    heat = np.asarray(result.space_heat_w)
    first = on.index(1)
    last = first

    while last + 1 < len(on) and on[last + 1] and heat[last + 1] > 0.0:
        last += 1

    # From its second step: the first is the horizon's first, a start part of
    # a quarter hour.
    assert last - first >= 3, "expected a run of several steps"
    assert np.all(np.diff(heat[first + 1 : last + 1]) <= 1e-6)
    assert heat[last] < heat[first + 1] - 300.0


HEATING = FloorCircuitModel(
    supply_at_zero_outdoor_c=28.0,
    supply_per_outdoor_k=-0.4,
    conductance_w_per_k=400.0,
    min_heat_w=1500.0,
)


def _heat(space_cop_model=COP_MODEL):
    return MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
        floor_circuit_model=HEATING,
        space_cop_model=space_cop_model,
    ).solve(_cold_day(steps=48))


def test_a_plan_takes_the_envelope_of_each_step():
    """An envelope given per step (MPCInput.zone_envelope_w_per_k) is planned
    as a building with that envelope."""

    leakier = TWO_NODE.ua_envelope_w_per_k + 40.0

    def plan(building, **overrides):
        return MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(),
            cop_model=COP_MODEL,
            building_model=building,
            floor_circuit_model=HEATING,
            space_cop_model=COP_MODEL,
        ).solve(_cold_day(steps=24, **overrides))

    scheduled = plan(TWO_NODE, zone_envelope_w_per_k=(leakier,) * 24)
    constant = plan(replace(TWO_NODE, ua_envelope_w_per_k=leakier))

    assert scheduled.zone_temperatures == pytest.approx(constant.zone_temperatures)
    assert scheduled.space_heat_w == pytest.approx(constant.space_heat_w, abs=1.0)


def test_a_plan_through_the_living_rooms_loops_alone_heats_its_floor():
    """With the rest's loops closed (MPCInput.zone_floor_open) the plan's floor
    heat lands under the living room alone, and the circuit passes what the
    open pipes do at the flow through them: the supply is that floor plus
    Q / G."""

    share = TWO_NODE.living_area_fraction
    circuit = replace(HEATING, pipe_ua_w_per_k=480.0, flow_lpm=18.0)
    data = _cold_day(steps=48, zone_floor_open=(share, 0.0), zone_floor_flow_lpm=16.0)
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
        floor_circuit_model=circuit,
        space_cop_model=COP_MODEL,
    ).solve(data)
    conductance = floor_circuit_conductance_w_per_k(circuit, share, 16.0)
    heat = np.asarray(result.space_heat_w)
    living_floor = np.asarray(result.zone_mass_temperatures)
    on = heat > 1.0

    assert on.any(), "expected the zone to be heated"
    assert np.asarray(result.space_supply_c)[on] == pytest.approx(
        living_floor[on] + heat[on] / conductance
    )

    a_d, b_d = discretize_zoh(*zone_state_space(TWO_NODE), 900.0)
    state = np.array(data.zone_state)

    for k in range(len(heat) - 1):
        state = a_d @ state + b_d @ np.array(
            [5.0, 150.0, 0.0, 0.0, heat[k], (1.0 - share) * heat[k]]
        )
        assert living_floor[k + 1] == pytest.approx(state[2], abs=1e-4)


def test_a_run_the_zone_opens_draws_the_starts_measured_power():
    """The compressor's start-up is the same in every mode, so a run the zone
    opens draws the start's measured power over its first step, as the tank's
    does - its running draw above the least heat's on top - and from the next
    step on its running draw."""

    start_w = 900.0
    cop = replace(COP_MODEL, start_step_power_w=start_w)
    data = _cold_day(steps=48)
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
        floor_circuit_model=HEATING,
        space_cop_model=cop,
    ).solve(data)
    on = list(result.space_schedule)
    first = on.index(1)

    before = (
        result.schedule[first - 1]
        if first
        else data.boiler_on_current or data.space_on_current
    )

    assert not before, "expected the zone to start the compressor"
    assert on[first + 1], "expected the run to go on"

    def draw(k: int, heat_w: float, supply_c: float) -> float:
        return heat_w / float(
            cop.clamped_cop(data.outdoor_temperature_forecast[k], supply_c)
        )

    heat = result.space_heat_w
    supply = result.space_supply_c
    # The floor the water delivers against, nothing heated before this run.
    sink_c = supply[first] - heat[first] / HEATING.conductance_w_per_k
    least_w = HEATING.min_heat_w
    expected_w = (
        start_w
        + draw(first, heat[first], supply[first])
        - draw(first, least_w, sink_c + least_w / HEATING.conductance_w_per_k)
    )

    assert result.space_electrical_w[first] == pytest.approx(expected_w, rel=0.02)
    assert result.space_electrical_w[first + 1] == pytest.approx(
        draw(first + 1, heat[first + 1], supply[first + 1]), rel=0.02
    )


def test_a_runs_heat_never_rises():
    """The plan chooses the supply, and with it the heat - but within a run
    only ever holds or lowers it, as the heat pump at its setpoint does."""

    result = _heat()
    on = list(result.space_schedule)
    heat = np.asarray(result.space_heat_w)

    assert any(on), "expected the zone to be heated"

    for k in range(len(on) - 1):
        if on[k] and on[k + 1]:
            assert heat[k + 1] <= heat[k] + 1e-6


def test_the_floor_is_planned_no_further_than_its_runs_have_gone():
    """The zone's heat stays within the most the floor's runs have moved, the
    range its COP model was fitted in, though the compressor gives more."""

    capped = replace(HEATING, max_heat_w=2000.0)
    free = np.asarray(_heat().space_heat_w)
    heat = np.asarray(
        MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(),
            cop_model=COP_MODEL,
            building_model=TWO_NODE,
            floor_circuit_model=capped,
            space_cop_model=COP_MODEL,
        )
        .solve(_cold_day(steps=48))
        .space_heat_w
    )

    assert free.max() > capped.max_heat_w
    assert heat.max() <= capped.max_heat_w + 1e-6


def test_the_plan_chooses_the_heating_supply():
    """Holding the supply it is given, the heat pump heats the floor as the plan
    chooses: never less than its least, drawing the heat over the COP at the
    supply that takes, T_mass + Q / G - and a run's supply never falls, which
    would stop the compressor."""

    result = _heat()
    heat = np.asarray(result.space_heat_w)
    on = np.asarray(result.space_schedule, dtype=bool)
    supply = np.asarray(result.space_supply_c)

    assert on.any(), "expected the zone to be heated"
    assert (heat[on] >= HEATING.min_heat_w - 1e-3).all()
    assert heat[~on] == pytest.approx(0.0, abs=1e-6)

    power = np.asarray(result.space_electrical_w)
    cop = COP_MODEL.clamped_cop(5.0, supply[on])
    assert power[on] == pytest.approx(heat[on] / cop, rel=0.01)

    for k in range(len(on) - 1):
        if on[k] and on[k + 1]:
            assert supply[k + 1] >= supply[k] - 1e-6


def test_without_a_heating_cop_model_the_plan_keeps_to_the_curve():
    """A flat COP would price a warmer supply the same as a cooler one, so
    without the heating model the plan heats on the curve's supply."""

    result = _heat(space_cop_model=None)
    on = np.asarray(result.space_schedule, dtype=bool)

    assert on.any(), "expected the zone to be heated"
    assert np.asarray(result.space_supply_c)[on] == pytest.approx(HEATING.supply_c(5.0))


def _mild_day(below_k: float, sun_w: float = 0.0, weight: float = 0.5):
    """A mild day begun just under the target, with or without sun through
    the glazing from 10:00 to 16:00, planned with a band below_k deep."""

    steps = 96
    hour = (np.arange(steps) * 0.25 + 6.0) % 24.0
    target = np.where(hour < 22.0, 20.0, 18.0)
    data = _cold_day(
        steps,
        outdoor_temperature_forecast=(12.0,) * steps,
        zone_state=_zone_state(19.6, 19.8),
        zone_minimum_temperature=tuple(target - below_k),
        zone_living_gain_w=tuple(np.where((hour >= 10.0) & (hour < 16.0), sun_w, 0.0)),
    )
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(weight_comfort_eur_per_k2h=weight),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
    ).solve(data)

    return hour, target, result


def test_a_band_lets_the_morning_dip_that_the_sun_brings_back():
    """Held to its target, the zone is heated in the morning; with a band
    below it the plan lets the morning dip go where the sun brings the zone
    back - never past the band's edge."""

    hour, target, exact = _mild_day(0.0, sun_w=2000.0)
    _, _, banded = _mild_day(1.0, sun_w=2000.0)
    morning = hour < 10.0

    assert np.asarray(exact.space_heat_w)[morning].sum() > 0.0
    assert np.asarray(banded.space_heat_w)[morning].sum() == pytest.approx(0.0)
    assert sum(banded.space_heat_w) < sum(exact.space_heat_w)
    assert (np.asarray(banded.zone_temperatures) >= target - 1.0 - 1e-6).all()


def test_the_comfort_weight_holds_the_zone_near_its_target():
    """Without a price on comfort the plan lets the zone sit at the band's
    edge; with one, a day without sun is held near the target - the dip is
    squared, so most of it costs more than the heat that prevents it."""

    _, target, free = _mild_day(1.0, weight=0.0)
    _, _, priced = _mild_day(1.0)

    def shortfall(result):
        return np.maximum(target - np.asarray(result.zone_temperatures), 0.0)

    assert shortfall(free).max() == pytest.approx(1.0, abs=0.01)
    # No more than the zone starts below its target.
    assert shortfall(priced).max() <= 0.4 + 1e-6
    assert shortfall(priced).mean() < 0.25 * shortfall(free).mean()


def test_cooling_is_free_below_its_target_and_priced_above():
    """The mirror of heating: a zone above its wanted temperature is cooled
    towards it - priced, rather than left at the band's edge - but not past
    it, which costs energy and buys no comfort."""

    def zone(weight: float) -> np.ndarray:
        result = MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(weight_comfort_eur_per_k2h=weight),
            cop_model=COP_MODEL,
            building_model=TWO_NODE,
            floor_circuit_model=COOLING,
            space_cop_model=replace(COP_MODEL, cooling=True),
        ).solve(
            _hot_day(
                zone_state=_zone_state(25.0, 24.5),
                zone_target_temperature=(23.0,) * HOT_DAY_STEPS,
                zone_minimum_temperature=(20.0,) * HOT_DAY_STEPS,
                zone_maximum_temperature=(25.0,) * HOT_DAY_STEPS,
            )
        )

        return np.asarray(result.zone_temperatures)

    free, priced = zone(0.0), zone(0.5)

    assert free.max() <= 25.0 + 1e-6
    assert priced.mean() < free.mean() - 1.0
    assert priced.min() >= 23.0 - 0.05


def test_space_heating_is_costed_with_the_heating_cop_once_there_is_one():
    """The heat pump's own heating runs, not its hot water ones, say what a
    floor run costs - but the hot water fit stands in until there are any."""

    heating_cop = replace(COP_MODEL, eta_carnot=COP_MODEL.eta_carnot / 2.0)

    def space_cop(**models):
        optimizer = MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(),
            cop_model=COP_MODEL,
            building_model=TWO_NODE,
            **models,
        )

        return optimizer._space_cop(5.0)

    assert space_cop(space_cop_model=heating_cop) < space_cop()
    assert space_cop() == pytest.approx(
        COP_MODEL.clamped_cop(5.0, MPCOptimizer.SPACE_HEATING_SUPPLY_C)
    )


def test_net_metering_does_not_heat_today_for_a_cloudy_tomorrow():
    """Under net metering own sun earns what grid power costs, so heat stored
    today for a cloudier tomorrow is paid in full today and partly lost by
    tomorrow. Priced as free, the sun made exactly that worth doing."""

    # Today and tomorrow morning: longer only slowed the solve.
    steps = 144
    hour = np.arange(steps) * 0.25
    sun = np.clip(2500.0 * np.sin(np.pi * ((hour % 24.0) - 7.0) / 12.0), 0.0, None)
    # Too little sun tomorrow to heat on: with 30% of today's, it still
    # covered tomorrow's run too, heating on either day was free, and which of
    # the two equal plans came out was up to the solver.
    sun[hour >= 24.0] *= 0.05
    target = [10.0] * steps
    target[18 * 4] = 45.0
    target[34 * 4] = 45.0
    # 5 kWh drawn tomorrow at 7:00, more than even a full tank holds above
    # tomorrow's target: tomorrow runs either way, so heat stored today saves
    # no start, only grid power. Without it, today's tank held tomorrow's
    # target by itself, storing more saved nothing, and which of the two equal
    # plans came out was up to the solver's version (it differed on CI).
    tap = [0.0] * steps
    tap[31 * 4 : 32 * 4] = [5000.0] * 4

    def run_end(config: MPCConfig) -> float:
        result = MPCOptimizer(THERMAL_MODEL, config, cop_model=COP_MODEL).solve(
            _make_input(
                solar_forecast_w=list(sun),
                target_temperature_top=tuple(target),
                outdoor_temperature_forecast=(15.0,) * steps,
                tap_forecast_w=tuple(tap),
            )
        )
        on = [k for k, value in enumerate(result.schedule[:96]) if value]
        return result.temperatures[on[-1] + 1]

    # Net metering: an exported kWh credited at the import price. At the
    # default export price instead own sun costs less than a third of grid
    # power, and storing today's sun for tomorrow is then really cheaper.
    net_metering = MPCConfig(
        export_price_eur_per_kwh=MPCConfig().import_price_eur_per_kwh
    )

    assert run_end(net_metering) < run_end(NO_EXPORT) - 0.5


def test_a_measured_supply_margin_does_not_depend_on_the_targets():
    """A 60 degC legionella target in the horizon must not change what a normal
    run costs: how much hotter the supply runs than the tank is the plate
    exchanger's property, not the target's."""

    optimizer = MPCOptimizer(
        replace(THERMAL_MODEL, supply_margin_k=8.0), MPCConfig(), cop_model=COP_MODEL
    )

    assert optimizer._power_line_coefficients(
        10.0, 45.0
    ) == optimizer._power_line_coefficients(10.0, 60.0)


def test_the_solver_log_stays_out_of_info(caplog):
    """HiGHS's progress lines are for debugging: at INFO only the plan's own
    lines should show."""

    with caplog.at_level("INFO"):
        MPCOptimizer(THERMAL_MODEL, MPCConfig()).solve(_make_input())

    assert not [r for r in caplog.records if r.name.startswith("pyomo")]


# A compressor ramping up over its first minutes (see _ramp_fraction).
STARTING_MODEL = replace(THERMAL_MODEL, q_in_ramp_seconds=420.0, supply_margin_k=9.6)
# The electrical power it draws meanwhile, measured (see
# HeatPumpCOPModel.start_step_power_w).
STARTING_COP_MODEL = replace(COP_MODEL, start_step_power_w=750.0)


@pytest.mark.parametrize(
    ("thermal_model", "cop_model"),
    [(THERMAL_MODEL, COP_MODEL), (STARTING_MODEL, STARTING_COP_MODEL)],
)
def test_the_reported_energy_is_what_the_objective_priced(thermal_model, cop_model):
    """The electricity and its grid part a result reports cost exactly the
    objective, less its starts: a plan log built on them ranks plans the way the
    optimizer does. Also with a start step drawing its measured power."""

    config = MPCConfig()
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    result = MPCOptimizer(thermal_model, config, cop_model=cop_model).solve(
        _make_input(
            target_temperature_top=tuple(target),
            outdoor_temperature_forecast=(5.0,) * len(SOLAR_FORECAST_W),
        )
    )
    starts = sum(
        1
        for k, on in enumerate(result.schedule)
        if on and (k == 0 or not result.schedule[k - 1])
    )
    energy_eur = (
        config.import_price_eur_per_kwh - config.export_price_eur_per_kwh
    ) * result.grid_kwh + config.export_price_eur_per_kwh * result.electricity_kwh

    assert result.electricity_kwh > 0.0
    assert energy_eur + config.weight_switching * starts == pytest.approx(
        result.objective_value, abs=1e-6
    )


def test_a_runs_first_step_draws_its_measured_power():
    """The start step draws what runs were measured to draw there, not the
    power line over the ramp's heat; the step after it the power line again."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    optimizer = MPCOptimizer(STARTING_MODEL, MPCConfig(), cop_model=STARTING_COP_MODEL)
    result = optimizer.solve(
        _make_input(
            target_temperature_top=tuple(target),
            outdoor_temperature_forecast=(5.0,) * len(SOLAR_FORECAST_W),
        )
    )
    alpha, beta = optimizer._power_line_coefficients(5.0, max(target))
    first = result.schedule.index(1)
    second_w = (alpha + beta * result.temperatures[first + 1]) * (
        result.heat_w[first + 1] / optimizer._heat_w
    )

    assert result.electrical_power_w[first] == pytest.approx(750.0)
    assert result.electrical_power_w[first + 1] == pytest.approx(second_w)


OVERSHOOT_MODEL = replace(THERMAL_MODEL, setpoint_overshoot_k=1.75)


def _run_end(result) -> float:
    """The tank after the first planned run."""

    first = result.schedule.index(1)
    end = first + result.schedule[first:].index(0)

    return result.temperatures[end]


def test_a_planned_run_ends_its_setpoint_overshoot_above_what_the_target_needs():
    """The heat pump stops on its setpoint and the tank settles above it: a run
    planned for a 45 degC target ends that much higher, as it really does."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    data = _make_input(target_temperature_top=tuple(target))

    without = MPCOptimizer(THERMAL_MODEL, MPCConfig()).solve(data)
    with_overshoot = MPCOptimizer(OVERSHOOT_MODEL, MPCConfig()).solve(data)

    assert with_overshoot.temperatures[18] >= 45.0 + 1.75 - 1e-6
    assert _run_end(with_overshoot) == pytest.approx(_run_end(without) + 1.75, abs=0.3)


def test_heat_already_in_the_tank_counts_in_full():
    """A tank that holds its target with less than the overshoot to spare is
    not topped up: the overshoot only applies to the end of a run the plan
    makes."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[6] = 45.0
    data = _make_input(
        target_temperature_top=tuple(target),
        current_temp_top=45.6,
        current_temp_bottom=45.6,
    )

    result = MPCOptimizer(OVERSHOOT_MODEL, MPCConfig()).solve(data)

    assert not any(result.schedule)


def test_a_colder_day_heats_the_tank_slower():
    """The heat pump gives less heat the colder the outdoor air, so the same
    target takes more of the plan's run on a cold day."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0
    cop_model = replace(
        COP_MODEL, q_in_at_zero_outdoor_w=3000.0, q_in_per_outdoor_w_per_k=60.0
    )

    def run_steps(outdoor_c: float) -> int:
        return sum(
            MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=cop_model)
            .solve(
                _make_input(
                    target_temperature_top=tuple(target),
                    outdoor_temperature_forecast=(outdoor_c,) * len(SOLAR_FORECAST_W),
                )
            )
            .schedule
        )

    assert run_steps(0.0) > run_steps(20.0)


def test_a_run_under_way_starts_from_the_heat_it_has_put_in():
    """Half an hour into a run the sensors lag the tank: the plan starts from
    the tank mixed before the run plus the run's own heat since, ramp and
    standing loss included - not from the sensors."""

    model = replace(STARTING_MODEL, heat_pump_max_tank_temperature_c=55.0)
    optimizer = MPCOptimizer(model, MPCConfig())
    data = _make_input(
        current_temp_top=42.0,
        current_temp_bottom=41.5,
        boiler_on_current=True,
        compressor_elapsed_hours=0.5,
        run_start_temp_top=34.0,
        run_start_temp_bottom=34.0,
    )
    a_d, b_d = discretize_zoh(
        *lumped_tank_state_space(
            model.volume_l, model.ua_top_w_per_k + model.ua_bottom_w_per_k
        ),
        1800.0,
    )
    heat_w = model.q_in_nominal_w * (1.0 - 420.0 / (2.0 * 1800.0))
    expected_c = a_d[0, 0] * 34.0 + b_d[0, 0] * 20.0 + b_d[0, 1] * heat_w

    assert optimizer._initial_temperature(data) == pytest.approx(expected_c)
    assert optimizer._initial_temperature(
        replace(data, run_start_temp_top=None, run_start_temp_bottom=None)
    ) == pytest.approx(41.75)
    assert optimizer._initial_temperature(
        replace(data, compressor_elapsed_hours=5.0)
    ) == pytest.approx(55.0)


def test_a_run_under_way_is_traced_by_the_same_heat_from_its_start():
    """Through the run, the same balance: the tank before it at its start,
    rising with the ramp, up to the plan's own start now."""

    optimizer = MPCOptimizer(STARTING_MODEL, MPCConfig())
    data = _make_input(
        boiler_on_current=True,
        compressor_elapsed_hours=0.5,
        run_start_temp_top=34.0,
        run_start_temp_bottom=34.0,
    )
    traced = [
        optimizer.run_temperature(data, hours) for hours in (0.0, 0.05, 0.25, 0.5)
    ]

    assert traced[0] == pytest.approx(34.0)
    assert traced == sorted(traced)
    # Slower within the ramp than after it.
    assert (traced[1] - traced[0]) / 0.05 < (traced[3] - traced[2]) / 0.25
    assert traced[-1] == pytest.approx(optimizer._initial_temperature(data))
    assert (
        optimizer.run_temperature(replace(data, boiler_on_current=False), 0.25) is None
    )


def test_a_run_under_way_is_traced_by_its_measured_heat_where_there_is_one():
    """The heat its calorimetry measured, times the calorimetric gain, in place
    of the plan's ramp: 7 kW for half an hour, over the tank's own loss."""

    model = replace(STARTING_MODEL, calorimetric_gain=1.1)
    optimizer = MPCOptimizer(model, MPCConfig())
    data = _make_input(
        boiler_on_current=True,
        compressor_elapsed_hours=0.5,
        run_start_temp_top=34.0,
        run_start_temp_bottom=34.0,
    )
    a_d, b_d = discretize_zoh(
        *lumped_tank_state_space(
            model.volume_l, model.ua_top_w_per_k + model.ua_bottom_w_per_k
        ),
        1800.0,
    )

    assert optimizer.run_temperature(data, 0.5, 7000.0 * 1800.0) == pytest.approx(
        a_d[0, 0] * 34.0 + b_d[0, 0] * 20.0 + b_d[0, 1] * 7000.0 * 1.1
    )
    assert optimizer.run_temperature(data, 0.0, 0.0) == pytest.approx(34.0)


def _hot_water_soon(due: int = 4, **overrides) -> MPCInput:
    """A cold tank wanted hot at step `due`, too soon to wait."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[due] = 45.0

    return _make_input(
        current_temp_top=30.0,
        current_temp_bottom=30.0,
        target_temperature_top=tuple(target),
        idle_elapsed_hours=5.0,
        **overrides,
    )


def test_a_plan_made_mid_quarter_heats_only_what_is_left_of_it():
    """Ten minutes into a quarter the tank already holds that quarter's heat
    so far; the plan's first step is the five minutes left, so the tank at the
    quarter's end is the tank now plus five minutes of heat - not a quarter's."""

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    data = _hot_water_soon(
        boiler_on_current=True,
        compressor_elapsed_hours=1.0,
        first_step_hours=5.0 / 60.0,
    )
    result = optimizer.solve(data)

    a_d, b_d = discretize_zoh(
        *lumped_tank_state_space(
            THERMAL_MODEL.volume_l,
            THERMAL_MODEL.ua_top_w_per_k + THERMAL_MODEL.ua_bottom_w_per_k,
        ),
        300.0,
    )
    expected_c = (
        a_d[0, 0] * 30.0 + b_d[0, 0] * 20.0 + b_d[0, 1] * THERMAL_MODEL.q_in_nominal_w
    )

    assert result.schedule[0] == 1
    assert result.temperatures[0] == pytest.approx(30.0)
    assert result.temperatures[1] == pytest.approx(expected_c, abs=0.01)


def test_a_run_started_mid_quarter_ramps_up_in_what_is_left_of_it():
    """Started ten minutes into a quarter, a run has five minutes of it left,
    all of them on its 300 s ramp: half the steady heat over that part."""

    ramped = replace(THERMAL_MODEL, q_in_steady_w=6000.0, q_in_ramp_seconds=300.0)
    result = MPCOptimizer(ramped, MPCConfig()).solve(
        _hot_water_soon(due=2, first_step_hours=5.0 / 60.0)
    )

    assert result.schedule[0] == 1
    assert result.heat_w[0] == pytest.approx(3000.0, rel=0.02)
    assert result.heat_w[1] == pytest.approx(6000.0, rel=0.02)


HOT_DAY_STEPS = 48


def _hot_day(**overrides):
    """Twelve hours at 30 degC outdoors, a zone above its ceiling."""

    steps = HOT_DAY_STEPS
    defaults = dict(
        solar_forecast_w=[0.0] * steps,
        target_temperature_top=(10.0,) * steps,
        outdoor_temperature_forecast=(30.0,) * steps,
        zone_state=_zone_state(24.0, 23.5),
        zone_target_temperature=(22.5,) * steps,
        zone_minimum_temperature=(18.0,) * steps,
        zone_maximum_temperature=(22.5,) * steps,
        zone_internal_gain_w=(150.0,) * steps,
        zone_cooling=True,
    )
    defaults.update(overrides)

    return _make_input(**defaults)


COOLING = FloorCircuitModel(
    supply_at_zero_outdoor_c=16.0,
    supply_per_outdoor_k=-0.1,
    conductance_w_per_k=719.0,
    min_heat_w=3500.0,
)


def _cool(data):
    return MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
        floor_circuit_model=COOLING,
        space_cop_model=replace(COP_MODEL, cooling=True),
    ).solve(data)


def test_cooling_keeps_the_floor_above_the_dew_point():
    """A warm zone is cooled, never by less than the compressor's least, and
    the floor's mass - supply less Q / G - stays above the dew point bound."""

    result = _cool(_hot_day(zone_mass_minimum_c=(21.0,) * HOT_DAY_STEPS))
    heat = np.asarray(result.space_heat_w)
    on = np.asarray(result.space_schedule, dtype=bool)

    assert on.any(), "expected the zone to be cooled"
    assert (heat <= 1e-6).all()
    assert (-heat[on] >= COOLING.min_heat_w - 1e-3).all()
    # What it draws is positive, and less than the heat it moves (EER > 1).
    power = np.asarray(result.space_electrical_w)
    assert (power[on] > 0.0).all()
    assert (power[on] < -heat[on]).all()
    # The heat over the EER at the supply it runs at - colder water for more
    # cooling costing more per kWh of it.
    eer = replace(COP_MODEL, cooling=True).clamped_cop(
        30.0, np.asarray(result.space_supply_c)[on]
    )
    assert power[on] == pytest.approx(-heat[on] / eer, rel=0.01)
    assert power[~on] == pytest.approx(0.0, abs=1e-6)

    supply = np.asarray(result.space_supply_c)[on]
    mass = supply - heat[on] / COOLING.conductance_w_per_k
    assert (mass >= 21.0 - 1e-6).all()


def test_a_cooling_run_takes_its_ramp_from_the_floor_in_the_step_it_starts():
    """The compressor needs its ramp to reach the heat it runs at, so the
    quarter a run starts in takes that heat less the ramp's shortfall from the
    floor: 1 - 530 / 1800 of it."""

    ramp_s = 530.0
    data = _hot_day()
    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE,
        floor_circuit_model=COOLING,
        space_cop_model=replace(COP_MODEL, cooling=True, start_ramp_seconds=ramp_s),
    ).solve(data)

    assert result.space_schedule[0] == 1, "expected a run to start now"

    a_d, b_d = discretize_zoh(*zone_state_space(TWO_NODE), 900.0)

    def zone_after(floor_w: float) -> float:
        state = a_d @ np.array(data.zone_state) + b_d @ np.array(
            [30.0, 150.0, 0.0, 0.0, floor_w, 0.0]
        )

        return float(zone_observation(TWO_NODE) @ state)

    heat_w = result.space_heat_w[0]
    share = 1.0 - ramp_s / (2.0 * 900.0)

    assert result.zone_temperatures[1] == pytest.approx(
        zone_after(share * heat_w), abs=1e-4
    )
    assert abs(zone_after(heat_w) - zone_after(share * heat_w)) > 1e-2


def test_a_day_is_cooled_before_its_hot_water():
    """Hot water warms the attic, so a day's cooling comes first: no cooling
    after the tank is heated that day, and none at all on a day it already
    was - until local midnight."""

    half = HOT_DAY_STEPS // 2
    target = [10.0] * HOT_DAY_STEPS
    target[half - 1] = 45.0
    result = _cool(
        _hot_day(
            target_temperature_top=tuple(target),
            zone_local_day=(0,) * HOT_DAY_STEPS,
        )
    )
    tank = np.flatnonzero(result.schedule)
    zone = np.flatnonzero(result.space_schedule)

    assert tank.size and zone.size, "expected both the tank and the zone served"
    assert zone.max() < tank.min()

    # Heated earlier today: no cooling before midnight, halfway, and after it.
    result = _cool(
        _hot_day(
            zone_local_day=(0,) * half + (1,) * half,
            dhw_earlier_today=True,
        )
    )
    zone = np.flatnonzero(result.space_schedule)

    assert zone.size, "expected the next day cooled"
    assert zone.min() >= half


def test_the_floor_follows_a_dew_point_forecast_step_by_step():
    """A dew point forecast to rise halfway holds the mass above each step's
    own bound: low early, high late."""

    half = HOT_DAY_STEPS // 2
    bound = (20.0,) * half + (23.5,) * half
    result = _cool(_hot_day(zone_mass_minimum_c=bound))
    heat = np.asarray(result.space_heat_w)
    on = np.asarray(result.space_schedule, dtype=bool)

    assert on.any(), "expected the zone to be cooled"

    mass = np.asarray(result.space_supply_c) - heat / COOLING.conductance_w_per_k
    assert (mass[on] >= np.asarray(bound)[on] - 1e-6).all()


def test_the_supply_never_rises_within_a_cooling_run():
    """Raising the setpoint over the loop's water stops the compressor, so a
    run cools at a falling or steady supply. Free to choose each step's heat,
    the plan otherwise ran at the dew point bound, then at its least - the
    supply stepping up (19.07 to 19.15 degC here, 15.2 to 16.2 in a real
    plan)."""

    result = _cool(
        _hot_day(
            zone_mass_minimum_c=(15.0,) * HOT_DAY_STEPS,
            zone_supply_minimum_c=(19.0,) * HOT_DAY_STEPS,
        )
    )
    on = np.asarray(result.space_schedule, dtype=bool)
    supply = np.asarray(result.space_supply_c)

    assert on.any(), "expected the zone to be cooled"

    for k in range(len(on) - 1):
        if on[k] and on[k + 1]:
            assert supply[k + 1] <= supply[k] + 1e-6


def test_uninsulated_pipes_rule_out_a_supply_below_the_dew_point():
    """With the supply held above a high dew point, the floor cannot take the
    compressor's least - at most 719 W/K over the ~3.4 K from a mass that
    warms to ~24.4 degC - so no supply does both and the zone is not cooled."""

    result = _cool(
        _hot_day(
            zone_mass_minimum_c=(19.0,) * HOT_DAY_STEPS,
            zone_supply_minimum_c=(21.0,) * HOT_DAY_STEPS,
        )
    )

    assert not any(result.space_schedule)


def test_a_run_the_zone_opens_is_not_charged_the_tanks_ramp():
    """The start-up ramp is the tank's: a compressor run that begins on the
    floor, with no hot water wanted, can still be planned."""

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=replace(COP_MODEL, start_ramp_seconds=800.0),
        building_model=TWO_NODE,
    ).solve(_cold_day())

    assert any(result.space_schedule), "expected the zone to be heated"
    assert not any(result.schedule)


def test_without_sun_the_run_goes_to_the_low_tariff():
    """No sun, a 45 degC target at the end, and a low tariff for the first
    three hours: heating then and holding the heat is cheaper than heating
    just in time at the high one."""

    horizon = len(SOLAR_FORECAST_W)
    target = [10.0] * horizon
    target[-1] = 45.0
    low_steps = horizon // 2

    result = MPCOptimizer(THERMAL_MODEL, NO_EXPORT).solve(
        _make_input(
            solar_forecast_w=[0.0] * horizon,
            target_temperature_top=tuple(target),
            import_price_eur_per_kwh=(0.10,) * low_steps
            + (0.40,) * (horizon - low_steps),
        )
    )

    on_steps = [k for k, on in enumerate(result.schedule) if on]

    assert on_steps
    assert max(on_steps) < low_steps


def test_the_surplus_scenarios_combine_the_solar_and_baseload_bands():
    """Little sun meeting much load sets the low end, the spreads of the two
    independent errors adding in quadrature per side; without a band there is
    one scenario."""

    data = _make_input(
        solar_forecast_w=[1000.0, 100.0],
        solar_p10_w=(800.0, 50.0),
        solar_p90_w=(1200.0, 150.0),
        baseload_forecast_w=(200.0, 200.0),
        baseload_p10_w=(150.0, 150.0),
        baseload_p90_w=(500.0, 500.0),
        target_temperature_top=(10.0, 10.0),
    )

    (w_low, low), (w_mid, mid), (w_high, high) = surplus_scenarios(data)

    assert (w_low, w_mid, w_high) == (0.3, 0.4, 0.3)
    assert mid == pytest.approx([800.0, 0.0])
    assert low == pytest.approx([800.0 - np.hypot(200.0, 300.0), 0.0])
    assert high == pytest.approx([800.0 + np.hypot(200.0, 50.0), 0.0])

    [(weight, only)] = surplus_scenarios(
        replace(
            data, solar_p10_w=(), solar_p90_w=(), baseload_p10_w=(), baseload_p90_w=()
        )
    )

    assert weight == 1.0 and only == pytest.approx([800.0, 0.0])


TANK_BOOSTER_MODEL = replace(
    THERMAL_MODEL,
    heat_pump_max_tank_temperature_c=55.0,
    max_tank_temperature_c=60.0,
    booster_heat_w=2000.0,
    booster_electrical_w=2400.0,
)


def _program_cost(optimizer: MPCOptimizer, data, plan=None) -> float:
    """The tank's plan as the mixed integer program costs it: its own optimum,
    or with the heat pump's and the booster's on/off fixed to `plan`."""

    model = optimizer._build_model(data)

    if plan is not None:
        for k in model.K:
            model.boiler_on[k].fix(int(plan.heat_pump_w[k] > 0.0))

            if not model.booster_on[k].fixed:
                model.booster_on[k].fix(int(plan.booster_w[k] > 0.0))

    solver = Highs()
    solver.highs_options = {"mip_abs_gap": MIP_ABSOLUTE_GAP_EUR, "mip_rel_gap": 0.0}
    solver.config.load_solution = False

    return solver.solve(model).best_feasible_objective


@pytest.mark.parametrize(
    ("thermal_model", "target_c"),
    [(THERMAL_MODEL, 45.0), (TANK_BOOSTER_MODEL, 59.0)],
    ids=["heat pump", "booster"],
)
def test_the_tanks_own_plan_is_the_programs_optimum(thermal_model, target_c):
    """Dynamic programming over the tank finds the plan the mixed integer
    program proves optimal: the same runs, costed the same by the program -
    with the sun, the power line and a start's ramp, and the booster finishing
    a run above the heat pump's limit."""

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[20] = target_c
    data = _make_input(
        target_temperature_top=tuple(target),
        outdoor_temperature_forecast=(5.0,) * len(SOLAR_FORECAST_W),
    )
    optimizer = MPCOptimizer(thermal_model, MPCConfig(), cop_model=COP_MODEL)
    plan = plan_tank(optimizer._tank_setting(data))

    assert any(plan.heat_pump_w)
    assert _program_cost(optimizer, data, plan) == pytest.approx(
        _program_cost(optimizer, data), abs=MIP_ABSOLUTE_GAP_EUR
    )


def test_taps_cannot_cool_the_tank_below_the_mains():
    """A draw replaces hot water with cold from the mains: forecast to draw far
    more than the tank holds, it ends at the mains' temperature, and from there
    the room warms it - not cooled on past it into the cold water's own."""

    horizon = len(SOLAR_FORECAST_W)
    heavy = [0.0] * horizon
    heavy[2:6] = [20000.0] * 4
    data = _make_input(
        target_temperature_top=(10.0,) * horizon,
        tap_forecast_w=tuple(heavy),
        mains_water_temperature=15.0,
    )
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    result = optimizer.solve(data)
    unbounded = optimizer.solve(replace(data, mains_water_temperature=None))

    assert not any(result.schedule)
    assert min(result.temperatures) == pytest.approx(15.0, abs=1e-6)
    assert result.temperatures[-1] > 15.0
    assert min(unbounded.temperatures) < 15.0


def test_a_tank_heated_to_its_target_has_the_draws_after_it():
    """The mains only hold back draws of a tank left unheated: emptied by an
    early draw and heated to a later target, the draw after that target is
    planned in full."""

    horizon = len(SOLAR_FORECAST_W)
    taps = [0.0] * horizon
    taps[2:6] = [20000.0] * 4
    taps[20] = 3000.0
    target = [10.0] * horizon
    target[18] = 45.0
    data = _make_input(
        target_temperature_top=tuple(target),
        tap_forecast_w=tuple(taps),
        mains_water_temperature=15.0,
    )
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    tap_w = optimizer._tank_setting(data).tap_w
    result = optimizer.solve(data)

    assert sum(tap_w[2:6]) < sum(taps[2:6])
    assert tap_w[20] == pytest.approx(3000.0)
    assert result.temperatures[21] < result.temperatures[20] - 1.0


def test_the_mains_follow_the_year_damped_and_late():
    """Coldest in late winter, warmest in late summer, a few degrees either
    side of the year's mean air - below the lowest the tank's bottom sensor
    read in the largest draws (Aug-Oct 2026: 17.5, 17.8, 16.5 degC)."""

    february, august = mains_water_temperature_c(46), mains_water_temperature_c(227)

    assert 9.0 < february < 11.5
    assert 16.0 < august < 17.5
    assert all(
        mains_water_temperature_c(day) < bound
        for day, bound in ((227, 17.5), (258, 17.8), (288, 16.5))
    )


# Fitted on Jun - Oct 2026 (see BoilerThermalIdentifier._identify_thermocline).
THERMOCLINE_MODEL = replace(
    THERMAL_MODEL,
    top_sensor_volume_l=139.0,
    bottom_sensor_volume_l=104.0,
    thermocline_width_l=34.0,
)


def test_the_sensors_hold_until_the_drawn_water_reaches_them():
    """A draw lets mains water in at the bottom: a mixed tank reads its
    temperature at both sensors, a fifth of its heat drawn the bottom has
    hardly moved, and the bottom goes before the top."""

    def read(mixed_c: float) -> tuple[float, float]:
        top_c, bottom_c = tank_sensor_temperatures(
            THERMOCLINE_MODEL, 48.0, mixed_c, 15.0
        )
        return float(top_c), float(bottom_c)

    # Within a sensor step: the front's tail reaches below the tank's bottom.
    assert read(48.0) == pytest.approx((48.0, 48.0), abs=TANK_SENSOR_RESOLUTION_K)
    assert read(48.0 - 33.0 / 5.0)[1] > 46.0
    top_c, bottom_c = read(48.0 - 33.0 * 0.6)
    assert bottom_c < 30.0 < top_c


def test_the_planned_sensors_start_at_their_readings_and_a_run_stirs_them():
    """The plan's sensors start from the hot water their readings show and
    read the tank itself once a run has stirred it."""

    top_c, bottom_c = (
        float(c) for c in tank_sensor_temperatures(THERMOCLINE_MODEL, 50.0, 30.0, 15.0)
    )
    data = _make_input(
        current_temp_top=top_c,
        current_temp_bottom=bottom_c,
        mains_water_temperature=15.0,
    )
    result = MPCResult(
        schedule=(0, 1, 0, 0),
        temperatures=(30.0, 29.9, 45.0, 44.9),
        electrical_power_w=(0.0,) * 4,
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        solver_status="ok",
        termination_condition="optimal",
    )

    top, bottom = MPCOptimizer(THERMOCLINE_MODEL, MPCConfig()).sensor_temperatures(
        data, result
    )

    assert (top[0], bottom[0]) == pytest.approx((top_c, bottom_c), abs=0.05)
    assert bottom[1] < top[1]
    assert (top[2], bottom[2]) == pytest.approx(
        (45.0, 45.0), abs=TANK_SENSOR_RESOLUTION_K
    )
    assert top[3] == pytest.approx(bottom[3], abs=0.1)


def test_after_a_draw_the_planned_sensors_start_at_both_readings():
    """A draw can leave a front thinner or thicker than a rest's, which one
    hot water fitted to both sensors cannot meet: with the hot water the
    last run left (MPCInput.hot_water_c) the plan starts at both readings,
    relaxes towards the rest's fit, and reads the tank once a run stirs it."""

    top_c, bottom_c = (
        float(c)
        for c in tank_sensor_temperatures(
            replace(THERMOCLINE_MODEL, thermocline_width_l=20.0), 48.0, 30.0, 15.0
        )
    )
    data = _make_input(
        current_temp_top=top_c,
        current_temp_bottom=bottom_c,
        mains_water_temperature=15.0,
        hot_water_c=48.0,
    )
    result = MPCResult(
        schedule=(0, 0, 0, 1, 0),
        # The plan's tank mixed from the sensors otherwise: a degree apart.
        temperatures=(29.0, 28.9, 28.8, 28.7, 45.0),
        electrical_power_w=(0.0,) * 5,
        heat_w=(0.0,) * 5,
        objective_value=0.0,
        solver_status="ok",
        termination_condition="optimal",
    )
    optimizer = MPCOptimizer(THERMOCLINE_MODEL, MPCConfig())

    top, bottom = optimizer.sensor_temperatures(data, result)
    one = optimizer.sensor_temperatures(replace(data, hot_water_c=None), result)

    assert (top[0], bottom[0]) == pytest.approx((top_c, bottom_c), abs=0.01)
    assert abs(one[0][0] - top_c) + abs(one[1][0] - bottom_c) > 0.5
    assert (top[4], bottom[4]) == pytest.approx(
        (45.0, 45.0), abs=TANK_SENSOR_RESOLUTION_K
    )


def test_no_planned_sensors_without_a_thermocline():
    result = MPCResult((0,), (30.0,), (0.0,), (0.0,), 0.0, "ok", "optimal")
    data = _make_input(mains_water_temperature=15.0)

    assert MPCOptimizer(THERMAL_MODEL, MPCConfig()).sensor_temperatures(
        data, result
    ) == ((), ())


def test_heat_left_in_the_floor_at_the_end_is_worth_what_it_saves(monkeypatch):
    """The heat a plan leaves in the zone at its end, up to what the zone
    left alone would fall short of its wanted temperature, is worth what the
    next plan would pay to make it: a plan ends with a warmer floor than one
    that counts it as nothing - and leaves none where the zone ends warm
    enough anyway (see test_heating_the_zone_is_not_free)."""

    def plan():
        return MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(),
            cop_model=COP_MODEL,
            building_model=TWO_NODE,
            floor_circuit_model=HEATING,
            space_cop_model=COP_MODEL,
        ).solve(_cold_day(steps=48))

    valued = plan()
    monkeypatch.setattr(
        optimizer_module,
        "zone_heat_capacities",
        lambda model: np.zeros(zone_size(model)),
    )
    worthless = plan()

    assert valued.zone_mass_temperatures[-1] > worthless.zone_mass_temperatures[-1]
