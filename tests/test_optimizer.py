import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from domain.dynamics import discretize_zoh
from domain.models import (
    BoilerThermalModel,
    BuildingLumpedModel,
    BuildingThermalModel,
    HeatPumpCOPModel,
)
from domain.mpc import MPCConfig, MPCInput
from domain.physics import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    lumped_tank_state_space,
)
from features.optimizer import MIP_ABSOLUTE_GAP_EUR, MPCOptimizer

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
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(boiler_min_runtime_steps=1))
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

    config = MPCConfig(boiler_min_runtime_steps=4)
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
            assert run_length >= config.boiler_min_runtime_steps or j == len(schedule)
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

    config = MPCConfig(heat_pump_min_off_steps=4, boiler_min_runtime_steps=1)
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

    config = MPCConfig(heat_pump_min_off_steps=4, boiler_min_runtime_steps=1)
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

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
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

    assert result.schedule[: config.boiler_min_runtime_steps] == (1, 1)


def test_a_run_past_its_minimum_runtime_may_stop():
    result = MPCOptimizer(THERMAL_MODEL, MPCConfig()).solve(_running_run_input(1.0))

    assert result.schedule[0] == 0


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
        THERMAL_MODEL, MPCConfig(boiler_min_runtime_steps=4), cop_model=COP_MODEL
    )
    result = optimizer.solve(data)

    on_steps = [k for k, v in enumerate(result.schedule) if v == 1]
    assert len(on_steps) >= 2
    assert on_steps == list(range(on_steps[0], on_steps[-1] + 1))  # one contiguous run

    on_steps_power = [result.electrical_power_w[k] for k in on_steps]
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


def test_build_step_plan_keeps_fine_resolution_within_fine_horizon_hours():
    """The first fine_horizon_hours worth of steps must map 1:1 onto model
    steps at the native step_hours resolution; only steps beyond that are
    grouped into coarser coarse_step_hours blocks.
    """

    config = MPCConfig(fine_horizon_hours=1.0, coarse_step_hours=1.0)
    optimizer = MPCOptimizer(THERMAL_MODEL, config)

    # step_hours=0.25 (default) -> 4 fine steps for 1.0h, then coarse blocks
    # of 4 fine steps each (coarse_step_hours=1.0 / step_hours=0.25 = 4).
    horizon = 12
    plan = optimizer._build_step_plan(horizon)

    assert plan.fine_to_model[:4] == [0, 1, 2, 3]
    assert plan.dt_hours[:4] == [0.25, 0.25, 0.25, 0.25]
    # Remaining 8 fine steps become 2 coarse model steps of 4 each.
    assert plan.fine_to_model[4:8] == [4, 4, 4, 4]
    assert plan.fine_to_model[8:12] == [5, 5, 5, 5]
    assert plan.dt_hours[4:] == [1.0, 1.0]
    assert plan.num_steps == 6


def test_build_step_plan_stays_fully_fine_when_horizon_is_short():
    """A horizon shorter than fine_horizon_hours must not be coarsened at
    all - every fine step gets its own model step.
    """

    config = MPCConfig(fine_horizon_hours=100.0, coarse_step_hours=1.0)
    optimizer = MPCOptimizer(THERMAL_MODEL, config)

    horizon = 8
    plan = optimizer._build_step_plan(horizon)

    assert plan.fine_to_model == list(range(horizon))
    assert plan.dt_hours == [0.25] * horizon


def test_build_step_plan_handles_a_partial_trailing_coarse_block():
    """A final coarse block shorter than coarse_step_hours (not enough
    remaining fine steps to fill it) must use however many fine steps
    actually remain, not silently drop them or overrun the horizon.
    """

    config = MPCConfig(fine_horizon_hours=0.5, coarse_step_hours=1.0)
    optimizer = MPCOptimizer(THERMAL_MODEL, config)

    # 2 fine steps (0.5h / 0.25h), then 5 remaining fine steps: one full
    # coarse block of 4, one partial trailing block of 1.
    horizon = 7
    plan = optimizer._build_step_plan(horizon)

    assert plan.fine_to_model == [0, 1, 2, 2, 2, 2, 3]
    assert plan.dt_hours == [0.25, 0.25, 1.0, 0.25]
    assert sum(plan.dt_hours) == pytest.approx(horizon * 0.25)


def test_aggregate_uses_mean_and_max_correctly():
    config = MPCConfig(fine_horizon_hours=0.25, coarse_step_hours=0.5)
    optimizer = MPCOptimizer(THERMAL_MODEL, config)

    plan = optimizer._build_step_plan(horizon=5)  # 1 fine + 2 coarse of 2

    values = [10.0, 20.0, 30.0, 90.0, 10.0]
    mean_agg = optimizer._aggregate(values, plan, lambda b: sum(b) / len(b))
    max_agg = optimizer._aggregate(values, plan, max)

    assert mean_agg == pytest.approx([10.0, 25.0, 50.0])
    assert max_agg == pytest.approx([10.0, 30.0, 90.0])


def test_coarsened_long_horizon_still_meets_a_late_target_with_fewer_variables():
    """A 48h horizon (192 quarter-hour steps) with the default
    fine_horizon_hours/coarse_step_hours must (a) still find a feasible
    schedule that meets a target near the end of the horizon, and (b)
    actually reduce the solver's own variable count relative to the
    uncoarsened case - the entire point of this feature (see real-world
    finding: solve time on a Raspberry Pi 5 was impractically slow at full
    resolution for a horizon this long).
    """

    horizon = 192  # 48h at 15-minute steps
    solar = [0.0] * horizon
    target = [10.0] * horizon
    # A deadline late in the second day, on a coarse step's boundary: the
    # model checks a coarse step's target where the step starts, so a deadline
    # inside it would see the tank's cooling over the rest of that step.
    target[168] = 45.0

    data = _make_input(
        solar_forecast_w=solar,
        target_temperature_top=tuple(target),
        current_temp_top=20.0,
        current_temp_bottom=20.0,
    )

    config = MPCConfig()  # defaults: fine_horizon_hours=18, coarse_step_hours=1
    optimizer = MPCOptimizer(THERMAL_MODEL, config)

    coarse_model = optimizer._build_model(data)
    coarse_variable_count = len(list(coarse_model.boiler_on))

    result = optimizer.solve(data)

    assert result.temperatures[168] >= 45.0 - 1e-6
    assert len(result.schedule) == horizon
    assert len(result.temperatures) == horizon

    # Fully fine resolution would need one boiler_on per fine step.
    assert coarse_variable_count < horizon


def test_target_holds_through_a_coarse_look_ahead_block():
    """A target must hold for the whole step, not just where it starts.

    T[k] is the temperature at the START of step k, so checking only that
    leaves a coasting tank free to sag below the target before the step ends.
    Over a quarter hour that is worth hundredths of a kelvin; over the one-hour
    blocks the far end of the horizon is aggregated into it is around 0.25 K,
    and a real plan met its 18:00 target while dropping under it by 18:45.
    """

    # Long enough that the far half is aggregated into coarse blocks.
    horizon = int((MPCConfig().fine_horizon_hours + 12.0) * 4)
    solar = (0.0,) * horizon

    # A deadline inside the coarse region, held for a few hours so the tank has
    # to still be above it at the end of a block and not only at its start.
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


BUILDING_MODEL = BuildingLumpedModel(
    ua_w_per_k=130.0,
    c_j_per_k=40.0e6,
    a_eff_m2=9.0,
    internal_gain_fraction=1.0,
)


def _zone_input(**overrides) -> MPCInput:
    """A zone starting below a target it has to reach later in the horizon."""

    target = [10.0] * len(SOLAR_FORECAST_W)

    for k in range(len(SOLAR_FORECAST_W) // 2, len(SOLAR_FORECAST_W)):
        target[k] = 20.0

    defaults = dict(
        zone_temperature=19.5,
        zone_target_temperature=tuple(target),
        zone_internal_gain_w=(0.0,) * len(SOLAR_FORECAST_W),
    )
    defaults.update(overrides)

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
    nothing: a DHW start spends an estimated 0.4-0.5 kWh reheating the loop and
    coil before the tank gains at all.
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


# The same zone as BUILDING_MODEL, split into the air it holds and the screed
# and internal walls behind it: the capacities sum to the single node's, so the
# two structures store the same energy and only differ in how fast the store
# reaches the air.
TWO_NODE_BUILDING_MODEL = BuildingThermalModel(
    ua_envelope_w_per_k=130.0,
    ua_air_mass_w_per_k=500.0,
    c_air_j_per_k=2.5e6,
    c_mass_j_per_k=37.5e6,
    a_eff_m2=9.0,
    # A pure air sensor, so the plans these tests check are judged on the air
    # node exactly as before; the operative reading has its own test.
    sensor_mass_fraction=0.0,
    internal_gain_fraction=1.0,
)


def test_the_two_node_zone_is_brought_to_its_target():
    """The MPC plans against whichever structure it was handed.

    The air reaches the target and holds it. It arrives slightly late, because
    the compressor's heat enters the screed and reaches the air only through
    the coupling - that lag is the structure telling the truth about floor
    heating, not the plan giving up, so the transient shortfall is bounded
    rather than forbidden.
    """

    data = _zone_input(zone_mass_temperature=19.5)

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE_BUILDING_MODEL,
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


def test_reaching_the_air_through_the_screed_costs_more_heat():
    """Why the structure is worth the extra state.

    Both zones store the same energy per kelvin and lose it through the same
    envelope; they differ only in that the two-node one has to raise the screed
    before the air follows. Holding the same comfort target therefore takes
    more heat and a longer run - which is exactly the cost a plan must know
    about before it starts a floor run.
    """

    two_node = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE_BUILDING_MODEL,
    ).solve(_zone_input(zone_mass_temperature=19.5))

    single_node = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=BUILDING_MODEL,
    ).solve(_zone_input())

    assert sum(two_node.space_heat_w) > sum(single_node.space_heat_w)
    assert sum(two_node.space_schedule) >= sum(single_node.space_schedule)


def test_a_two_node_zone_is_not_planned_without_its_mass_temperature():
    """Nothing measures the screed, so a plan that needs it must be given the
    filter's estimate. Guessing it would silently decide whether the floor is
    charged or cold, which is the whole question the second state answers.
    """

    result = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE_BUILDING_MODEL,
    ).solve(_zone_input())

    assert result.space_schedule == ()
    assert result.zone_temperatures == ()


def test_a_charged_screed_needs_less_heating_than_a_cold_one():
    """The effect the second state exists for.

    Same air temperature, same comfort target, same weather - only the heat
    already stored in the floor differs. A single-node zone cannot express
    this at all: its one temperature is the air's.
    """

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE_BUILDING_MODEL,
    )

    cold = optimizer.solve(_zone_input(zone_mass_temperature=18.0))
    charged = optimizer.solve(_zone_input(zone_mass_temperature=24.0))

    assert sum(charged.space_heat_w) < sum(cold.space_heat_w)


def test_solar_gain_warms_the_two_node_zone_through_its_mass():
    """Shortwave through the glazing lands on floor and furnishings, not in the
    air, so it reaches the air node only via the coupling - but it does reach
    it, and it displaces heating the compressor would otherwise have to deliver.
    """

    optimizer = MPCOptimizer(
        THERMAL_MODEL,
        MPCConfig(),
        cop_model=COP_MODEL,
        building_model=TWO_NODE_BUILDING_MODEL,
    )

    steps = len(SOLAR_FORECAST_W)
    dark = optimizer.solve(_zone_input(zone_mass_temperature=19.5))
    sunny = optimizer.solve(
        _zone_input(
            zone_mass_temperature=19.5,
            zone_solar_gain_w=(2000.0,) * steps,
        )
    )

    assert sum(sunny.space_heat_w) < sum(dark.space_heat_w)


def test_comfort_is_judged_on_what_the_thermostat_reads():
    """A wall thermostat reads part mass, so warm air over a cold floor reads
    colder than the air - and that reading is what the plan is held to and
    reports, not the air node.
    """

    data = _zone_input(
        zone_temperature=22.0,
        zone_mass_temperature=16.0,
        zone_target_temperature=(20.0,) * len(SOLAR_FORECAST_W),
    )

    def solve(building):
        return MPCOptimizer(
            THERMAL_MODEL,
            MPCConfig(),
            cop_model=COP_MODEL,
            building_model=building,
        ).solve(data)

    air_only = solve(TWO_NODE_BUILDING_MODEL)
    operative = solve(replace(TWO_NODE_BUILDING_MODEL, sensor_mass_fraction=0.5))

    # Half air, half mass against the air node alone.
    assert air_only.zone_temperatures[0] == pytest.approx(22.0)
    assert operative.zone_temperatures[0] == pytest.approx(19.0)
    # A reading this far below target is worth heat, and never less of it than
    # the same zone judged on its warmer air.
    assert sum(operative.space_heat_w) >= sum(air_only.space_heat_w)
    assert sum(operative.space_heat_w) > 0.0
