import numpy as np
import pytest

from domain.types import BoilerThermalModel, HeatPumpCOPModel, MPCConfig, MPCInput
from features.boiler import CP_WATER_J_PER_KG_K, RHO_WATER_KG_PER_L, discretize_zoh
from features.cop import HeatPumpCOPIdentifier
from features.optimizer import MPCOptimizer, _lumped_state_space

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
    (boiler_start[k] == boiler_on[k] - boiler_on[k-1]) forces boiler_start to -1
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
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    baseline = optimizer.solve(_make_input(target_temperature_top=low_target))

    with_draws = optimizer.solve(
        _make_input(
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

    # Force a known, arbitrary on/off pattern and re-derive T by hand.
    pattern = [1, 1, 0, 0, 1, 0] * 4
    pattern = pattern[: len(SOLAR_FORECAST_W)]

    for k, v in enumerate(pattern):
        model.boiler_on[k].fix(v)

    from pyomo.contrib.appsi.solvers.highs import Highs

    results = Highs().solve(model)

    from pyomo.contrib.appsi.base import TerminationCondition

    assert results.termination_condition == TerminationCondition.optimal

    import pyomo.environ as pyo

    solved_T = [pyo.value(model.T[k]) for k in range(len(pattern))]

    a, b = _lumped_state_space(
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
            assert run_length >= config.boiler_min_runtime_steps or j == len(
                schedule
            )
            k = j
        else:
            k += 1


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

    assert with_band.schedule == on_p50.schedule
    assert with_band.objective_value == pytest.approx(on_p50.objective_value)


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

    assert {k for k, on in enumerate(on_p50.schedule) if on} <= set(uncertain)
    assert {k for k, on in enumerate(with_band.schedule) if on} <= set(certain)
    assert with_band.temperatures[20] >= 45.0 - 1e-6


def test_validate_input_rejects_a_solar_band_of_the_wrong_length():
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    with pytest.raises(ValueError):
        optimizer.solve(_make_input(solar_p10_w=(0.0, 0.0), solar_p90_w=(0.0, 0.0)))

    with pytest.raises(ValueError):
        optimizer.solve(_make_input(solar_p10_w=tuple(SOLAR_FORECAST_W)))


def test_validate_input_rejects_mismatched_target_length():
    data = _make_input(target_temperature_top=(10.0, 10.0))  # wrong length

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    with pytest.raises(ValueError):
        optimizer.solve(data)


def test_no_heating_scheduled_when_target_already_below_current():
    """If the whole horizon's requirement is already satisfied by the current
    temperature, the optimizer must not spend money heating anyway.
    """

    data = _make_input(
        current_temp_top=50.0,
        current_temp_bottom=50.0,
        target_temperature_top=(10.0,) * len(SOLAR_FORECAST_W),
    )

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    assert all(v == 0 for v in result.schedule)
    assert result.objective_value == pytest.approx(0.0, abs=1e-9)


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
    expected_power_w = alpha + beta * result.temperatures[10]

    assert result.electrical_power_w[10] == pytest.approx(expected_power_w)
    # Differs meaningfully from the flat fallback, proving the model is
    # actually driving the value, not coincidentally matching it.
    assert result.electrical_power_w[10] != pytest.approx(
        MPCConfig().boiler_electrical_power_w
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

    t_k_high = HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C - margin
    power_at_high = alpha + beta * t_k_high

    expected_cop = COP_MODEL.cop(5.0, HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C)
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


def test_electrical_power_falls_back_to_flat_assumption_without_cop_model():
    data = _make_input()
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    assert all(
        p == pytest.approx(MPCConfig().boiler_electrical_power_w)
        for p in result.electrical_power_w
    )


def test_electrical_power_falls_back_without_outdoor_forecast_even_with_cop_model():
    data = _make_input()  # no outdoor_temperature_forecast

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    result = optimizer.solve(data)

    assert all(
        p == pytest.approx(MPCConfig().boiler_electrical_power_w)
        for p in result.electrical_power_w
    )


def test_cop_clamped_to_sanity_range_for_implausible_inputs():
    """An outdoor/target combination outside anything the model was fit on
    must not translate into an absurd electrical-power estimate - COP is
    clamped to HeatPumpCOPIdentifier's own [MIN_COP, MAX_COP] sanity range
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

    raw_cop = COP_MODEL.cop(60.0, HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C)
    assert raw_cop > HeatPumpCOPIdentifier.MAX_COP  # sanity-check the setup

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(), cop_model=COP_MODEL)
    result = optimizer.solve(data)

    alpha, beta = optimizer._power_line_coefficients(60.0, max(target))
    expected_power_w = alpha + beta * result.temperatures[10]

    assert result.electrical_power_w[10] == pytest.approx(expected_power_w)
    # The clamp must actually have engaged for this scenario, i.e. the fit's
    # high endpoint used the clamped, not the implausible raw, COP. The line
    # is valid in T[k] terms, so the high reference point corresponds to
    # T[k] = POWER_FIT_T_HIGH_C - margin (T_supply = T[k] + margin).
    unclamped_power_at_high = COP_MODEL.q_th_at_power_fit_high_w / raw_cop
    t_k_high = HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C - margin
    clamped_power_at_high = alpha + beta * t_k_high
    assert clamped_power_at_high == pytest.approx(
        COP_MODEL.q_th_at_power_fit_high_w / HeatPumpCOPIdentifier.MAX_COP
    )
    assert clamped_power_at_high != pytest.approx(unclamped_power_at_high)


def test_lumped_state_space_matches_full_tank_capacity():
    c_expected = RHO_WATER_KG_PER_L * THERMAL_MODEL.volume_l * CP_WATER_J_PER_KG_K
    ua_total = THERMAL_MODEL.ua_top_w_per_k + THERMAL_MODEL.ua_bottom_w_per_k

    a, b = _lumped_state_space(THERMAL_MODEL.volume_l, ua_total)

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
    target[170] = 45.0  # a deadline late in the second day

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

    assert result.temperatures[170] >= 45.0 - 1e-6
    assert len(result.schedule) == horizon
    assert len(result.temperatures) == horizon

    # Fully fine resolution would need one boiler_on per fine step.
    assert coarse_variable_count < horizon
