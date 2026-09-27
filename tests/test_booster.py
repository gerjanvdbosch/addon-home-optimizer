from dataclasses import replace

import pandas as pd
import pytest

from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput
from features.boiler import BoilerThermalIdentifier, booster_active
from features.cop import HeatPumpCOPIdentifier
from features.optimizer import MPCOptimizer

SWW = "SWW"


def _run(frequency: list[float], flow: list[float]) -> pd.DataFrame:
    """A DHW run followed by one idle row."""

    return pd.DataFrame(
        {
            "state": [SWW] * len(frequency) + ["Uit"],
            "compressor_frequency": frequency + [0.0],
            "flow_lpm": flow + [0.0],
        }
    )


def test_booster_is_recognised_without_a_sensor_up_to_its_last_row():
    """Compressor stopped after running, while the DHW run continues with water
    circulating, is the booster - including the run's final booster row."""

    df = _run([60.0, 0.0, 0.0, 0.0], [18.5] * 4)

    assert booster_active(df, SWW).tolist() == [False, True, True, True, False]


def test_pump_overrun_at_a_normal_run_end_is_not_the_booster():
    df = _run([60.0, 60.0, 0.0], [18.5] * 3)

    assert not booster_active(df, SWW).any()


def test_pump_circulating_before_the_compressor_starts_is_not_the_booster():
    df = _run([0.0, 60.0, 60.0], [18.5] * 3)

    assert not booster_active(df, SWW).any()


def test_booster_sensor_counts_in_home_assistant_and_influxdb_form():
    df = pd.DataFrame({"state": [SWW] * 3, "booster": ["on", 1.0, 0.0]})

    assert booster_active(df, SWW).tolist() == [True, True, False]


MIXING_MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=1.0,
    ua_bottom_w_per_k=1.0,
    ua_mix_idle_w_per_k=0.1,
    ua_mix_active_w_per_k=500.0,
    q_in_nominal_w=6000.0,
    cold_layer_fraction=0.35,
    cold_water_temperature_c=15.0,
    cold_layer_spread_k=8.0,
)


def test_a_mixed_tank_is_its_sensors_average():
    assert MIXING_MODEL.mixed_temperature(46.0, 46.0) == 46.0


def test_one_sensor_step_apart_is_a_mixed_tank():
    """At rest the sensors flip a 0.5 K step apart: no stratification."""

    assert MIXING_MODEL.mixed_temperature(46.0, 45.5) == 45.75


def test_a_stratified_tank_holds_the_cold_layer_below_its_sensors():
    """Fully stratified - 8 K beyond one sensor step: 35% of the tank at
    15 degC, the rest at the 41 degC average; half as stratified, half that
    layer."""

    assert MIXING_MODEL.mixed_temperature(45.25, 36.75) == pytest.approx(
        41.0 - 0.35 * (41.0 - 15.0)
    )
    assert MIXING_MODEL.mixed_temperature(43.25, 38.75) == pytest.approx(
        41.0 - 0.35 * 0.5 * (41.0 - 15.0)
    )


def test_the_cold_layer_is_learned_from_each_runs_closed_heat_balance():
    """Twelve runs on the model above: a row before, three rows heating at
    3 kW, then idle rows until the loop's heat has reached the tank. The tank
    settles mixed at its mean before the run plus the heat put in, so the fit
    recovers the layer's share and temperature. No standing loss: the ambient
    air is at the tank's own temperature."""

    rows = []
    time = pd.Timestamp("2026-09-01 10:00", tz="UTC")
    step = pd.Timedelta(minutes=5)
    heat_w = 3000.0
    added_k = 3 * heat_w * step.total_seconds() / (200.0 * 4186.0)

    for n in range(12):
        top, bottom = 30.0 + 1.5 * n, 22.0 + 1.5 * n - (n % 3)
        settled = MIXING_MODEL.mixed_temperature(top, bottom) + added_k
        run = [(top, bottom, False, float("nan"))]
        run += [(top + 2.0, bottom + 4.0, True, heat_w)] * 3
        run += [(settled, settled, False, float("nan"))] * 5

        for T_top, T_bottom, on, q in run:
            rows.append(
                dict(
                    time=time,
                    T_top=T_top,
                    T_bottom=T_bottom,
                    T_ambient=(T_top + T_bottom) / 2.0,
                    boiler_on=on,
                    booster_on=False,
                    q_in_override_w=q,
                )
            )
            time += step

    identifier = BoilerThermalIdentifier()
    identifier.model = MIXING_MODEL
    fraction, cold_c, spread_k = identifier._identify_cold_layer(pd.DataFrame(rows))

    assert fraction == pytest.approx(0.35, abs=0.02)
    assert cold_c == pytest.approx(15.0, abs=1.0)
    assert spread_k == pytest.approx(8.0, abs=1.0)


def _excess_heat_j(model: BoilerThermalModel, bottom_after: float) -> float:
    """The tap target over one idle 5-minute step in which the bottom sensor
    goes from 46 to bottom_after degC (the top stays at 46)."""

    identifier = BoilerThermalIdentifier()
    identifier.model = model
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-27 08:00", periods=2, freq="5min"),
            "T_top": [46.0, 46.0],
            "T_bottom": [46.0, bottom_after],
            "T_ambient": [20.0, 20.0],
            "boiler_on": [False, False],
            "dt_seconds": [300.0, 300.0],
        }
    )

    return float(identifier.excess_loss_w(df).excess_loss_w.iloc[0]) * 300.0


def test_a_tap_counts_the_cold_layer_it_fills():
    """Cold water fills the tank from the bottom: once the bottom sensor drops,
    the layer below it is cold too, so the tap took more heat than the sensors'
    average shows. Without a tap the two targets agree."""

    average_only = replace(MIXING_MODEL, cold_layer_fraction=None)

    assert _excess_heat_j(MIXING_MODEL, 38.0) > 1.5 * _excess_heat_j(average_only, 38.0)
    assert _excess_heat_j(MIXING_MODEL, 46.0) == pytest.approx(
        _excess_heat_j(average_only, 46.0)
    )


def test_a_plan_starts_from_the_tank_mixed():
    data = MPCInput(
        solar_forecast_w=[0.0] * 8,
        ambient_temperature=20.0,
        current_temp_top=45.0,
        current_temp_bottom=37.0,
        boiler_on_current=False,
        target_temperature_top=(10.0,) * 8,
    )

    result = MPCOptimizer(MIXING_MODEL, MPCConfig()).solve(data)

    assert result.temperatures[0] == pytest.approx(
        MIXING_MODEL.mixed_temperature(45.0, 37.0)
    )


def test_the_supply_margin_is_against_the_tanks_energy_temperature():
    """A run starting 16 K stratified, so its tank mixes below the sensors'
    average, and heated by 6976 W: 2.5 K per 5-minute row, half of it by a
    row's middle. Its first quarter hour's supply 1, 2, 3 K below the energy
    temperature there, the rest 9, 10, 11 K above it - a margin of 10 K past
    the first step, whatever the sensors read meanwhile. The booster row
    (resistive, no supply to speak of) and the idle row with no flow do not
    count."""

    heat_w = 2.5 * 200.0 * 4186.0 / 300.0
    mixed = MIXING_MODEL.mixed_temperature(48.0, 32.0)
    middle = [mixed + 2.5 * (row + 0.5) for row in range(6)]
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-27 08:00", periods=8, freq="5min"),
            "boiler_on": [True] * 7 + [False],
            "booster_on": [False] * 6 + [True, False],
            "flow_lpm": [15.0] * 7 + [0.0],
            "q_in_override_w": [heat_w] * 7 + [float("nan")],
            "T_supply": [
                t + dt for t, dt in zip(middle, (-1, -2, -3, 9, 10, 11), strict=True)
            ]
            + [90.0, 20.0],
            "T_top": [48.0] + [50.0] * 7,
            "T_bottom": [32.0] + [40.0] * 7,
        }
    )
    identifier = BoilerThermalIdentifier()
    identifier.model = MIXING_MODEL

    assert identifier._identify_supply_margin(df) == pytest.approx(10.0)


def test_setpoint_overshoot_is_learned_from_runs_that_stopped_on_the_setpoint():
    """Four runs, each followed by 20 minutes idle: two heat pump runs that
    settle 1.8 and 1.5 K above their setpoint, one that stopped on the heat
    pump's limit below its setpoint, and a booster run - only the first two
    count, at their median."""

    runs = [
        # (setpoint, booster, tank temperature: two heating rows, four idle)
        (46.0, False, [40.0, 44.0, 45.0, 47.0, 47.8, 47.8]),
        (60.0, False, [50.0, 54.0, 54.5, 55.0, 55.2, 55.2]),
        (60.0, True, [56.0, 59.0, 60.0, 60.6, 60.8, 60.8]),
        (47.0, False, [41.0, 45.0, 46.0, 48.0, 48.5, 48.5]),
    ]
    heating = [True, True, False, False, False, False]
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-15 10:00", periods=24, freq="5min"),
            "setpoint": [setpoint for setpoint, _, _ in runs for _ in heating],
            "boiler_on": heating * len(runs),
            "booster_on": [booster and on for _, booster, _ in runs for on in heating],
            "T_top": [T for _, _, temperatures in runs for T in temperatures],
            "T_bottom": [T for _, _, temperatures in runs for T in temperatures],
        }
    )

    overshoot = BoilerThermalIdentifier()._identify_setpoint_overshoot(df)

    assert overshoot == pytest.approx((1.8 + 1.5) / 2.0)


def test_heat_pump_limit_and_booster_heat_are_identified_from_booster_runs():
    nan = float("nan")
    # A full run - heat pump, booster, then settling after the cut-out - and a
    # second run stopped early.
    T = [50.0, 54.0, 55.0, 57.0, 59.0, 60.0, 61.0, 60.8, 60.6]
    T += [54.0, 55.0, 56.0, 56.5, 56.4]
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-15 10:00", periods=14, freq="5min"),
            "boiler_on": [True] * 6 + [False] * 3 + [True] * 3 + [False] * 2,
            "booster_on": [False, False]
            + [True] * 4
            + [False] * 4
            + [True] * 2
            + [False] * 2,
            "T_top": T,
            "T_bottom": T,
            "q_in_override_w": [6000.0, 3000.0, 1500.0, 2000.0, 2000.0, 2000.0]
            + [nan] * 3
            + [3000.0, 2000.0, 2000.0]
            + [nan] * 2,
        }
    )
    identifier = BoilerThermalIdentifier()

    # Both runs hand over at 55 degC (heat pump limit). The full run keeps warming
    # after the booster is cut out, to 61 degC - the tank's maximum, whatever the
    # early-stopped run reached.
    assert identifier._identify_booster(df) == (55.0, 61.0, 2000.0)
    assert identifier._identify_booster(df.assign(booster_on=False)) == (
        None,
        None,
        None,
    )


def test_cop_fit_excludes_booster_rows_recognised_without_a_sensor():
    # The first three readings are the compressor's start-up, which prepare()
    # drops on its own (see HeatPumpCOPIdentifier.STARTUP).
    times = pd.date_range("2026-01-01T10:00:00Z", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "time": times,
            "T_outdoor": [10.0] * 6,
            "T_supply": [58.0] * 6,
            "T_return": [53.0] * 6,
            "flow_lpm": [18.5] * 6,
            "P_el": [2000.0] * 6,
            "state": [SWW] * 6,
            "compressor_frequency": [40.0] * 4 + [0.0, 30.0],
        }
    )

    prepared = HeatPumpCOPIdentifier(key="dhw").prepare(df)

    assert prepared["time"].tolist() == [times[3], times[5]]


BOOSTER_MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=0.15,
    ua_bottom_w_per_k=0.20,
    ua_mix_idle_w_per_k=0.03,
    ua_mix_active_w_per_k=9638.6,
    q_in_nominal_w=3700.0,
    heat_pump_max_tank_temperature_c=55.0,
    max_tank_temperature_c=60.0,
    booster_heat_w=2000.0,
)
HORIZON = 24


def test_booster_heats_above_the_heat_pump_limit_and_only_there():
    target = [10.0] * HORIZON
    # Below the maximum, as a real legionella target is: a target at the
    # thermostat's own cut-out could only be met at the moment it cuts out.
    target[20] = 59.0
    data = MPCInput(
        solar_forecast_w=[0.0] * HORIZON,
        ambient_temperature=20.0,
        current_temp_top=40.0,
        current_temp_bottom=40.0,
        boiler_on_current=False,
        target_temperature_top=tuple(target),
    )

    result = MPCOptimizer(BOOSTER_MODEL, MPCConfig()).solve(data)
    temperatures = result.temperatures
    # Only the booster can have raised the tank above the heat pump's own limit.
    above_the_limit = [t for t in temperatures if t > 55.0 + 1e-6]
    heated = [i for i, heat_w in enumerate(result.heat_w) if heat_w > 0.0]

    assert temperatures[20] >= 59.0 - 1e-6
    assert above_the_limit
    # The thermostat cuts the booster out at the maximum, and it takes over in
    # the very step the heat pump reaches its limit: one run, no idle step
    # between the two.
    assert max(temperatures) <= BOOSTER_MODEL.max_tank_temperature_c + 1e-6
    assert heated == list(range(heated[0], heated[-1] + 1))


def test_the_booster_cannot_start_a_run_by_itself():
    """The booster only takes over from a compressor that cannot lift the tank
    further - it never starts a DHW run on its own. With the tank already above
    the heat pump's limit, that leaves the target unreachable rather than served
    by a booster-only run."""

    target = [10.0] * HORIZON
    target[8] = 57.0
    data = MPCInput(
        solar_forecast_w=[0.0] * HORIZON,
        ambient_temperature=20.0,
        current_temp_top=56.0,
        current_temp_bottom=56.0,
        boiler_on_current=False,
        target_temperature_top=tuple(target),
    )

    result = MPCOptimizer(BOOSTER_MODEL, MPCConfig()).solve(data)

    assert not any(result.schedule)


def test_a_tank_near_the_limit_still_starts_and_hands_over():
    """The minimum runtime must not outlaw a handover the physics requires.

    One degree under the heat pump's own limit there is not two steps of
    compressor work to be had. Requiring compressor time specifically for the
    whole minimum runtime made the solver drop the run and pay the slack
    instead - missing a target it could have reached by letting the booster
    finish. The booster may only run once the compressor is at its limit, so
    "still heating" already implies "compressor still running unless it cannot".
    """

    target = [10.0] * HORIZON
    target[20] = 60.0
    data = MPCInput(
        solar_forecast_w=[0.0] * HORIZON,
        ambient_temperature=20.0,
        # Just under the heat pump's 55 degC limit: it can lift the tank by
        # about a degree, the booster has to do the rest.
        current_temp_top=54.0,
        current_temp_bottom=54.0,
        boiler_on_current=False,
        target_temperature_top=tuple(target),
    )

    result = MPCOptimizer(BOOSTER_MODEL, MPCConfig()).solve(data)

    assert any(result.schedule), "expected the tank to be heated at all"
    assert result.temperatures[20] >= 60.0 - 1e-6
