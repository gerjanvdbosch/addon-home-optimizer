import pandas as pd

from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput
from domain.physics import CP_WATER_J_PER_KG_K, RHO_WATER_KG_PER_L
from features.boiler import BoilerThermalIdentifier, booster_active
from features.cop import HeatPumpCOPIdentifier
from features.optimizer import MPCOptimizer

SWW = BoilerThermalIdentifier.DHW_ACTIVE_STATE


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
    times = pd.date_range("2026-01-01T10:00:00Z", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "time": times,
            "T_outdoor": [10.0] * 3,
            "T_supply": [58.0] * 3,
            "T_return": [53.0] * 3,
            "flow_lpm": [18.5] * 3,
            "P_el": [2000.0] * 3,
            "state": [SWW] * 3,
            "compressor_frequency": [40.0, 0.0, 30.0],
        }
    )

    prepared = HeatPumpCOPIdentifier(mode=SWW, key="dhw").prepare(df)

    assert times[1] not in set(prepared["time"])
    assert len(prepared) == 2


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
    target[20] = 60.0
    data = MPCInput(
        solar_forecast_w=[0.0] * HORIZON,
        ambient_temperature=20.0,
        current_temp_top=40.0,
        current_temp_bottom=40.0,
        boiler_on_current=False,
        target_temperature_top=tuple(target),
    )

    config = MPCConfig()
    result = MPCOptimizer(BOOSTER_MODEL, config).solve(data)
    temperatures = result.temperatures
    # Only the booster can have raised the tank above the heat pump's own limit.
    above_the_limit = [t for t in temperatures if t > 55.0 + 1e-6]
    # It cannot modulate, so the step it is cut out in may overshoot by its own
    # full heat input.
    booster_step_k = (
        BOOSTER_MODEL.booster_heat_w
        * config.step_hours
        * 3600.0
        / (RHO_WATER_KG_PER_L * BOOSTER_MODEL.volume_l * CP_WATER_J_PER_KG_K)
    )

    assert temperatures[20] >= 60.0 - 1e-6
    assert above_the_limit
    assert max(temperatures) <= BOOSTER_MODEL.max_tank_temperature_c + booster_step_k


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
