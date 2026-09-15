import pandas as pd
import pytest

from domain.types import BoilerThermalModel, MPCConfig, MPCInput
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
    df = pd.DataFrame(
        {
            "boiler_on": [True] * 6,
            "booster_on": [False, False, True, True, True, True],
            "T_top": [50.0, 54.0, 55.0, 57.0, 59.0, 61.0],
            "T_bottom": [50.0, 54.0, 55.0, 57.0, 59.0, 61.0],
            "q_in_override_w": [6000.0, 3000.0, 1500.0, 2000.0, 2000.0, 2000.0],
        }
    )
    identifier = BoilerThermalIdentifier()

    assert identifier._identify_booster(df) == (55.0, 2000.0)
    assert identifier._identify_booster(df.assign(booster_on=False)) == (None, None)


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

    result = MPCOptimizer(BOOSTER_MODEL, MPCConfig()).solve(data)
    temperatures = result.temperatures
    booster_steps = [
        i
        for i, on in enumerate(result.schedule)
        if on and result.electrical_power_w[i] == BOOSTER_MODEL.booster_heat_w
    ]
    heat_pump_steps = [
        i for i, on in enumerate(result.schedule) if on and i not in booster_steps
    ]

    assert temperatures[20] >= 60.0 - 1e-6
    assert booster_steps
    assert all(temperatures[i] >= 55.0 - 1e-6 for i in booster_steps)
    assert all(temperatures[i + 1] <= 55.0 + 1e-6 for i in heat_pump_steps)


def test_a_run_started_by_the_booster_counts_as_a_start():
    """A booster-only run (tank already above the heat pump's limit) is still a
    DHW run that has to be started - its start carries the switching cost."""

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

    free = MPCOptimizer(BOOSTER_MODEL, MPCConfig(weight_switching=0.0)).solve(data)
    charged = MPCOptimizer(BOOSTER_MODEL, MPCConfig()).solve(data)

    assert any(charged.schedule)
    assert charged.objective_value == pytest.approx(
        free.objective_value + MPCConfig().weight_switching
    )
