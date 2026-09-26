from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from app.optimization import Optimization
from domain.config import BoilerConfig, LegionellaConfig
from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.time import local_day_start

START = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)
TIMES = [START + timedelta(minutes=15 * i) for i in range(4)]
TEMPERATURES = (40.0, 44.0, 48.3, 48.2)


class _RecordingHomeAssistant:
    def __init__(self) -> None:
        self.states: dict[str, tuple[str, dict]] = {}

    def set_state(self, entity_id: str, state: str, attributes: dict) -> None:
        self.states[entity_id] = (state, attributes)


def _published(schedule: tuple[int, ...]) -> dict[str, str]:
    home_assistant = _RecordingHomeAssistant()
    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=home_assistant,  # type: ignore[arg-type]
    )
    result = MPCResult(
        schedule=schedule,
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0,) * len(schedule),
        heat_w=(0.0,) * len(schedule),
        objective_value=0.0,
        solver_status="optimal",
        termination_condition="optimal",
    )

    optimization.publish_dhw(result, TIMES)

    return {entity: state for entity, (state, _) in home_assistant.states.items()}


def test_a_run_planned_later_is_off_now_with_its_start_time():
    published = _published((0, 1, 1, 0))

    assert published[Optimization.DHW_STATUS_ENTITY] == "off"
    assert published[Optimization.DHW_START_ENTITY] == TIMES[1].isoformat()


def test_a_run_planned_now_is_on_and_starts_now():
    published = _published((1, 1, 0, 0))

    assert published[Optimization.DHW_STATUS_ENTITY] == "on"
    assert published[Optimization.DHW_START_ENTITY] == TIMES[0].isoformat()


def test_no_planned_run_leaves_the_start_and_setpoint_unknown():
    assert _published((0, 0, 0, 0)) == {
        Optimization.DHW_STATUS_ENTITY: "off",
        Optimization.DHW_START_ENTITY: "unknown",
        Optimization.DHW_SETPOINT_ENTITY: "unknown",
    }


def test_the_setpoint_is_the_planned_end_temperature_rounded_to_the_nearest():
    """The tank is at 48.2 degC after the run's last step, 48.3 degC after a
    one-step run - each to the heat pump's nearest half degree."""

    assert _published((0, 1, 1, 0))[Optimization.DHW_SETPOINT_ENTITY] == "48.0"
    assert _published((0, 1, 0, 0))[Optimization.DHW_SETPOINT_ENTITY] == "48.5"


LEGIONELLA = LegionellaConfig(temperature=60.0, interval=7)
STEPS = 12
NOW = datetime.now(UTC).replace(second=0, microsecond=0)
PLAN_TIMES = [NOW + timedelta(minutes=15 * i) for i in range(STEPS)]
# Sun only in steps 6-9. From 50 degC the run takes the heat pump 12 minutes to
# its 55 degC limit and the booster 35 more to 60: four quarter hours.
SOLAR_W = [0.0] * 6 + [2000.0] * 4 + [0.0] * 2
TANK = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=1.0,
    ua_bottom_w_per_k=1.0,
    ua_mix_idle_w_per_k=0.1,
    ua_mix_active_w_per_k=500.0,
    q_in_nominal_w=6000.0,
    heat_pump_max_tank_temperature_c=55.0,
    max_tank_temperature_c=61.0,
    booster_heat_w=2000.0,
)
DATA = MPCInput(
    solar_forecast_w=SOLAR_W,
    ambient_temperature=20.0,
    current_temp_top=50.0,
    current_temp_bottom=50.0,
    boiler_on_current=False,
    target_temperature_top=(45.0,) * STEPS,
)


def test_the_disinfection_goes_where_the_sun_covers_its_run():
    step = Optimization.disinfection_step(
        DATA, PLAN_TIMES, PLAN_TIMES[-1], TANK, 60.0, MPCConfig()
    )

    # The four steps before it are the four sunny ones.
    assert step == 10


def test_the_disinfection_is_done_by_its_deadline():
    step = Optimization.disinfection_step(
        DATA, PLAN_TIMES, PLAN_TIMES[8], TANK, 60.0, MPCConfig()
    )

    assert step == 8


def _targets_with_legionella(days_ago: int, top: float, bottom: float):
    # Local midnight, so the deadline falls on the same local day in any zone.
    day = local_day_start(NOW, days=-days_ago)
    loader = _DailyMaxLoader(
        pd.DataFrame({"time": [day], "top": [top], "bottom": [bottom]})
    )
    optimization = Optimization(
        loader=loader,  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=_RecordingHomeAssistant(),  # type: ignore[arg-type]
    )
    boiler = BoilerConfig.model_validate(
        {
            "setpoint": "sensor.setpoint",
            "top_temperature": "sensor.top",
            "bottom_temperature": "sensor.bottom",
            "ambient_temperature": "sensor.ambient",
            "target_temperature": 45.0,
            "legionella": LEGIONELLA.model_dump(),
        }
    )

    # A forecast to the end of tomorrow, as Solcast's is: its last quarter hour
    # starts at 23:45 and runs to the deadline at midnight.
    end = local_day_start(NOW, days=2)
    steps = int((end - NOW).total_seconds() // (15 * 60)) + 1
    data = MPCInput(
        solar_forecast_w=[0.0] * steps,
        ambient_temperature=20.0,
        current_temp_top=50.0,
        current_temp_bottom=50.0,
        boiler_on_current=False,
        target_temperature_top=(45.0,) * steps,
    )
    times = [NOW + timedelta(minutes=15 * i) for i in range(steps)]

    return optimization._with_legionella(
        data, times, boiler, TANK, MPCConfig()
    ).target_temperature_top


class _DailyMaxLoader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load(self, dataset, start, end) -> pd.DataFrame:
        return self.frame


def test_no_disinfection_is_planned_while_the_interval_runs():
    assert 60.0 not in _targets_with_legionella(1, 61.0, 60.5)


def test_a_disinfection_is_planned_on_the_day_the_interval_ends():
    assert 60.0 in _targets_with_legionella(7, 61.0, 60.5)


def test_an_interval_ending_tomorrow_is_planned_today_or_tomorrow():
    assert 60.0 in _targets_with_legionella(6, 61.0, 60.5)


def test_an_interval_ending_beyond_the_plan_plans_nothing_yet():
    """The interval ends at the end of the day after tomorrow, past the
    forecast: nothing to choose from yet."""

    assert 60.0 not in _targets_with_legionella(5, 61.0, 60.5)


def test_one_sensor_at_temperature_does_not_count_as_disinfected():
    assert 60.0 in _targets_with_legionella(1, 61.0, 58.0)
