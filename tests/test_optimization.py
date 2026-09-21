from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.optimization import Optimization
from domain.mpc import MPCResult

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


def test_the_setpoint_is_the_planned_end_temperature_rounded_up():
    """The tank is at 48.2 degC after the run's last step - rounded up to the
    heat pump's half degrees."""

    assert _published((0, 1, 1, 0))[Optimization.DHW_SETPOINT_ENTITY] == "48.5"
