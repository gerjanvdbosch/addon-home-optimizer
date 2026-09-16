from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.optimization import Optimization

START = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)
TIMES = [START + timedelta(minutes=15 * i) for i in range(4)]


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

    optimization.publish_dhw(schedule, TIMES)

    return {entity: state for entity, (state, _) in home_assistant.states.items()}


def test_a_run_planned_later_is_off_now_with_its_start_time():
    assert _published((0, 0, 1, 1)) == {
        Optimization.DHW_STATUS_ENTITY: "off",
        Optimization.DHW_START_ENTITY: TIMES[2].isoformat(),
    }


def test_a_run_planned_now_is_on_and_starts_now():
    assert _published((1, 1, 0, 0)) == {
        Optimization.DHW_STATUS_ENTITY: "on",
        Optimization.DHW_START_ENTITY: TIMES[0].isoformat(),
    }


def test_no_planned_run_leaves_the_start_unknown():
    assert _published((0, 0, 0, 0)) == {
        Optimization.DHW_STATUS_ENTITY: "off",
        Optimization.DHW_START_ENTITY: "unknown",
    }
