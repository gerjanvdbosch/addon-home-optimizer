import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from domain.mpc import MPCResult
from domain.state import Plan, State
from infrastructure.repositories import StateRepository
from infrastructure.storage import JsonStorage
from web.server import state as state_endpoint

START = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)


def _reject(constant: str):
    raise ValueError(f"not strict JSON: {constant}")


def test_state_endpoint_writes_nan_as_null(tmp_path):
    """The plan marks quarters with the zone off with NaN supply; state.json
    keeps it, the endpoint must still return strict JSON."""

    repository = StateRepository(JsonStorage(tmp_path / "state.json"))
    state = State()
    state.plan = Plan(
        made=START,
        times=[START + timedelta(minutes=15 * i) for i in range(3)],
        result=MPCResult(
            schedule=(0, 0, 0),
            temperatures=(45.0, 45.0, 45.0),
            electrical_power_w=(0.0, 0.0, 0.0),
            heat_w=(0.0, 0.0, 0.0),
            objective_value=0.0,
            termination_condition="optimal",
            space_schedule=(1, 0, 1),
            space_heat_w=(3000.0, 0.0, 3000.0),
            space_supply_c=(26.0, math.nan, 26.5),
        ),
    )
    repository.save(state)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(repositories=SimpleNamespace(state=repository))
        )
    )

    response = asyncio.run(state_endpoint(request))  # type: ignore[arg-type]

    assert response.status_code == 200
    assert response.media_type == "application/json"
    data = json.loads(bytes(response.body), parse_constant=_reject)
    assert data["plan"]["result"]["space_supply_c"] == [26.0, None, 26.5]
