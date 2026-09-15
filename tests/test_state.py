from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.state import StateManager
from domain.types import SeriesPoint, State
from features.solar import nowcast_solar

LATITUDE = 52.0
LONGITUDE = 5.0
NOW = datetime(2026, 6, 21, 10, 37, tzinfo=UTC)


def _state_manager() -> StateManager:
    return StateManager(
        loader=None,  # type: ignore[arg-type]
        state_repository=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )


def test_future_series_starts_at_the_period_running_now():
    """Solcast times are period starts: at 10:37 the 10:30 value is the forecast
    for right now and must not be dropped."""

    points = [
        SeriesPoint(time=datetime(2026, 6, 21, 10, 0, tzinfo=UTC), value=1.0),
        SeriesPoint(time=datetime(2026, 6, 21, 10, 30, tzinfo=UTC), value=2.0),
        SeriesPoint(time=datetime(2026, 6, 21, 11, 0, tzinfo=UTC), value=3.0),
    ]

    series = StateManager._future_series(points, NOW)

    assert series.tolist() == [2.0, 3.0]


def test_running_quarter_takes_the_measurement_so_far():
    quarter = datetime(2026, 6, 21, 10, 30, tzinfo=UTC)
    measured = [SeriesPoint(time=quarter, value=800.0)]

    result = _state_manager()._current_quarter_nowcast(measured, NOW)

    assert result is not None
    assert result[0] == quarter
    # The running bucket's mean covers 10:30-10:37, so it is centred at 10:33:30.
    assert result[1] == pytest.approx(
        nowcast_solar(
            800.0,
            quarter + (NOW - quarter) / 2,
            quarter + timedelta(minutes=7.5),
            LATITUDE,
            LONGITUDE,
        )
    )


def test_no_estimate_from_a_stale_measurement():
    """The PV sensor stops reporting at night; an hour-old reading says nothing
    about the running quarter hour."""

    measured = [SeriesPoint(time=NOW - timedelta(hours=1), value=800.0)]

    assert _state_manager()._current_quarter_nowcast(measured, NOW) is None


QUARTER = datetime(2026, 6, 21, 10, 30, tzinfo=UTC)
TIMES = [QUARTER, QUARTER + timedelta(minutes=15)]


def _baseload_state(forecast: list[float], measured_w: float) -> State:
    state = State(updated=NOW)
    state.predictions.baseload = [
        # strict=False: a forecast may cover fewer steps than TIMES (or none).
        SeriesPoint(time=t, value=value)
        for t, value in zip(TIMES, forecast, strict=False)
    ]
    state.measurements.baseload = [SeriesPoint(time=QUARTER, value=measured_w)]
    return state


def test_running_quarter_baseload_takes_the_measurement():
    """Load that is on right now counts - above or below the forecast - so a
    running appliance keeps the heat pump from counting on sun it takes."""

    manager = _state_manager()

    lower = manager.baseload_forecast(
        _baseload_state([300.0, 250.0], 180.0), TIMES, NOW
    )
    higher = manager.baseload_forecast(
        _baseload_state([300.0, 250.0], 900.0), TIMES, NOW
    )

    assert lower == [180.0, 250.0]
    assert higher == [900.0, 250.0]


def test_baseload_without_a_forecast_uses_the_measurement_only_for_now():
    result = _state_manager().baseload_forecast(_baseload_state([], 180.0), TIMES, NOW)

    assert result == [180.0, 0.0]
