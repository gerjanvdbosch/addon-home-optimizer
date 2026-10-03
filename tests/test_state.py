import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.state import StateManager
from domain.state import SeriesPoint, State
from features.solar import nowcast_solar
from infrastructure.repositories import StateRepository
from infrastructure.storage import JsonStorage

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


def test_measurement_moves_the_running_quarter_band_without_narrowing_it(
    monkeypatch,
):
    """The nowcast says where the output is now, not how much the sky will
    still change: the running quarter hour's p10-p90 band keeps its width."""

    quarter = datetime(2026, 6, 21, 10, 30, tzinfo=UTC)
    grid = pd.date_range(quarter, periods=2, freq="15min")

    class Identifier:
        model = object()

        def __init__(self, *args) -> None:
            pass

        def load(self, path) -> None:
            pass

    monkeypatch.setattr("app.state.SolarBiasIdentifier", Identifier)
    monkeypatch.setattr(
        "app.state.predict_solar", lambda *args: pd.Series([1000.0] * 2, grid)
    )
    monkeypatch.setattr(
        "app.state.predict_solar_band",
        lambda *args: (pd.Series([600.0] * 2, grid), pd.Series([1300.0] * 2, grid)),
    )
    state = State(updated=NOW)
    point = [SeriesPoint(time=quarter, value=1.0)]
    state.forecast.solcast.p10 = state.forecast.solcast.p50 = point
    state.forecast.solcast.p90 = point
    state.measurements.solar = [SeriesPoint(time=quarter, value=400.0)]
    # A steady sky: the measurement carries all the weight.
    recent = pd.Series(
        [400.0] * 15, index=pd.date_range(end=NOW, periods=15, freq="1min")
    )

    _state_manager()._predict_solar(state, NOW, recent)

    p10, p50, p90 = (
        series[0].value
        for series in (
            state.predictions.solar_p10,
            state.predictions.solar,
            state.predictions.solar_p90,
        )
    )
    nowcast = _state_manager()._current_quarter_nowcast(state.measurements.solar, NOW)
    assert p50 == pytest.approx(nowcast[1])
    assert p50 - p10 == pytest.approx(400.0)
    assert p90 - p50 == pytest.approx(300.0)
    # The next quarter hour is the forecast's alone.
    assert state.predictions.solar[1].value == 1000.0


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


def test_the_floors_plan_joins_the_heat_pumps(tmp_path):
    """One machine, one plan: the floor's power and heat are added to the
    tank's, cooling counted positive, and a new run starts from the tank's
    plan again rather than adding on top of the last one."""
    manager = _state_manager()
    manager.state_repository = StateRepository(JsonStorage(tmp_path / "state.json"))
    times = [NOW + timedelta(minutes=15 * i) for i in range(3)]

    for _ in range(2):
        manager.update_schedule(
            schedule=[1, 0, 0],
            temperatures=[45.0, 46.0, 46.0],
            power_w=[1500.0, 0.0, 0.0],
            times=times,
            heat_w=[6000.0, 0.0, 0.0],
        )
        manager.update_building_schedule(
            heat_w=[0.0, -3571.0, -3571.0],
            temperatures=[21.5, 21.4, 21.3],
            times=times,
            power_w=[0.0, 800.0, 820.0],
        )

    plan = manager.load().schedule.heat_pump
    assert [p.value for p in plan.power] == [1500.0, 800.0, 820.0]
    assert [p.value for p in plan.heat] == [6000.0, 3571.0, 3571.0]


def test_a_forecast_is_extended_with_the_day_before():
    """Beyond its end a forecast repeats its own day, however far: the
    afternoon sun comes back in the afternoon rather than held into the night."""

    day = [QUARTER + timedelta(hours=6 * i) for i in range(4)]
    forecast = dict(zip(day, [0.0, 2000.0, 500.0, 0.0], strict=True))
    times = [t + timedelta(days=d) for d in range(3) for t in day]

    extended = StateManager.extend_forecast(forecast, times, 0.0)

    assert extended == [0.0, 2000.0, 500.0, 0.0] * 3


def test_beyond_the_day_before_a_forecast_falls_back():
    """Neither the step nor a day before it: the fallback, or the nearest
    value when there is none; NaN for no forecast at all."""

    forecast = {QUARTER: 12.0}
    times = [QUARTER - timedelta(minutes=15), QUARTER, QUARTER + timedelta(hours=1)]

    assert StateManager.extend_forecast(forecast, times, 0.0) == [0.0, 12.0, 0.0]
    assert StateManager.extend_forecast(forecast, times) == [12.0, 12.0, 12.0]
    assert all(math.isnan(v) for v in StateManager.extend_forecast({}, times))
