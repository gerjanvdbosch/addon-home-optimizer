import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from app.state import StateManager
from domain.config import HeatPumpStates
from domain.models import BoilerThermalModel
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


def test_the_thermal_mass_ahead_is_the_plans(tmp_path):
    """The building model's estimate holds up to now; from there the slab and
    its screed are the ones the plan heats, not the zone left alone - which
    showed the slab cooling all through a planned floor run."""
    manager = _state_manager()
    manager.state_repository = StateRepository(JsonStorage(tmp_path / "state.json"))
    times = [NOW + timedelta(minutes=15 * i) for i in range(3)]
    past = NOW - timedelta(minutes=15)
    manager.update_zone(
        temperature=pd.Series([20.0], index=[past]),
        thermal_mass=pd.Series(
            [20.5, 20.4, 20.3, 20.2], index=[past, *times], dtype=float
        ),
        screed=pd.Series([21.0, 20.9, 20.8, 20.7], index=[past, *times], dtype=float),
    )

    manager.update_building_schedule(
        heat_w=[3000.0, 3000.0, 0.0],
        temperatures=[20.0, 20.1, 20.2],
        times=times,
        thermal_mass=[20.4, 20.8, 21.1],
        screed=[20.9, 22.0, 22.6],
    )

    predictions = manager.load().predictions
    assert [(p.time, p.value) for p in predictions.thermal_mass] == [
        (past, 20.5),
        *zip(times, [20.4, 20.8, 21.1], strict=True),
    ]
    assert [(p.time, p.value) for p in predictions.screed] == [
        (past, 21.0),
        *zip(times, [20.9, 22.0, 22.6], strict=True),
    ]


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


def test_a_run_the_quarters_miss_still_mixes_the_tank():
    """The quarters read each one's first state, and a state reports only its
    changes: a run that began inside a quarter reads as off throughout. By the
    state's own changes the tank mixed, and its cold layer is gone."""

    start = datetime(2026, 10, 18, 14, 0, tzinfo=UTC)
    times = [start + timedelta(minutes=15 * i) for i in range(6)]
    df = pd.DataFrame(
        {
            "time": times,
            "heat_pump_state": ["Uit"] * 6,
            "boiler_top_temperature": [37.0, 33.5, 33.5, 42.5, 47.5, 47.5],
            "boiler_bottom_temperature": [29.0, 29.0, 32.5, 42.5, 47.5, 47.5],
        }
    )
    existing = State()
    existing.measurements.heat_pump.state_changes = [
        SeriesPoint(time=start + timedelta(minutes=20), value="SWW"),
        SeriesPoint(time=start + timedelta(minutes=70), value="Uit"),
    ]
    config = SimpleNamespace(
        heat_pump=SimpleNamespace(
            states=HeatPumpStates(),
            boiler=SimpleNamespace(target_temperature=45.0),
            mode=None,
        ),
        building=SimpleNamespace(comfort=lambda cooling: None),
    )

    state = _state_manager()._map(df, existing, config=config)  # type: ignore[arg-type]

    assert [p.value for p in state.measurements.heat_pump.boiler.stratification] == [
        8.0,
        4.5,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def test_a_quarters_stratification_is_held_across_updates():
    """29 October 2026 at rest after a tap: the quarter's last reading so far
    showed 46.0/44.0 degC at 21:05, 45.5/44.0 at 21:10. The 2.0 K the first
    update held stays: the layer did not shrink, only the top stepped down."""

    times = [
        datetime(2026, 10, 29, 19, 45, tzinfo=UTC),
        datetime(2026, 10, 29, 20, 0, tzinfo=UTC),
    ]
    config = SimpleNamespace(
        heat_pump=SimpleNamespace(
            states=HeatPumpStates(),
            boiler=SimpleNamespace(target_temperature=45.0),
            mode=None,
        ),
        building=SimpleNamespace(comfort=lambda cooling: None),
    )

    def update(top: float, bottom: float, existing: State) -> State:
        df = pd.DataFrame(
            {
                "time": times,
                "heat_pump_state": ["Uit"] * 2,
                "boiler_top_temperature": [46.5, top],
                "boiler_bottom_temperature": [45.5, bottom],
            }
        )

        return _state_manager()._map(df, existing, config=config)  # type: ignore[arg-type]

    state = update(45.5, 44.0, update(46.0, 44.0, State()))

    assert [p.value for p in state.measurements.heat_pump.boiler.stratification] == [
        1.0,
        2.0,
    ]


def test_a_run_ends_the_stratification_held_before_it():
    """30 October 2026: the 8 K before a run was held on into the forecast's
    quarters and came back once they were measured, after the run had mixed
    the tank - 36 degC read at 48/48. Only measured quarters are kept."""

    times = [
        datetime(2026, 10, 30, 12, 15, tzinfo=UTC) + timedelta(minutes=15 * i)
        for i in range(3)
    ]
    config = SimpleNamespace(
        heat_pump=SimpleNamespace(
            states=HeatPumpStates(),
            boiler=SimpleNamespace(target_temperature=45.0),
            mode=None,
        ),
        building=SimpleNamespace(comfort=lambda cooling: None),
    )

    def update(state: list[str], top: list[float], bottom: list[float], existing):
        df = pd.DataFrame(
            {
                "time": times,
                "heat_pump_state": state,
                "boiler_top_temperature": top,
                "boiler_bottom_temperature": bottom,
            }
        )

        return _state_manager()._map(df, existing, config=config)  # type: ignore[arg-type]

    nan = float("nan")
    before = update(["Uit"] * 3, [22.0, nan, nan], [14.0, nan, nan], State())
    state = update(
        ["Uit", "SWW", "Uit"], [22.0, 48.5, 48.0], [14.0, 48.5, 48.0], before
    )

    assert [p.value for p in before.measurements.heat_pump.boiler.stratification] == [
        8.0
    ]
    assert [p.value for p in state.measurements.heat_pump.boiler.stratification] == [
        8.0,
        0.0,
        0.0,
    ]


RUN_START = datetime(2026, 10, 27, 10, 0, 49, tzinfo=UTC)
# 27 Oct 2026, the quarters around a run from 10:00:49 to 10:53:06, each read
# at its end: as it starts its circulation stirs the cold layer up and the top
# sensor drops.
TANK = pd.DataFrame(
    {
        "time": [
            datetime(2026, 10, 27, 9, 45, tzinfo=UTC) + timedelta(minutes=15 * i)
            for i in range(5)
        ],
        "boiler_top_temperature": [35.5, 29.0, 34.5, 44.0, 48.5],
        "boiler_bottom_temperature": [24.5, 24.5, 34.0, 44.0, 48.5],
        "boiler_ambient_temperature": [21.0] * 5,
        "boiler_stratification": [15.5, 4.5, 0.5, 0.0, 0.0],
    }
)


def test_a_run_holds_the_tank_at_no_less_than_before_it():
    """Without a calibrated model the sensors' average: through the run never
    below the 30.0 it began from, after it the sensors again."""

    changes = [
        SeriesPoint(time=RUN_START, value="SWW"),
        SeriesPoint(time=datetime(2026, 10, 27, 10, 53, 6, tzinfo=UTC), value="Uit"),
    ]

    assert _state_manager()._tank_temperature(TANK, changes, "SWW").tolist() == [
        30.0,
        30.0,
        34.25,
        44.0,
        48.5,
    ]


def test_a_run_under_way_leaves_the_tank_to_the_plan(tmp_path):
    """Its lagging sensors not drawn; before it, the tank mixed."""

    model = BoilerThermalModel(
        volume_l=200.0,
        ua_top_w_per_k=1.0,
        ua_bottom_w_per_k=1.0,
        ua_mix_idle_w_per_k=0.1,
        ua_mix_active_w_per_k=500.0,
        q_in_nominal_w=6000.0,
        cold_layer_fraction=0.48,
        cold_layer_above_ambient_k=0.69,
        cold_layer_spread_k=2.38,
    )
    joblib.dump(model, tmp_path / "boiler.joblib")
    manager = _state_manager()
    manager.models_path = tmp_path

    tank = manager._tank_temperature(
        TANK, [SeriesPoint(time=RUN_START, value="SWW")], "SWW"
    )

    assert tank[0] == pytest.approx(model.mixed_temperature(35.5, 24.5, 21.0, 15.5))
    assert tank[1:].isna().all()


def test_the_plan_line_starts_at_the_tank_now(tmp_path):
    manager = _state_manager()
    manager.state_repository = StateRepository(JsonStorage(tmp_path / "state.json"))
    times = [NOW.replace(minute=30) + timedelta(minutes=15 * i) for i in range(3)]

    manager.update_schedule(
        schedule=[1, 1, 0],
        temperatures=[40.0, 44.0, 47.0],
        power_w=[1500.0] * 3,
        times=times,
        now=NOW,
    )

    boiler = manager.load().schedule.heat_pump.boiler
    assert [p.time for p in boiler.temperatures] == [NOW, *times[1:]]
    assert [p.time for p in boiler.on] == times


def test_through_a_dhw_run_the_plan_line_shows_only_quarters(tmp_path):
    manager = _state_manager()
    manager.state_repository = StateRepository(JsonStorage(tmp_path / "state.json"))
    times = [NOW.replace(minute=30) + timedelta(minutes=15 * i) for i in range(3)]

    manager.update_schedule(
        schedule=[1, 1, 0],
        temperatures=[40.0, 44.0, 47.0],
        power_w=[1500.0] * 3,
        times=times,
        now=NOW,
        dhw_running=True,
        run_temperatures=[SeriesPoint(time=times[0], value=42.0)],
    )

    boiler = manager.load().schedule.heat_pump.boiler
    assert [(p.time, p.value) for p in boiler.temperatures] == [
        (times[0], 42.0),
        (times[1], 44.0),
        (times[2], 47.0),
    ]


def test_a_run_the_booster_finishes_is_drawn_as_ended():
    """The compressor ran and is at 0 Hz in the run under way: the booster has
    taken over and the plan starts from the sensors, so they are drawn, never
    below the tank before the run."""

    changes = [SeriesPoint(time=RUN_START, value="SWW")]
    tank = TANK.assign(heat_pump_compressor_frequency=[0.0, 52.0, 60.0, 30.0, 0.0])

    assert _state_manager()._tank_temperature(tank, changes, "SWW").tolist() == [
        30.0,
        30.0,
        34.25,
        44.0,
        48.5,
    ]
