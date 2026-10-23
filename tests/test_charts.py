import math
import time
from datetime import UTC, datetime, timedelta

import pytest

from domain.models import BoilerThermalModel
from domain.state import BoilerMeasurement, SeriesPoint, State
from domain.time import to_local_time
from web.charts import (
    broken_between_runs,
    dashboard_chart,
    ended,
    handed_over,
    joined,
    mixed_tank,
    on_clock,
    read_at,
    spans,
    tank_line,
)

START = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _points(values: list[float], first: int = 0) -> list[SeriesPoint]:
    return [
        SeriesPoint(time=START + timedelta(minutes=15 * (first + i)), value=v)
        for i, v in enumerate(values)
    ]


def test_one_line_runs_from_the_measurements_into_the_plan():
    """Everything measured, then the plan's points after the last measurement -
    a plan made before it overlaps nothing."""

    measured = _points([31.0, 37.0])
    planned = _points([31.0, 34.0, 46.0])

    assert [p.value for p in joined(measured, planned)] == [31.0, 37.0, 46.0]
    assert joined([], planned) == planned


def test_the_running_quarters_power_gives_way_to_the_plan():
    """A run started in the running quarter measured 0 W over its first
    second; the plan's 1870 W for that quarter is drawn instead."""

    measured = ended(_points([563.0, 0.0]), START + timedelta(minutes=15, seconds=1))
    planned = _points([1870.0, 2214.0], first=1)

    assert [p.value for p in joined(measured, planned)] == [563.0, 1870.0, 2214.0]


def test_a_planned_supply_stops_between_runs():
    """Two runs a quarter hour apart in the plan's steps: one line each, not
    one line through the hour nothing ran."""

    supply = _points([14.5, 14.3]) + _points([15.0], first=6)
    broken = broken_between_runs(supply, timedelta(minutes=15))

    assert [p.value for p in broken][:2] == [14.5, 14.3]
    assert math.isnan(broken[2].value)
    assert broken[2].time == supply[1].time + timedelta(minutes=15)
    assert broken[3] == supply[2]


def test_the_hour_the_clocks_go_back_is_drawn_once(monkeypatch):
    """25 Oct 2026, 03:00 CEST becomes 02:00 CET: the line runs from 02:45 on
    to 03:00, not back to 02:00 over itself."""

    monkeypatch.setenv("TZ", "Europe/Amsterdam")
    time.tzset()
    first = datetime(2026, 10, 24, 23, 45, tzinfo=UTC)  # 01:45 CEST
    points = [
        SeriesPoint(time=first + timedelta(minutes=15 * i), value=float(i))
        for i in range(10)
    ]
    clock = [to_local_time(p.time).strftime("%H:%M") for p in on_clock(points)]
    monkeypatch.undo()
    time.tzset()

    assert clock == ["01:45", "02:00", "02:15", "02:30", "02:45", "03:00"]


def test_the_tanks_last_quarter_runs_on_into_the_zone():
    """The tank done after a tenth of the quarter (720 of 7200 W), the zone
    from the next: the rest of the quarter drawn at the zone's power, as the
    heat pump runs it - not the idle dip the plan's one demand per quarter
    leaves."""

    state = State()
    heat_pump, building = state.schedule.heat_pump, state.schedule.building
    heat_pump.power = _points([2000.0, 200.0, 800.0, 800.0])
    heat_pump.heat = _points([7200.0, 720.0, 4500.0, 4500.0])
    heat_pump.boiler.on = _points([1, 1, 0, 0])
    building.on = _points([0, 0, 1, 1])

    drawn = [p.value for p in handed_over(state.schedule)]

    assert drawn == pytest.approx([2000.0, 200.0 + 0.9 * 800.0, 800.0, 800.0])


def test_the_dashboard_renders_without_any_data():
    assert "plotly" in dashboard_chart(State())


def test_the_dashboard_draws_a_cooling_plan():
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    times = [now + timedelta(minutes=15 * i) for i in range(12)]
    state = State()
    state.schedule.building.supply = [
        SeriesPoint(time=t, value=14.5) for t in times[:4]
    ] + [SeriesPoint(time=times[8], value=15.0)]
    # Measured before now, forecast after: one line through both.
    measured_at = now - timedelta(minutes=37)
    state.measurements.building.dew_point = [SeriesPoint(time=measured_at, value=15.2)]
    state.predictions.dew_point = [SeriesPoint(time=t, value=15.6) for t in times]

    html = dashboard_chart(state)

    assert "Supply plan" in html
    assert html.count('"name":"Dew point"') == 1
    assert to_local_time(measured_at).isoformat() in html


def test_the_dashboard_draws_the_comfort_band():
    """The band filled between its edges, the wanted temperature within it,
    all three toggled together."""

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    times = [now + timedelta(minutes=15 * i) for i in range(4)]
    state = State()
    building = state.schedule.building

    for points, value in (
        ("minimum_temperature", 19.0),
        ("target_temperature", 20.0),
        ("maximum_temperature", 22.0),
    ):
        setattr(building, points, [SeriesPoint(time=t, value=value) for t in times])

    html = dashboard_chart(state)

    assert html.count('"legendgroup":"climate"') == 3
    # Filled up to the band's upper edge, as Solcast's is.
    assert html.count('"fill":"tonexty"') == 2


MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=1.0,
    ua_bottom_w_per_k=1.0,
    ua_mix_idle_w_per_k=0.1,
    ua_mix_active_w_per_k=500.0,
    q_in_nominal_w=6000.0,
    cold_layer_fraction=0.35,
    cold_layer_above_ambient_k=-5.0,
    cold_layer_spread_k=8.0,
)
BOILER = BoilerMeasurement(
    top_temperature=_points([45.0, 46.0]),
    bottom_temperature=_points([37.0, 46.0]),
    ambient_temperature=_points([20.0, 20.0]),
)


def test_the_measured_tank_is_drawn_mixed_as_it_is_planned():
    """Stratified, the tank holds its cold layer; mixed after a run, it is the
    sensors' average."""

    drawn = mixed_tank(BOILER, MODEL)

    assert drawn[0].value == pytest.approx(MODEL.mixed_temperature(45.0, 37.0, 20.0))
    assert drawn[1].value == 46.0


def test_without_a_calibrated_model_the_sensors_average_is_drawn():
    assert [p.value for p in mixed_tank(BOILER, None)] == [41.0, 46.0]


def test_the_dashboard_renders_with_a_calibrated_boiler():
    state = State()
    state.measurements.heat_pump.boiler = BOILER

    assert "Boiler bottom" in dashboard_chart(state, MODEL)


def test_a_quarters_last_reading_is_drawn_when_it_was_read():
    """At the quarter's end; the quarter still running is left to the plan."""

    updated = START + timedelta(minutes=20)

    assert [p.time for p in read_at(_points([31.0, 41.0]), updated)] == [
        START + timedelta(minutes=15)
    ]


def test_a_run_under_way_is_drawn_as_the_state_has_it_not_its_sensors():
    """Half an hour into a run, the sensors still read the tank as it was:
    the line leaves them where the run began, for the run as the state has
    it, then the plan."""

    state = State()
    boiler = state.schedule.heat_pump.boiler
    boiler.run = [
        SeriesPoint(time=START + timedelta(minutes=20), value=29.2),
        SeriesPoint(time=START + timedelta(minutes=30), value=31.0),
        SeriesPoint(time=START + timedelta(minutes=45), value=37.5),
        SeriesPoint(time=START + timedelta(minutes=50), value=40.8),
    ]
    boiler.temperatures = _points([40.8, 45.6], first=3)
    measured = _points([29.3, 29.2, 28.2], first=1)

    assert [p.value for p in tank_line(measured, state)] == [
        29.3,
        29.2,
        31.0,
        37.5,
        40.8,
        45.6,
    ]

    # Ended, the sensors take over again after it.
    measured += _points([44.0], first=4)

    assert [p.value for p in tank_line(measured, state)][-2:] == [40.8, 44.0]

    # Without a run, the sensors are the tank.
    boiler.run = []

    assert [p.value for p in tank_line(measured, state)] == [
        29.3,
        29.2,
        28.2,
        44.0,
    ]


def test_each_state_holds_until_its_next_change():
    """A run from its change to the next, to the second; the last state up to
    the update."""

    changes = [
        SeriesPoint(time=START, value="Uit"),
        SeriesPoint(time=START + timedelta(minutes=7, seconds=12), value="SWW"),
        SeriesPoint(time=START + timedelta(minutes=58), value="Uit"),
    ]
    updated = START + timedelta(hours=2)

    assert spans(changes, updated) == [
        (START, changes[1].time, "Uit"),
        (changes[1].time, changes[2].time, "SWW"),
        (changes[2].time, updated, "Uit"),
    ]


def test_the_dashboard_shades_what_the_heat_pump_serves():
    """A DHW run with a defrost in it, measured, then a planned heating run:
    one legend entry each, and nothing for the idle time."""

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    state = State(updated=now)
    heat_pump = state.measurements.heat_pump
    heat_pump.state_changes = [
        SeriesPoint(time=now - timedelta(minutes=50), value="SWW"),
        SeriesPoint(time=now - timedelta(minutes=5), value="Uit"),
    ]
    heat_pump.defrost = [
        SeriesPoint(time=now - timedelta(minutes=20), value=True),
        SeriesPoint(time=now - timedelta(minutes=17), value=False),
    ]
    state.schedule.building.on = [
        SeriesPoint(time=now + timedelta(minutes=15 * i), value=on)
        for i, on in enumerate([0, 1, 1, 0])
    ]

    html = dashboard_chart(state)

    for name in ("DHW", "Heating", "Defrost"):
        assert html.count(f'"name":"{name}"') == 1
    assert '"name":"Cooling"' not in html
