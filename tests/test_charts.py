import math
from datetime import UTC, datetime, timedelta

import pytest

from domain.state import SeriesPoint, State
from domain.time import to_local_time
from web.charts import (
    broken_between_runs,
    dashboard_chart,
    ended,
    handed_over,
    joined,
    on_clock,
    read_at,
    spans,
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


def test_the_hour_the_clocks_go_back_is_drawn_once():
    """25 Oct 2026, 03:00 CEST becomes 02:00 CET: the line runs from 02:45 on
    to 03:00, not back to 02:00 over itself."""

    first = datetime(2026, 10, 24, 23, 45, tzinfo=UTC)  # 01:45 CEST
    points = [
        SeriesPoint(time=first + timedelta(minutes=15 * i), value=float(i))
        for i in range(10)
    ]
    clock = [to_local_time(p.time).strftime("%H:%M") for p in on_clock(points)]

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

    assert html.count('"legendgroup":"zone"') == 3
    # Filled up to the band's upper edge, as Solcast's is.
    assert html.count('"fill":"tonexty"') == 2


def test_the_dashboard_renders_the_boiler():
    state = State()
    state.measurements.heat_pump.boiler.top_temperature = _points([45.0, 46.0])
    state.measurements.heat_pump.boiler.temperature = _points([41.0, 46.0])

    assert "Boiler bottom" in dashboard_chart(state)


def test_a_quarters_last_reading_is_drawn_when_it_was_read():
    """At the quarter's end; the quarter still running is left to the plan."""

    updated = START + timedelta(minutes=20)

    assert [p.time for p in read_at(_points([31.0, 41.0]), updated)] == [
        START + timedelta(minutes=15)
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
