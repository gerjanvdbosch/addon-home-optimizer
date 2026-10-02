import math
from datetime import UTC, datetime, timedelta

import pytest

from domain.models import BoilerThermalModel
from domain.state import BoilerMeasurement, SeriesPoint, State
from domain.time import to_local_time
from web.charts import (
    broken_between_runs,
    continued,
    dashboard_chart,
    ended,
    joined,
    mixed_tank,
    read_at,
)

START = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _points(values: list[float], first: int = 0) -> list[SeriesPoint]:
    return [
        SeriesPoint(time=START + timedelta(minutes=15 * (first + i)), value=v)
        for i, v in enumerate(values)
    ]


def test_a_plan_runs_on_from_the_last_measurement():
    """Measured 31 and 37 degC; a plan made a quarter hour earlier still starts
    at 31 there. The line goes on from 37, with the plan's own later points."""

    measured = _points([31.0, 37.0])
    planned = _points([31.0, 34.0, 46.0])

    joined = continued(measured, planned)

    assert [p.value for p in joined] == [37.0, 46.0]
    assert joined[0].time == measured[-1].time


def test_one_line_runs_from_the_measurements_into_the_plan():
    """Everything measured, then the plan's points after the last measurement -
    a plan made before it overlaps nothing."""

    measured = _points([31.0, 37.0])
    planned = _points([31.0, 34.0, 46.0])

    assert [p.value for p in joined(measured, planned)] == [31.0, 37.0, 46.0]
    assert joined([], planned) == planned


def test_without_measurements_the_plan_is_drawn_whole():
    planned = _points([31.0, 46.0])

    assert continued([], planned) == planned


def test_the_running_quarters_power_gives_way_to_the_plan():
    """A run started in the running quarter measured 0 W over its first
    second; the plan's 1870 W for that quarter is drawn instead."""

    measured = ended(_points([563.0, 0.0]), START + timedelta(minutes=15, seconds=1))
    planned = _points([1870.0, 2214.0], first=1)

    assert [p.value for p in continued(measured, planned)] == [563.0, 1870.0, 2214.0]


def test_a_planned_supply_stops_between_runs():
    """Two runs a quarter hour apart in the plan's steps: one line each, not
    one line through the hour nothing ran."""

    supply = _points([14.5, 14.3]) + _points([15.0], first=6)
    broken = broken_between_runs(supply, timedelta(minutes=15))

    assert [p.value for p in broken][:2] == [14.5, 14.3]
    assert math.isnan(broken[2].value)
    assert broken[2].time == supply[1].time + timedelta(minutes=15)
    assert broken[3] == supply[2]


def test_the_dashboard_renders_without_any_data():
    assert "plotly" in dashboard_chart(State())


def test_the_dashboard_draws_a_cooling_plan():
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    times = [now + timedelta(minutes=15 * i) for i in range(12)]
    state = State()
    state.schedule.heat_pump.heat = [
        SeriesPoint(time=t, value=3571.0 if i < 4 else 0.0) for i, t in enumerate(times)
    ]
    state.schedule.building.supply = [
        SeriesPoint(time=t, value=14.5) for t in times[:4]
    ] + [SeriesPoint(time=times[8], value=15.0)]
    # Measured before now, forecast after: one line through both.
    measured_at = now - timedelta(minutes=37)
    state.measurements.building.dew_point = [SeriesPoint(time=measured_at, value=15.2)]
    state.predictions.dew_point = [SeriesPoint(time=t, value=15.6) for t in times]

    html = dashboard_chart(state)

    assert "Supply plan" in html
    assert "Heat pump heat" in html
    assert html.count('"name":"Dew point"') == 1
    assert to_local_time(measured_at).isoformat() in html


MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=1.0,
    ua_bottom_w_per_k=1.0,
    ua_mix_idle_w_per_k=0.1,
    ua_mix_active_w_per_k=500.0,
    q_in_nominal_w=6000.0,
    cold_layer_fraction=0.35,
    cold_water_temperature_c=15.0,
    cold_layer_spread_k=8.0,
)
BOILER = BoilerMeasurement(
    top_temperature=_points([45.0, 46.0]), bottom_temperature=_points([37.0, 46.0])
)


def test_the_measured_tank_is_drawn_mixed_as_it_is_planned():
    """Stratified, the tank holds its cold layer; mixed after a run, it is the
    sensors' average."""

    drawn = mixed_tank(BOILER, MODEL)

    assert drawn[0].value == pytest.approx(MODEL.mixed_temperature(45.0, 37.0))
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
