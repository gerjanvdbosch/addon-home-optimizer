from datetime import UTC, datetime, timedelta

import pytest

from domain.models import BoilerThermalModel
from domain.state import BoilerMeasurement, SeriesPoint, State
from web.charts import continued, dashboard_chart, ended, mixed_tank, read_at

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


def test_without_measurements_the_plan_is_drawn_whole():
    planned = _points([31.0, 46.0])

    assert continued([], planned) == planned


def test_the_running_quarters_power_gives_way_to_the_plan():
    """A run started in the running quarter measured 0 W over its first
    second; the plan's 1870 W for that quarter is drawn instead."""

    measured = ended(_points([563.0, 0.0]), START + timedelta(minutes=15, seconds=1))
    planned = _points([1870.0, 2214.0], first=1)

    assert [p.value for p in continued(measured, planned)] == [563.0, 1870.0, 2214.0]


def test_the_dashboard_renders_without_any_data():
    assert "plotly" in dashboard_chart(State())


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
    """At the quarter's end, and at the update in the quarter still running."""

    updated = START + timedelta(minutes=20)

    assert [p.time for p in read_at(_points([31.0, 41.0]), updated)] == [
        START + timedelta(minutes=15),
        updated,
    ]
