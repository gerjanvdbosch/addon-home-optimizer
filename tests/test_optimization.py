import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.optimization import Optimization
from app.state import StateManager
from domain.config import BoilerConfig, LegionellaConfig, PricesConfig
from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.sensors import SensorReference
from domain.state import SeriesPoint
from domain.time import local_day_start
from features.optimizer import MPCOptimizer

START = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)
TIMES = [START + timedelta(minutes=15 * i) for i in range(4)]
TEMPERATURES = (40.0, 44.0, 48.3, 48.2)


class _RecordingHomeAssistant:
    def __init__(self) -> None:
        self.states: dict[str, tuple[str, dict]] = {}

    def set_state(self, entity_id: str, state: str, attributes: dict) -> None:
        self.states[entity_id] = (state, attributes)


def _published(
    schedule: tuple[int, ...], thermal_model: BoilerThermalModel | None = None
) -> dict[str, str]:
    home_assistant = _RecordingHomeAssistant()
    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
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

    optimization.publish_dhw(result, TIMES, thermal_model)

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


def test_the_setpoint_is_the_planned_end_temperature_rounded_to_the_nearest():
    """The tank is at 48.2 degC after the run's last step, 48.3 degC after a
    one-step run - each to the heat pump's nearest half degree."""

    assert _published((0, 1, 1, 0))[Optimization.DHW_SETPOINT_ENTITY] == "48.0"
    assert _published((0, 1, 0, 0))[Optimization.DHW_SETPOINT_ENTITY] == "48.5"


def test_the_setpoint_leaves_the_overshoot_the_plan_counts():
    """A run planned to end at 48.2 degC with 1.7 K of overshoot is set to
    stop at 46.5; one ending above the heat pump's own limit ends on the
    booster's thermostat and keeps its end as setpoint."""

    model = BoilerThermalModel(
        volume_l=200.0,
        ua_top_w_per_k=1.0,
        ua_bottom_w_per_k=1.0,
        ua_mix_idle_w_per_k=0.1,
        ua_mix_active_w_per_k=500.0,
        q_in_nominal_w=6000.0,
        setpoint_overshoot_k=1.7,
    )

    assert _published((0, 1, 1, 0), model)[Optimization.DHW_SETPOINT_ENTITY] == "46.5"
    assert (
        _published((0, 1, 1, 0), replace(model, heat_pump_max_tank_temperature_c=48.0))[
            Optimization.DHW_SETPOINT_ENTITY
        ]
        == "48.0"
    )


LEGIONELLA = LegionellaConfig(temperature=60.0, interval=7)
STEPS = 12
NOW = datetime.now(UTC).replace(second=0, microsecond=0)
PLAN_TIMES = [NOW + timedelta(minutes=15 * i) for i in range(STEPS)]
# Sun only in steps 6-9. From 50 degC the run takes the heat pump 12 minutes to
# its 55 degC limit and the booster 35 more to 60: four quarter hours.
SOLAR_W = [0.0] * 6 + [2000.0] * 4 + [0.0] * 2
TANK = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=1.0,
    ua_bottom_w_per_k=1.0,
    ua_mix_idle_w_per_k=0.1,
    ua_mix_active_w_per_k=500.0,
    q_in_nominal_w=6000.0,
    heat_pump_max_tank_temperature_c=55.0,
    max_tank_temperature_c=61.0,
    booster_heat_w=2000.0,
)
DATA = MPCInput(
    solar_forecast_w=SOLAR_W,
    ambient_temperature=20.0,
    current_temp_top=50.0,
    current_temp_bottom=50.0,
    boiler_on_current=False,
    target_temperature_top=(45.0,) * STEPS,
)


def test_the_disinfection_goes_where_the_sun_covers_its_run():
    step = Optimization.disinfection_step(
        DATA, PLAN_TIMES, PLAN_TIMES[-1], TANK, 60.0, MPCConfig()
    )

    # The four steps before it are the four sunny ones.
    assert step == 10


def test_the_disinfection_is_done_by_its_deadline():
    step = Optimization.disinfection_step(
        DATA, PLAN_TIMES, PLAN_TIMES[8], TANK, 60.0, MPCConfig()
    )

    assert step == 8


def _targets_with_legionella(days_ago: int, top: float, bottom: float):
    # Local midnight, so the deadline falls on the same local day in any zone.
    day = local_day_start(NOW, days=-days_ago)
    loader = _DailyMaxLoader(
        pd.DataFrame({"time": [day], "top": [top], "bottom": [bottom]})
    )
    optimization = Optimization(
        loader=loader,  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=_RecordingHomeAssistant(),  # type: ignore[arg-type]
    )
    boiler = BoilerConfig.model_validate(
        {
            "setpoint": "sensor.setpoint",
            "top_temperature": "sensor.top",
            "bottom_temperature": "sensor.bottom",
            "ambient_temperature": "sensor.ambient",
            "target_temperature": 45.0,
            "legionella": LEGIONELLA.model_dump(),
        }
    )

    # A forecast to the end of tomorrow, as Solcast's is: its last quarter hour
    # starts at 23:45 and runs to the deadline at midnight.
    end = local_day_start(NOW, days=2)
    steps = int((end - NOW).total_seconds() // (15 * 60)) + 1
    data = MPCInput(
        solar_forecast_w=[0.0] * steps,
        ambient_temperature=20.0,
        current_temp_top=50.0,
        current_temp_bottom=50.0,
        boiler_on_current=False,
        target_temperature_top=(45.0,) * steps,
    )
    times = [NOW + timedelta(minutes=15 * i) for i in range(steps)]

    return optimization._with_legionella(
        data, times, boiler, TANK, MPCConfig()
    ).target_temperature_top


class _DailyMaxLoader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load(self, dataset, start, end) -> pd.DataFrame:
        return self.frame


def test_no_disinfection_is_planned_while_the_interval_runs():
    assert 60.0 not in _targets_with_legionella(1, 61.0, 60.5)


def test_a_disinfection_is_planned_on_the_day_the_interval_ends():
    assert 60.0 in _targets_with_legionella(7, 61.0, 60.5)


def test_an_interval_ending_tomorrow_is_planned_today_or_tomorrow():
    assert 60.0 in _targets_with_legionella(6, 61.0, 60.5)


def test_an_interval_ending_beyond_the_plan_plans_nothing_yet():
    """The interval ends at the end of the day after tomorrow, past the
    forecast: nothing to choose from yet."""

    assert 60.0 not in _targets_with_legionella(5, 61.0, 60.5)


def test_one_sensor_at_temperature_does_not_count_as_disinfected():
    assert 60.0 in _targets_with_legionella(1, 61.0, 58.0)


# The optimizer and input a four-step plan log is priced by: MPCConfig's flat
# prices.
LOG_PLANNER = (
    MPCOptimizer(TANK, MPCConfig()),
    replace(DATA, solar_forecast_w=[0.0] * 4, target_temperature_top=(45.0,) * 4),
)


def test_the_plan_log_reports_the_objective_s_own_energy_per_day(caplog):
    """1.00 kWh of which 0.50 from the grid, as the optimizer counted them:
    the import at the price, the sun at the export it forgoes - per day, and
    in total."""

    result = MPCResult(
        schedule=(0, 1, 1, 0),
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0, 2000.0, 2000.0, 0.0),
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        solver_status="optimal",
        termination_condition="optimal",
        electricity_kwh=1.0,
        grid_kwh=0.5,
        electricity_step_kwh=(0.0, 0.5, 0.5, 0.0),
        grid_step_kwh=(0.0, 0.25, 0.25, 0.0),
    )

    with caplog.at_level("INFO", logger="app.optimization"):
        Optimization.log_dhw_plan(result, TIMES, *LOG_PLANNER)

    assert "to 48.2 degC: 1.00 kWh, of which sun 0.50 and grid 0.50" in caplog.text
    assert "(expected), EUR 0.150 | total 1.00 kWh, EUR 0.150" in caplog.text


def test_the_plan_log_splits_its_energy_by_day(caplog):
    """A run today and one tomorrow: each day's own electricity and cost."""

    times = [START + timedelta(hours=12 * i) for i in range(4)]
    result = MPCResult(
        schedule=(1, 0, 1, 0),
        temperatures=TEMPERATURES,
        electrical_power_w=(2000.0, 0.0, 2000.0, 0.0),
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        solver_status="optimal",
        termination_condition="optimal",
        electricity_kwh=1.5,
        grid_kwh=1.0,
        electricity_step_kwh=(0.5, 0.0, 1.0, 0.0),
        grid_step_kwh=(0.0, 0.0, 1.0, 0.0),
    )

    with caplog.at_level("INFO", logger="app.optimization"):
        Optimization.log_dhw_plan(result, times, *LOG_PLANNER)

    assert ": 0.50 kWh, of which sun 0.50 and grid 0.00" in caplog.text
    assert ": 1.00 kWh, of which sun 0.00 and grid 1.00" in caplog.text
    assert "| total 1.50 kWh" in caplog.text


def test_explain_logs_the_plan_beside_earlier_finishes(caplog):
    """A 45 degC target at the end of six hours with sun only in the last
    hour and a half: the plan heats in it, and each earlier finish is solved and
    logged beside it."""

    steps = 24
    data = MPCInput(
        solar_forecast_w=[0.0] * (steps - 6) + [3000.0] * 6,
        ambient_temperature=20.0,
        current_temp_top=40.0,
        current_temp_bottom=40.0,
        boiler_on_current=False,
        target_temperature_top=(10.0,) * (steps - 1) + (45.0,),
    )
    times = [NOW + timedelta(minutes=15 * i) for i in range(steps)]
    optimizer = MPCOptimizer(replace(TANK, setpoint_overshoot_k=1.7), MPCConfig())

    with caplog.at_level("INFO", logger="app.optimization"):
        Optimization.explain_dhw_plan(optimizer, data, optimizer.solve(data), times)

    for line in ("plan:", "0.5 h earlier:", "1 h earlier:", "2 h earlier:"):
        assert line in caplog.text

    # Each finishes where the plan does, not an overshoot above it.
    ends = re.findall(r"to (\d+\.\d) degC", caplog.text)
    assert float(ends[1]) - float(ends[0]) < 0.5


DAY = datetime(2026, 9, 28, tzinfo=UTC)


def _changes(*changes: tuple[str, object]) -> list[SeriesPoint]:
    """Reported changes at "HH:MM:SS" on DAY."""

    return [
        SeriesPoint(time=DAY + pd.Timedelta(clock), value=value)
        for clock, value in changes
    ]


def test_a_run_under_way_began_at_its_change_into_dhw():
    """The state changed to DHW at 13:15:38; the compressor came up minutes
    later, the pump circulating at 0 Hz before it - the run's own start."""

    state = _changes(("09:00:00", "Uit"), ("13:15:38", "SWW"))
    frequency = _changes(("12:00:00", 0.0), ("13:18:00", 38.0), ("13:25:00", 57.0))

    assert Optimization.dhw_timing(
        state, frequency, "SWW", DAY + pd.Timedelta("13:45:00")
    ) == (True, DAY + pd.Timedelta("13:15:38"), 0.0)


def test_the_booster_is_no_compressor_run():
    """0 Hz after the compressor has run in this DHW stretch: the booster has
    taken over, and no compressor run is under way."""

    state = _changes(("09:00:00", "Uit"), ("13:15:38", "SWW"))
    frequency = _changes(("13:18:00", 57.0), ("13:55:00", 0.0))

    assert Optimization.dhw_timing(
        state, frequency, "SWW", DAY + pd.Timedelta("14:05:00")
    ) == (True, None, 0.0)


def test_the_pause_runs_from_the_change_that_ended_the_run():
    state = _changes(("13:15:38", "SWW"), ("13:53:30", "Uit"))

    assert Optimization.dhw_timing(
        state, [], "SWW", DAY + pd.Timedelta("14:23:30")
    ) == (False, None, 0.5)


def test_readings_on_a_grid_read_the_same_as_changes():
    """Quarter-hour readings, as before: the run began at its first DHW one."""

    state = _changes(("13:00:00", "Uit"), ("13:15:00", "SWW"), ("13:30:00", "SWW"))
    frequency = _changes(("13:15:00", 38.0), ("13:30:00", 57.0))

    assert Optimization.dhw_timing(
        state, frequency, "SWW", DAY + pd.Timedelta("13:40:00")
    ) == (True, DAY + pd.Timedelta("13:15:00"), 0.0)


def test_the_heat_pump_changes_are_read_per_series():
    """The loaded frame holds both series on their own change times: each
    keeps its own points, and a series with none is empty."""

    frame = pd.DataFrame(
        {
            "time": [DAY + pd.Timedelta("13:15:38"), DAY + pd.Timedelta("13:18:00")],
            "state": ["SWW", None],
            "frequency": [None, 38.0],
        }
    )
    optimization = Optimization.__new__(Optimization)
    optimization.loader = SimpleNamespace(load=lambda *_: frame)
    config = SimpleNamespace(
        heat_pump=SimpleNamespace(
            state=SensorReference(entity_id="state"),
            compressor_frequency=SensorReference(entity_id="frequency"),
        )
    )

    state, frequency = optimization._heat_pump_changes(config, DAY)

    assert state == _changes(("13:15:38", "SWW"))
    assert frequency == _changes(("13:18:00", 38.0))
    optimization.loader = SimpleNamespace(load=lambda *_: frame[["time", "state"]])
    assert optimization._heat_pump_changes(config, DAY)[1] == []


def test_a_settled_cooling_run_is_set_the_planned_heat_below_its_return():
    """4 kW taken from 20 L/min (1393 W/K) is 2.9 K under an 18.4 degC
    return: 15.5 degC, to the half degree."""

    assert Optimization.zone_setpoint_c(16.8, 4000.0, 18.4, 20.0, None, None) == 15.5


def test_a_cooling_setpoint_without_a_return_is_the_planned_supply():
    assert Optimization.zone_setpoint_c(16.8, 4000.0, None, None, None, None) == 17.0
    assert Optimization.zone_setpoint_c(16.8, 4000.0, 18.4, 0.0, None, None) == 17.0


def test_a_cooling_setpoint_never_rises_within_its_run():
    assert Optimization.zone_setpoint_c(16.8, 4000.0, 19.4, 20.0, None, 15.5) == 15.5


def test_a_cooling_setpoint_never_goes_under_its_minimum():
    """Rounded up to the half degree at or above a 15.2 degC dew point, even
    past an earlier setpoint of the run."""

    assert Optimization.zone_setpoint_c(16.8, 4000.0, 18.4, 20.0, 15.2, 15.0) == 15.5


def test_a_run_is_under_way_since_its_change_into_the_mode():
    changes = [
        SeriesPoint(time=START, value="Uit"),
        SeriesPoint(time=START + timedelta(minutes=7), value="Koelen"),
        SeriesPoint(time=START + timedelta(minutes=40), value="Koelen"),
    ]

    assert Optimization.running_since(changes, "Koelen") == TIMES[0] + timedelta(
        minutes=7
    )
    assert Optimization.running_since(changes, "Verwarmen") is None
    assert Optimization.running_since(changes[:1], "Koelen") is None


def test_the_weekend_tariff_holds_all_saturday():
    """A high and low tariff on weekdays and the low one all weekend: Friday
    noon is high, Saturday noon low."""

    prices = PricesConfig.model_validate(
        {"import": {"weekdays": [["07:00", 0.25], ["23:00", 0.21]], "weekend": 0.21}}
    )
    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
        state_manager=StateManager(
            loader=None,  # type: ignore[arg-type]
            state_repository=None,  # type: ignore[arg-type]
            config_repository=None,  # type: ignore[arg-type]
            models_path=Path("."),
            latitude=52.0,
            longitude=5.0,
        ),
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=None,  # type: ignore[arg-type]
    )
    # Friday 2 October 2026, local noon, and the day after.
    friday_noon = local_day_start(datetime(2026, 10, 2, 12, tzinfo=UTC)) + timedelta(
        hours=12
    )

    assert optimization._price(
        prices.import_, [friday_noon, friday_noon + timedelta(days=1)]
    ) == (0.25, 0.21)
