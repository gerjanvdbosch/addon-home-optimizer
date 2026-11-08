import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.optimization import Optimization
from app.state import StateManager
from domain.config import BoilerConfig, HeatPumpStates, LegionellaConfig, PricesConfig
from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.sensors import SensorReference
from domain.state import SeriesPoint, State, ZoneHabit
from domain.time import local_day_start, to_local_time
from features.boiler import BoilerThermalIdentifier
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
        termination_condition="optimal",
    )

    optimization.publish_dhw(result, TIMES, thermal_model)

    return {entity: state for entity, (state, _) in home_assistant.states.items()}


def _published_zone(
    tank: tuple[int, ...],
    zone: tuple[int, ...],
    cooling: bool = False,
    room_c: tuple[float, ...] = (),
) -> dict[str, str]:
    states = _zone_states(tank, zone, cooling, room_c)

    return {entity: state for entity, (state, _) in states.items()}


def _zone_states(
    tank: tuple[int, ...],
    zone: tuple[int, ...],
    cooling: bool = False,
    room_c: tuple[float, ...] = (),
    band_c: tuple[float, ...] = (),
    state: State | None = None,
    heat_pump_state: list[SeriesPoint] | None = None,
    maximum_c: tuple[float, ...] = (),
    rest_minimum_c: float | None = None,
) -> dict[str, tuple[str, dict]]:
    home_assistant = _RecordingHomeAssistant()
    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
        state_manager=SimpleNamespace(update_supply_setpoint=lambda point: None),  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=home_assistant,  # type: ignore[arg-type]
    )
    result = MPCResult(
        schedule=tank,
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0,) * len(tank),
        heat_w=(0.0,) * len(tank),
        objective_value=0.0,
        termination_condition="optimal",
        space_schedule=zone,
        space_heat_w=tuple(3000.0 * on for on in zone),
        space_supply_c=tuple(26.0 if on else float("nan") for on in zone),
        zone_temperatures=room_c,
    )
    data = MPCInput(
        solar_forecast_w=[0.0] * len(tank),
        ambient_temperature=20.0,
        current_temp_top=45.0,
        current_temp_bottom=45.0,
        boiler_on_current=bool(tank[0]),
        zone_cooling=cooling,
        zone_maximum_temperature=band_c if cooling else maximum_c,
        zone_minimum_temperature=() if cooling else band_c,
    )
    config = SimpleNamespace(
        heat_pump=SimpleNamespace(states=HeatPumpStates(), auto_adaptive=False),
        building=SimpleNamespace(
            rest_temperature=rest_minimum_c, rest_idle_temperature=15.0
        ),
    )

    optimization.publish_zone(
        result,
        TIMES,
        state or State(),
        config,  # type: ignore[arg-type]
        data,
        heat_pump_state or [],
        START,
    )

    return home_assistant.states


def test_the_rest_is_given_its_minimum_while_a_run_is_served():
    """The setpoint's attribute "rest": the rest's minimum while a run is
    served - handed over from the tank too - so its rooms below it open their
    loops; the idle setpoint otherwise, cooling too; absent without a minimum,
    the rooms' thermostats then left to themselves."""

    def upstairs(tank, zone, minimum_c=18.5) -> float | None:
        states = _zone_states(tank, zone, rest_minimum_c=minimum_c)
        return states[Optimization.ZONE_SETPOINT_ENTITY][1].get("rest", "absent")

    assert upstairs((0, 0, 0, 0), (1, 1, 0, 0)) == 18.5
    assert upstairs((1, 0, 0, 0), (0, 1, 1, 0)) == 18.5
    assert upstairs((0, 0, 0, 0), (0, 0, 1, 0)) == 15.0
    assert upstairs((0, 0, 0, 0), (1, 1, 0, 0), None) == "absent"


def test_between_runs_the_thermostat_is_given_the_comfort_bands_edge():
    """Off, the setpoint is the band's edge now, on the thermostat's step and
    inside the band - heat only below its minimum, cold only above its
    maximum."""

    def idle(band_c: float, cooling: bool = False) -> str:
        states = _zone_states(
            (0, 0, 0, 0), (0, 0, 1, 0), cooling, band_c=(band_c, 21.0, 21.0, 21.0)
        )
        return states[Optimization.ZONE_SETPOINT_ENTITY][0]

    assert idle(18.5) == "18.5"
    assert idle(18.7) == "18.5"
    assert idle(25.2, cooling=True) == "25.5"


def test_the_zone_is_on_in_the_quarter_the_tank_hands_over_in():
    """The heat pump finishes the hot water first; with heating already on it
    goes straight on to the floor rather than stopping and starting again."""

    published = _published_zone(
        tank=(1, 0, 0, 0), zone=(0, 1, 1, 0), room_c=(19.2, 19.24, 19.6, 19.83)
    )

    # Set up, but on only from its own start.
    assert published[Optimization.ZONE_STATUS_ENTITY] == "off"
    assert published[Optimization.ZONE_START_ENTITY] == TIMES[1].isoformat()
    assert published[Optimization.ZONE_SETPOINT_ENTITY] == "20.0"
    assert published[Optimization.ZONE_SUPPLY_ENTITY] == "26.0"


def test_the_zone_setpoint_is_past_the_room_the_next_run_heats_to():
    """During a run the least step past the warmest the plan has the room from
    the run's start to the next run's - after the run too, as the screed warms
    the room later - and unknown without a run or a band."""

    published = _published_zone(
        tank=(0, 0, 0, 0), zone=(1, 1, 0, 0), room_c=(19.2, 19.24, 19.6, 19.83)
    )

    assert published[Optimization.ZONE_SETPOINT_ENTITY] == "20.0"
    assert (
        _published_zone(tank=(0, 0, 0, 0), zone=(0, 0, 0, 0), room_c=(19.2,) * 4)[
            Optimization.ZONE_SETPOINT_ENTITY
        ]
        == "unknown"
    )


FALLING = (19.75, 19.7, 19.65, 19.6)


@pytest.mark.parametrize(
    ("schedule", "temperatures", "measured_c", "limit_c", "cooling", "expected"),
    [
        # A run under way past a room warmer than the plan has it, which it
        # heats for later: a whole step past the room, or the thermostat asks
        # for nothing as soon as the sun warms its air. In half degrees, the
        # thermostat's step.
        ((1, 1, 0, 0), FALLING, 19.82, None, False, "20.5"),
        # At a step of its own: still strictly past it.
        ((1, 1, 0, 0), FALLING, 20.0, None, False, "20.5"),
        # The next run's temperatures are not this run's.
        ((1, 0, 1, 1), (19.75, 19.7, 19.65, 20.4), None, None, False, "20.0"),
        # Never past the comfort band.
        ((1, 1, 0, 0), FALLING, 20.1, (20.0,) * 4, False, "20.0"),
        ((1, 1, 0, 0), FALLING, 19.82, (19.5,) * 4, False, "19.5"),
        # Cooling the other way.
        ((1, 1, 0, 0), FALLING, None, None, True, "19.5"),
    ],
)
def test_the_zone_setpoint(
    schedule, temperatures, measured_c, limit_c, cooling, expected
):
    assert (
        Optimization.zone_setpoint_c(
            schedule, 0, temperatures, measured_c, limit_c, cooling
        )
        == expected
    )


def test_a_tank_block_within_the_run_does_not_end_its_setpoint():
    """Floor, hot water and floor again are one compressor run: its setpoint
    looks past the hot water to where the floor stops."""

    temperatures = (19.2, 19.3, 19.35, 19.8)

    assert (
        Optimization.zone_setpoint_c(
            (1, 0, 1, 1), 0, temperatures, None, None, False, tank=(0, 1, 0, 0)
        )
        == "20.0"
    )
    assert (
        Optimization.zone_setpoint_c((1, 0, 1, 1), 0, temperatures, None, None, False)
        == "19.5"
    )


def test_a_run_under_way_holds_the_thermostats_setpoint():
    """A replan that lowers the run's setpoint publishes the thermostat's, which
    the blueprint only raises within a run; above it, the plan's."""

    def published(thermostat_c: float) -> str:
        state = State()
        state.measurements.building.setpoint = [
            SeriesPoint(time=START, value=thermostat_c)
        ]
        states = _zone_states(
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            room_c=(19.2, 19.24, 19.6, 19.83),
            state=state,
            heat_pump_state=[
                SeriesPoint(time=START - timedelta(hours=1), value="Verwarmen")
            ],
        )
        return states[Optimization.ZONE_SETPOINT_ENTITY][0]

    assert published(21.0) == "21.0"
    assert published(19.0) == "20.0"


def test_a_run_under_way_is_not_ended_by_the_band_at_the_thermostat():
    """The sun takes the room past the band's edge within a run: the setpoint
    stays a step past it, the plan ends the run; a run about to start is
    still held to the band."""

    def published(since: list[SeriesPoint]) -> str:
        state = State()
        state.measurements.building.zone_temperature = [
            SeriesPoint(time=START, value=20.6)
        ]
        states = _zone_states(
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            room_c=(19.2, 19.24, 19.6, 19.83),
            state=state,
            heat_pump_state=since,
            maximum_c=(20.5,) * 4,
        )
        return states[Optimization.ZONE_SETPOINT_ENTITY][0]

    under_way = [SeriesPoint(time=START - timedelta(hours=1), value="Verwarmen")]

    assert published(under_way) == "21.5"
    assert published([]) == "20.5"


def test_the_zone_is_set_up_from_the_tanks_start_when_its_run_follows():
    """Its setpoints from the hot water's start, so a tank done sooner than
    planned goes straight on to the floor too; the run's supply its own first
    step's, its status and start its own."""

    published = _published_zone(
        tank=(1, 1, 0, 0), zone=(0, 0, 1, 1), room_c=(19.2, 19.24, 19.6, 19.83)
    )

    assert published[Optimization.ZONE_STATUS_ENTITY] == "off"
    assert published[Optimization.ZONE_START_ENTITY] == TIMES[2].isoformat()
    assert published[Optimization.ZONE_SETPOINT_ENTITY] == "20.0"
    assert published[Optimization.ZONE_SUPPLY_ENTITY] == "26.0"


def test_no_hand_over_without_a_zone_run_straight_after_or_while_cooling():
    """Nothing set up for a run that does not follow the tank straight away."""

    for tank, zone, cooling in (
        ((1, 0, 0, 0), (0, 0, 1, 0), False),
        ((1, 1, 0, 0), (0, 0, 0, 1), False),
        ((1, 0, 0, 0), (0, 1, 1, 0), True),
    ):
        published = _published_zone(tank=tank, zone=zone, cooling=cooling)

        assert published[Optimization.ZONE_STATUS_ENTITY] == "off"
        assert published[Optimization.ZONE_SUPPLY_ENTITY] == "unknown"


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


def test_the_tank_level_is_its_heat_above_the_mains():
    """200 l at 31 degC over 15 degC mains: half of a tank at its 47 degC
    setpoint, 16 K x 200 kg x 4186 J/kg/K = 3.72 kWh; above the setpoint
    full, below the mains empty, and no share without a setpoint."""

    level = Optimization.tank_level(31.0, 15.0, 47.0, 200.0, 1.8e6)

    assert level == (50.0, {"available_kwh": 3.72, "drawn_today_kwh": 0.5})
    assert Optimization.tank_level(48.5, 15.0, 47.0, 200.0, 0.0)[0] == 100.0
    assert (
        Optimization.tank_level(14.5, 15.0, 47.0, 200.0, 0.0)[1]["available_kwh"] == 0.0
    )
    assert Optimization.tank_level(31.0, 15.0, None, 200.0, 0.0)[0] is None


def test_the_tank_level_is_its_own_sensor():
    home_assistant = _RecordingHomeAssistant()
    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=home_assistant,  # type: ignore[arg-type]
    )
    result = MPCResult(
        schedule=(0, 0, 0, 0),
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0,) * 4,
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        termination_condition="optimal",
    )

    optimization.publish_dhw(result, TIMES, tank_level=(50.0, {"available_kwh": 3.72}))

    state, attributes = home_assistant.states[Optimization.DHW_SOC_ENTITY]
    assert state == "50.0"
    assert attributes["unit_of_measurement"] == "%"
    assert attributes["available_kwh"] == 3.72
    assert (
        "available_kwh" not in home_assistant.states[Optimization.DHW_STATUS_ENTITY][1]
    )


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


def test_the_plan_log_reports_the_objective_s_own_energy_per_day():
    """1.00 kWh of which 0.50 from the grid, as the optimizer counted them:
    the import at the price, the sun at the export it forgoes - per day, and
    in total."""

    result = MPCResult(
        schedule=(0, 1, 1, 0),
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0, 2000.0, 2000.0, 0.0),
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        termination_condition="optimal",
        electricity_kwh=1.0,
        grid_kwh=0.5,
        electricity_step_kwh=(0.0, 0.5, 0.5, 0.0),
        grid_step_kwh=(0.0, 0.25, 0.25, 0.0),
    )

    summary = Optimization.dhw_summary(result, TIMES, *LOG_PLANNER)

    assert "to 48.2 degC: 1.00 kWh, of which sun 0.50 and grid 0.50" in summary
    assert "(expected), EUR 0.150 | total 1.00 kWh, EUR 0.150" in summary


def test_the_plan_log_splits_its_energy_by_day():
    """A run today and one tomorrow: each day's own electricity and cost."""

    times = [START + timedelta(hours=12 * i) for i in range(4)]
    result = MPCResult(
        schedule=(1, 0, 1, 0),
        temperatures=TEMPERATURES,
        electrical_power_w=(2000.0, 0.0, 2000.0, 0.0),
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        termination_condition="optimal",
        electricity_kwh=1.5,
        grid_kwh=1.0,
        electricity_step_kwh=(0.5, 0.0, 1.0, 0.0),
        grid_step_kwh=(0.0, 0.0, 1.0, 0.0),
    )

    summary = Optimization.dhw_summary(result, times, *LOG_PLANNER)

    assert ": 0.50 kWh, of which sun 0.50 and grid 0.00" in summary
    assert ": 1.00 kWh, of which sun 0.00 and grid 1.00" in summary
    assert "| total 1.50 kWh" in summary


def test_the_plan_log_counts_only_the_tank_s_electricity():
    """The plan's electricity is the whole heat pump's: a floor run after the
    tank's must not be reported as hot water."""

    result = MPCResult(
        schedule=(0, 1, 0, 0),
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0, 2000.0, 0.0, 0.0),
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        termination_condition="optimal",
        space_schedule=(0, 0, 1, 1),
        space_electrical_w=(0.0, 0.0, 1000.0, 1000.0),
        electricity_kwh=1.0,
        grid_kwh=0.5,
        electricity_step_kwh=(0.0, 0.5, 0.25, 0.25),
        grid_step_kwh=(0.0, 0.25, 0.25, 0.0),
    )

    summary = Optimization.dhw_summary(result, TIMES, *LOG_PLANNER)

    assert ": 0.50 kWh, of which sun 0.25 and grid 0.25" in summary
    assert "| total 0.50 kWh" in summary


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


def test_a_compressor_pause_mid_run_keeps_the_run():
    """0 Hz for two minutes mid-run, the compressor up again after (real
    data, 17 Oct 2026): one run from its change into DHW, not a new one from
    the restart."""

    state = _changes(("09:00:00", "Uit"), ("11:45:45", "SWW"))
    frequency = _changes(("11:49:00", 40.0), ("12:02:22", 0.0), ("12:04:32", 18.0))

    assert Optimization.dhw_timing(
        state, frequency, "SWW", DAY + pd.Timedelta("12:10:00")
    ) == (True, DAY + pd.Timedelta("11:45:45"), 0.0)


def test_a_compressor_stop_in_a_defrost_keeps_the_run():
    """0 Hz from inside a defrost, the defrost just over and the compressor
    not up yet (real data, 31 Oct 2026): the run goes on, not the booster."""

    state = _changes(("09:00:00", "Uit"), ("11:43:48", "SWW"))
    frequency = _changes(("11:47:00", 36.0), ("11:58:00", 0.0))
    defrost = _changes(("11:56:18", True), ("11:59:53", False))
    now = DAY + pd.Timedelta("12:00:00")

    assert Optimization.dhw_timing(state, frequency, "SWW", now, defrost) == (
        True,
        DAY + pd.Timedelta("11:43:48"),
        0.0,
    )
    assert Optimization.dhw_timing(state, frequency, "SWW", now) == (True, None, 0.0)


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


def test_a_runs_measured_heat_is_summed_to_each_trace_time(monkeypatch):
    """Per minute, from the run's start: 20 L/min warmed 5 K is 6977 W, the
    first minute's 0 K (no heat yet) - so 4 and 9 minutes of it by 10:05 and
    10:10. The minutes before the run carry none."""

    start = DAY + pd.Timedelta("10:00:00")
    minutes = pd.date_range(start - pd.Timedelta("3min"), periods=14, freq="1min")
    on = minutes >= start
    frame = pd.DataFrame(
        {
            "time": minutes,
            "T_ambient": 20.0,
            "T_top": 40.0,
            "T_bottom": 35.0,
            "state": ["SWW" if o else "Uit" for o in on],
            "T_supply": [45.0 if t > start else 40.0 for t in minutes],
            "T_return": 40.0,
            "flow_lpm": [20.0 if o else float("nan") for o in on],
        }
    )
    monkeypatch.setattr(
        BoilerThermalIdentifier,
        "dataset",
        lambda self, config: setattr(self, "states", HeatPumpStates()),
    )
    optimization = Optimization.__new__(Optimization)
    optimization.loader = SimpleNamespace(load=lambda *_: frame)
    times = [start, start + pd.Timedelta("5min"), start + pd.Timedelta("10min")]

    def heat_j():
        return optimization._run_heat_j(
            optimization._run_minutes(None, start, times[-1]), start, times
        )

    watts = 20.0 / 60.0 * 4186.0 * 5.0
    assert heat_j() == pytest.approx([0.0, 4 * 60 * watts, 9 * 60 * watts])

    frame["flow_lpm"] = float("nan")
    assert heat_j() is None


def test_a_settled_cooling_run_is_set_the_planned_heat_below_its_return():
    """4 kW taken from 20 L/min (1393 W/K) is 2.9 K under an 18.4 degC
    return: 15.5 degC, to the half degree."""

    assert (
        Optimization.zone_supply_c(16.8, -4000.0, 18.4, 20.0, None, None, True) == 15.5
    )


def test_a_cooling_setpoint_without_a_return_is_the_planned_supply():
    assert (
        Optimization.zone_supply_c(16.8, -4000.0, None, None, None, None, True) == 17.0
    )
    assert (
        Optimization.zone_supply_c(16.8, -4000.0, 18.4, 0.0, None, None, True) == 17.0
    )


def test_a_cooling_setpoint_never_rises_within_its_run():
    assert (
        Optimization.zone_supply_c(16.8, -4000.0, 19.4, 20.0, None, 15.5, True) == 15.5
    )


def test_a_cooling_setpoint_never_goes_under_its_minimum():
    """Rounded up to the half degree at or above a 15.2 degC dew point, even
    past an earlier setpoint of the run."""

    assert (
        Optimization.zone_supply_c(16.8, -4000.0, 18.4, 20.0, 15.2, 15.0, True) == 15.5
    )


def test_a_settled_heating_run_is_set_the_planned_heat_above_its_return():
    """4 kW into the floor from 20 L/min (1393 W/K) is 2.9 K over a 26.1 degC
    return: 29.0 degC, to the half degree."""

    assert (
        Optimization.zone_supply_c(30.0, 4000.0, 26.1, 20.0, None, None, False) == 29.0
    )


def test_a_heating_setpoint_never_falls_within_its_run():
    assert (
        Optimization.zone_supply_c(30.0, 4000.0, 25.1, 20.0, None, 29.0, False) == 29.0
    )


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


def test_a_floor_run_counts_from_the_compressors_start_on_the_zone():
    """From the change into heating, or the compressor's last start within it
    (the state stays on heating at 0 Hz between runs); nothing for a run the
    tank handed over."""

    def at(minutes: float) -> datetime:
        return START + timedelta(minutes=minutes)

    now = at(70)
    started = [
        SeriesPoint(time=at(0), value="Uit"),
        SeriesPoint(time=at(60), value="Verwarmen"),
    ]
    paused = [SeriesPoint(time=at(0), value="Verwarmen")]
    pause = [
        SeriesPoint(time=at(0), value=30.0),
        SeriesPoint(time=at(20), value=0.0),
        SeriesPoint(time=at(55), value=26.0),
    ]
    handed = [
        SeriesPoint(time=at(0), value="SWW"),
        SeriesPoint(time=at(60), value="Verwarmen"),
    ]
    running = [SeriesPoint(time=at(0), value=40.0)]

    def hours(states, frequency) -> float:
        return Optimization.zone_run_hours(states, frequency, "Verwarmen", "SWW", now)

    assert hours(started, running) == pytest.approx(10 / 60)
    assert hours(paused, pause) == pytest.approx(15 / 60)
    assert hours(handed, running) == 0.0


def test_a_floor_run_keeps_its_start_through_a_defrost():
    """0 Hz in a defrost is no new start: the run counts from the compressor's
    start before it, not from its restart after."""

    def at(minutes: float) -> datetime:
        return START + timedelta(minutes=minutes)

    states = [SeriesPoint(time=at(0), value="Verwarmen")]
    frequency = [
        SeriesPoint(time=at(0), value=0.0),
        SeriesPoint(time=at(10), value=35.0),
        SeriesPoint(time=at(30), value=0.0),
        SeriesPoint(time=at(33), value=30.0),
    ]
    defrost = [
        SeriesPoint(time=at(29), value=True),
        SeriesPoint(time=at(33), value=False),
    ]

    def hours(defrost) -> float:
        return Optimization.zone_run_hours(
            states, frequency, "Verwarmen", "SWW", at(40), defrost
        )

    assert hours(defrost) == pytest.approx(30 / 60)
    assert hours(()) == pytest.approx(7 / 60)


def test_a_floor_start_after_the_booster_is_no_hand_over():
    """The state on DHW at 0 Hz while the booster finished the tank, then
    heating and the compressor started from off: a cold start, counted from
    the compressor's start, not a hand-over."""

    def at(minutes: float) -> datetime:
        return START + timedelta(minutes=minutes)

    states = [
        SeriesPoint(time=at(0), value="SWW"),
        SeriesPoint(time=at(40), value="Verwarmen"),
    ]
    frequency = [
        SeriesPoint(time=at(0), value=50.0),
        SeriesPoint(time=at(30), value=0.0),
        SeriesPoint(time=at(45), value=30.0),
    ]

    assert Optimization.zone_run_hours(
        states, frequency, "Verwarmen", "SWW", at(60)
    ) == pytest.approx(15 / 60)


def test_a_floor_run_is_under_way_only_with_the_compressor_running():
    """The state stays on heating at 0 Hz between runs (real data, 11 Oct)."""

    heating = [SeriesPoint(time=START, value="Verwarmen")]
    running = [SeriesPoint(time=START, value=34.0)]
    stopped = [SeriesPoint(time=START, value=0.0)]

    assert Optimization.zone_running(heating, running, "Verwarmen")
    assert not Optimization.zone_running(heating, stopped, "Verwarmen")
    assert not Optimization.zone_running(heating, running, "Koelen")
    assert not Optimization.zone_running([], running, "Verwarmen")


def test_a_defrost_does_not_end_a_floor_run():
    """The compressor stops in a defrost and comes up again in the same run
    (7 Nov 2026: 0 Hz 16 minutes into a run); a stop outside one ends it."""

    heating = [SeriesPoint(time=START, value="Verwarmen")]
    stopped = [SeriesPoint(time=START + timedelta(minutes=16), value=0.0)]
    defrost = [
        SeriesPoint(time=START + timedelta(minutes=14), value=True),
        SeriesPoint(time=START + timedelta(minutes=20), value=False),
    ]

    assert Optimization.zone_running(heating, stopped, "Verwarmen", defrost)
    assert not Optimization.zone_running(heating, stopped, "Verwarmen", defrost[1:])


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
        "import",
        prices.import_,
        State(),
        [friday_noon, friday_noon + timedelta(days=1)],
    ) == (0.25, 0.21)


def test_the_rooms_habit_is_made_once_a_day():
    """Three weeks' median moves little in a day: today's is kept in the state
    and taken from there, yesterday's is made anew."""

    made: list[datetime] = []

    class Identifier:
        HABIT_HISTORY_DAYS = 21

        def dataset(self, config):
            return None

        def room_heat_habit(self, df, now):
            made.append(now)
            return pd.DataFrame(
                {"living": 1.0, "rest": 2.0}, index=range(0, 24 * 60, 15)
            )

    saved: list[ZoneHabit] = []
    optimization = Optimization(
        loader=SimpleNamespace(load=lambda *_: None),  # type: ignore[arg-type]
        state_manager=SimpleNamespace(  # type: ignore[arg-type]
            update_habit=lambda _, habit: saved.append(habit)
        ),
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=None,  # type: ignore[arg-type]
    )
    state = State()

    first = optimization._room_heat_habit(state, None, Identifier(), START)  # type: ignore[arg-type]
    state.predictions.zone_habit = saved[-1]
    again = optimization._room_heat_habit(  # type: ignore[arg-type]
        state, None, Identifier(), START + timedelta(hours=6)
    )
    optimization._room_heat_habit(  # type: ignore[arg-type]
        state, None, Identifier(), START + timedelta(days=1)
    )

    assert len(made) == 2
    assert first.loc[22 * 60, "rest"] == 2.0
    pd.testing.assert_frame_equal(first, again, check_dtype=False)


def test_the_hot_water_is_the_tops_highest_since_the_last_run():
    """Each quarter is labelled at its start: one that ended after the run's
    end counts, the quarter under way too; the run itself before it not."""

    start = datetime(2026, 10, 30, 13, 0, tzinfo=UTC)
    top = [
        SeriesPoint(time=start + timedelta(minutes=15 * i), value=value)
        for i, value in enumerate([52.0, 48.4, 48.5, 47.0, 42.5])
    ]

    assert Optimization.hot_water_c(
        top, start + timedelta(minutes=25), MPCConfig()
    ) == pytest.approx(48.5)
    assert (
        Optimization.hot_water_c(top, start + timedelta(hours=2), MPCConfig()) is None
    )


def _day_targets(now, windows):
    """Today's targets per quarter from local midnight: 10 degC (none, below
    the mains), and the (first quarter, last quarter, target) windows."""

    start = local_day_start(now).astimezone(UTC)
    values = [10.0] * 96

    for first, last, target in windows:
        values[first : last + 1] = [target] * (last + 1 - first)

    return [
        SeriesPoint(time=start + timedelta(minutes=15 * i), value=value)
        for i, value in enumerate(values)
    ]


def test_a_target_heated_for_today_is_lowered_by_what_was_drawn_since():
    """A run today to 48 degC and 3.7 K drawn since: today's 45 degC target
    asks 41.3, tomorrow's still 45; nor without a run today, or for a target
    the run did not reach."""

    now = datetime(2026, 10, 31, 16, 20, tzinfo=UTC)
    today = _day_targets(now, [(72, 73, 45.0)])
    times = [today[72].time, today[72].time + timedelta(days=1)]
    targets = (45.0, 45.0)
    run_end = datetime(2026, 10, 31, 12, 52, tzinfo=UTC)

    def served(run_end, hot_water_c):
        return Optimization.served_targets(
            targets, times, today, run_end, hot_water_c, 13.0, lambda _: 3.7, now
        )

    assert served(run_end, 48.0) == (pytest.approx(41.3), 45.0)
    assert served(None, 48.0) == targets
    assert served(run_end - timedelta(days=1), 48.0) == targets
    assert served(run_end, 44.0) == targets


def test_an_evening_target_is_not_lowered_by_draws_for_the_morning_one():
    """One run at 06:30 for a 40 degC target at 07:00 and a 45 degC one at
    18:00: 5 K drawn since the run, 3 K of it the morning's showers before
    07:30. The evening target is lowered by the 2 K drawn after the morning
    window only."""

    now = datetime(2026, 10, 31, 16, 20, tzinfo=UTC)
    today = _day_targets(now, [(28, 29, 40.0), (72, 73, 45.0)])
    morning_end = today[30].time

    assert Optimization.served_targets(
        (45.0,),
        [today[72].time],
        today,
        today[26].time,
        48.0,
        13.0,
        lambda since: 5.0 if since < morning_end else 2.0,
        now,
    ) == (pytest.approx(43.0),)


def _zone_bias(
    previous: SeriesPoint | None,
    on: bool,
    heat_w: float = 4500.0,
    surplus_w: float | None = None,
    setpoint_c: float = 20.0,
    run_setpoint_c: float | None = None,
    heating: bool = True,
    flow_lpm: float = 18.0,
) -> tuple[float, list[SeriesPoint]]:
    stored: list[SeriesPoint] = []
    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
        state_manager=SimpleNamespace(update_zone_bias=stored.append),  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=None,  # type: ignore[arg-type]
    )
    optimization._latest = lambda reference, now: 8.0  # type: ignore[method-assign]
    optimization._surplus_w = lambda config, now: surplus_w  # type: ignore[method-assign]
    config = SimpleNamespace(
        heat_pump=SimpleNamespace(
            outdoor_temperature=SensorReference(entity_id="outdoor")
        )
    )
    state = State()
    state.measurements.building.setpoint = [SeriesPoint(time=START, value=setpoint_c)]
    state.measurements.building.zone_temperature = [SeriesPoint(time=START, value=19.5)]
    state.measurements.heat_pump.flow = [SeriesPoint(time=START, value=flow_lpm)]
    state.schedule.building.bias_target = previous
    result = MPCResult(
        schedule=(0,) * 4,
        temperatures=TEMPERATURES,
        electrical_power_w=(0.0,) * 4,
        heat_w=(0.0,) * 4,
        objective_value=0.0,
        termination_condition="optimal",
        space_schedule=(1, 1, 1, 1),
        space_heat_w=(heat_w,) * 4,
        # COP 5, the plan expecting 50 W from the grid in each step.
        space_electrical_w=(heat_w / 5.0,) * 4,
        grid_step_kwh=(0.0125,) * 4,
    )
    data = MPCInput(
        solar_forecast_w=[0.0] * 4,
        ambient_temperature=20.0,
        current_temp_top=45.0,
        current_temp_bottom=45.0,
        boiler_on_current=False,
        zone_maximum_temperature=(21.0,) * 4,
        zone_floor_flow_lpm=18.0,
    )
    since = START - timedelta(hours=1) if heating else None
    bias = optimization._zone_bias(  # type: ignore[arg-type]
        config, data, result, state, 0, on, since, START, run_setpoint_c
    )

    return bias, stored


def test_the_zone_bias_holds_within_a_run_and_drops_after_it():
    """What is held is the controller's target, the setpoint (20 degC here)
    plus the bias."""

    free, _ = _zone_bias(None, on=True)
    assert free == 0.2

    # Changed 10 minutes ago in this run: held, though 0.4 K off.
    held, stored = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=10), value=19.8), on=True
    )
    assert (held, stored) == (-0.2, [])

    # Within the deadband: held however long ago.
    held, _ = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=40), value=20.1), on=True
    )
    assert held == 0.1

    # Past both: moves, and the change is stored.
    moved, stored = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=40), value=19.8), on=True
    )
    assert moved == 0.2 and stored == [SeriesPoint(time=START, value=20.2)]

    # The run over: back to 0 at once.
    off, _ = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=10), value=20.2), on=False
    )
    assert off == 0.0


def test_a_target_from_a_runs_start_up_is_not_held():
    """Set in the run's first quarter hour - here 55 minutes after its start,
    5 minutes ago - a target 0.4 K off moves; set past it, it holds."""

    since = START - timedelta(hours=1)
    moved, stored = _zone_bias(
        SeriesPoint(time=since + timedelta(minutes=5), value=19.8), on=True
    )
    assert moved == 0.2 and stored == [SeriesPoint(time=START, value=20.2)]

    held, _ = _zone_bias(
        SeriesPoint(time=since + timedelta(minutes=50), value=19.8), on=True
    )
    assert held == -0.2


def test_within_a_run_the_bias_target_may_pass_the_band():
    """A run under way gets the plan's heat whatever the band's edge (21 degC
    here); a run about to start is held to it."""

    under_way, _ = _zone_bias(None, on=True, heat_w=9000.0)
    starting, _ = _zone_bias(
        None, on=True, heat_w=9000.0, heating=False, run_setpoint_c=20.0
    )

    assert 20.0 + under_way > 21.0
    assert 20.0 + starting == pytest.approx(21.0)


def test_a_raised_setpoint_lowers_the_held_bias_as_much():
    # Held at 20.2 degC; the blueprint raises the thermostat to 20.5 within
    # the run: the target stays, the bias drops by the 0.5 K.
    held, stored = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=10), value=20.2),
        on=True,
        setpoint_c=20.5,
    )
    assert (held, stored) == (-0.3, [])


def test_the_zone_bias_takes_the_runs_setpoint_until_the_heat_pump_heats():
    # The thermostat still at its night 15 degC as the run starts: against
    # the 20 degC the blueprint gives it, not clamped at the bias limit.
    starting, _ = _zone_bias(
        None, on=True, setpoint_c=15.0, run_setpoint_c=20.0, heating=False
    )
    assert starting == 0.2

    # Heating: the thermostat's own setpoint, whoever set it.
    heating, _ = _zone_bias(None, on=True, setpoint_c=15.0, run_setpoint_c=20.0)
    assert heating == 2.5


def test_the_zone_bias_takes_the_floors_flow_until_the_run_has_settled():
    # Handed over, the flow measured is the hot water's (30 L/min here): the
    # floor's last run's 18 L/min instead, the same bias as heating at 18.
    handed_over, _ = _zone_bias(
        None, on=True, run_setpoint_c=20.0, heating=False, flow_lpm=30.0
    )
    assert handed_over == 0.2

    # A quarter hour into the run: the flow measured.
    heating, _ = _zone_bias(None, on=True, flow_lpm=30.0)
    assert heating < 0.2


def test_the_zone_bias_keeps_to_the_measured_surplus():
    # The plan's 4.5 kW (900 W at COP 5) with 850 W surplus measured and 50 W
    # import planned: within what the plan expected, so its bias.
    planned, _ = _zone_bias(None, on=True, surplus_w=850.0)
    assert planned == 0.2

    # 250 W less sun than that: lowered at once, though changed 10 minutes
    # ago and less than the deadband away.
    lowered, stored = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=10), value=20.2),
        on=True,
        surplus_w=600.0,
    )
    assert lowered < 0.2 and stored == [
        SeriesPoint(time=START, value=pytest.approx(20.0 + lowered))
    ]

    # The same surplus again: the same bias, not lowered further.
    again, stored = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=5), value=20.0 + lowered),
        on=True,
        surplus_w=600.0,
    )
    assert (again, stored) == (lowered, [])

    # The sun back: raised only after the hold.
    held, _ = _zone_bias(
        SeriesPoint(time=START - timedelta(minutes=10), value=20.0 + lowered),
        on=True,
        surplus_w=850.0,
    )
    assert held == lowered


def test_auto_adaptive_needs_the_units_outdoor_sensor():
    from domain.config import HeatPumpConfig

    sensor = {"entity_id": "sensor.x"}
    heat_pump = {
        name: sensor
        for name in (
            "state",
            "power",
            "supply_temperature",
            "return_temperature",
            "compressor_frequency",
            "flow",
        )
    } | {
        "boiler": {
            "setpoint": sensor,
            "top_temperature": sensor,
            "bottom_temperature": sensor,
            "ambient_temperature": sensor,
            "volume": 200,
            "target_temperature": 45.0,
        },
        "auto_adaptive": True,
    }

    with pytest.raises(ValueError, match="outdoor_temperature"):
        HeatPumpConfig.model_validate(heat_pump)

    assert HeatPumpConfig.model_validate(
        heat_pump | {"outdoor_temperature": sensor}
    ).auto_adaptive


def test_the_plans_day_energy_splits_the_heat_pump_and_prices_the_grid():
    """Two quarter hours: the tank at 2 kW in 1 kW surplus, then the floor at
    1 kW in 1.5 kW surplus - 0.25 kWh in, then 0.125 kWh out; and a night
    quarter hour, the house's 300 W alone, 0.075 kWh in."""

    result = MPCResult(
        schedule=(1, 0, 0),
        temperatures=(45.0, 50.0, 50.0, 50.0),
        electrical_power_w=(2000.0, 0.0, 0.0),
        heat_w=(6000.0, 0.0, 0.0),
        objective_value=0.0,
        termination_condition="optimal",
        space_schedule=(0, 1, 0),
        space_heat_w=(0.0, 4000.0, 0.0),
        space_electrical_w=(0.0, 1000.0, 0.0),
        electricity_step_kwh=(0.5, 0.25, 0.0),
    )
    data = MPCInput(
        solar_forecast_w=[1300.0, 1800.0, 0.0],
        baseload_forecast_w=(300.0, 300.0, 300.0),
        ambient_temperature=20.0,
        current_temp_top=45.0,
        current_temp_bottom=45.0,
        boiler_on_current=False,
        import_price_eur_per_kwh=(0.2, 0.2, 0.2),
        export_price_eur_per_kwh=(0.08, 0.08, 0.08),
    )
    optimizer = SimpleNamespace(
        config=MPCConfig(),
        prices=lambda data: (
            data.import_price_eur_per_kwh,
            data.export_price_eur_per_kwh,
        ),
    )

    frame = Optimization.planned_energy(result, data, TIMES[:3], optimizer)  # type: ignore[arg-type]

    assert list(frame["import"]) == pytest.approx([0.25, 0.0, 0.075])
    assert list(frame["export"]) == pytest.approx([0.0, 0.125, 0.0])
    assert list(frame["dhw"]) == pytest.approx([0.5, 0.0, 0.0])
    assert list(frame["heating"]) == pytest.approx([0.0, 0.25, 0.0])
    assert list(frame["house"]) == pytest.approx([0.075, 0.075, 0.075])
    assert frame["cost"].sum() == pytest.approx(0.325 * 0.2 - 0.125 * 0.08)
    # The balance: in - out = heat pump + house - sun.
    assert (frame["import"] - frame["export"]).sum() == pytest.approx(
        (frame[["dhw", "heating", "house"]].sum(axis=1) - frame["solar"]).sum()
    )


def test_the_measured_day_counts_import_and_export_per_minute():
    """A quarter hour's grid that nets to 0 over it: 500 W in for half of it
    and out for the other half still bought and sold 0.058 kWh each."""

    now = datetime(2026, 11, 6, 11, 15, tzinfo=UTC)
    start = datetime(2026, 11, 6, 11, 0, tzinfo=UTC)
    minutes = [start + timedelta(minutes=i) for i in range(15)]
    frame = pd.DataFrame(
        {
            "time": minutes,
            "grid": [500.0 if i < 7.5 else -500.0 for i in range(15)],
            "solar": [1000.0] * 15,
            "power": [800.0] * 15,
            "state": ["Verwarmen"] * 15,
        }
    )
    frame.loc[7, "grid"] = 0.0
    optimization = Optimization(
        loader=SimpleNamespace(load=lambda *args: frame),  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=None,  # type: ignore[arg-type]
    )
    optimization._price = lambda key, price, state, times: (
        (  # type: ignore[method-assign]
            (0.2,) if key == "import" else (0.08,)
        )
        * len(times)
    )
    config = SimpleNamespace(
        grid=SensorReference(entity_id="grid"),
        solar=SensorReference(entity_id="solar"),
        heat_pump=SimpleNamespace(
            power=SensorReference(entity_id="power"),
            state=SensorReference(entity_id="state"),
            states=HeatPumpStates(),
        ),
        prices=SimpleNamespace(import_=None, export=None),
    )

    quarters = optimization.measured_energy(config, State(), now)  # type: ignore[arg-type]

    assert len(quarters) == 1
    assert quarters["import"].iloc[0] == pytest.approx(7 * 500.0 / 60000.0)
    assert quarters["export"].iloc[0] == pytest.approx(7 * 500.0 / 60000.0)
    assert quarters["heating"].iloc[0] == pytest.approx(15 * 800.0 / 60000.0)
    assert quarters["cost"].iloc[0] == pytest.approx(7 * 500.0 / 60000.0 * (0.2 - 0.08))


def test_the_day_costs_are_one_table_with_today_measured_and_planned(caplog):
    """Today's measured and planned add up in its row, tomorrow is planned
    only; nothing is a dash."""

    today, tomorrow = START, START + timedelta(days=1)

    def frame(time, cost, heating):
        row = dict.fromkeys(Optimization.DAY_COLUMNS, 0.0)
        return pd.DataFrame([{**row, "time": time, "cost": cost, "heating": heating}])

    optimization = Optimization(
        loader=None,  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=None,  # type: ignore[arg-type]
    )
    optimization.measured_energy = lambda config, state, now: frame(today, 0.5, 1.0)  # type: ignore[method-assign]
    optimization.planned_energy = lambda *args: pd.concat(  # type: ignore[method-assign]
        [frame(today, 0.25, 0.0), frame(tomorrow, 1.0, 2.0)]
    )

    with caplog.at_level("INFO", logger="app.optimization"):
        optimization.log_day_costs(  # type: ignore[arg-type]
            None,
            None,
            None,
            SimpleNamespace(schedule=(), space_schedule=()),
            SimpleNamespace(config=MPCConfig()),
            [today, tomorrow],
            today,
        )

    lines = caplog.messages[-1].splitlines()
    assert lines[1].split()[:3] == ["day", "eur", "meas"]
    today_cells = lines[2].split()[3:]
    tomorrow_cells = lines[3].split()[3:]
    # EUR, meas, plan, in, out, solar, hp, dhw, heat
    assert today_cells[:3] == ["0.75", "0.50", "0.25"]
    assert today_cells[6] == "1.00" and today_cells[8] == "1.00"
    assert tomorrow_cells[:3] == ["1.00", "-", "1.00"]


def test_the_planned_runs_are_listed_by_their_start():
    """The tank's run and the floor's after it, each with its own steps'
    electricity."""

    result = SimpleNamespace(schedule=(1, 1, 0, 0), space_schedule=(0, 0, 1, 1))
    planned = pd.DataFrame(
        {"dhw": [0.5, 0.5, 0.0, 0.0], "heating": [0.0, 0.0, 0.25, 0.25]}
    )
    data = SimpleNamespace(zone_cooling=False)

    lines = Optimization.planned_runs(  # type: ignore[arg-type]
        result, data, planned, TIMES, timedelta(minutes=15)
    )

    assert [line.split()[-2] for line in lines] == ["run", "dhw", "heat"]
    assert lines[1].split()[-1] == "1.00" and lines[2].split()[-1] == "0.50"
    assert lines[2].split()[3:5] == [
        f"{to_local_time(TIMES[2]):%H:%M}",
        f"{to_local_time(TIMES[3] + timedelta(minutes=15)):%H:%M}",
    ]
