import dataclasses
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.state import StateManager
from domain.config import BoilerConfig, Config
from domain.jobs import OptimizeConfig
from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.physics import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    dew_point_from_humidity_c,
    indoor_dew_point_c,
)
from domain.state import SeriesPoint, State
from domain.time import local_day_start, to_local_time
from features.boiler import BoilerThermalIdentifier
from features.building import BuildingThermalIdentifier
from features.cop import HeatPumpCOPIdentifier
from features.dataset import DatasetBuilder, DatasetLoader
from features.dew_point import DewPointIdentifier
from features.floor import FloorCircuitIdentifier
from features.optimizer import SOLAR_SCENARIO_WEIGHTS, MPCOptimizer
from infrastructure.home_assistant import HomeAssistant
from infrastructure.repositories import ConfigRepository

logger = logging.getLogger(__name__)


class Optimization:
    # What a Home Assistant automation acts on: whether the quarter hour running
    # now is planned to heat the hot water, and when the next planned run starts
    # and with which SWW setpoint.
    DHW_STATUS_ENTITY = "binary_sensor.home_optimizer_dhw_status"
    DHW_START_ENTITY = "sensor.home_optimizer_dhw_start"
    DHW_SETPOINT_ENTITY = "sensor.home_optimizer_dhw_setpoint"
    # The floor's, published the same way while the plan drives it (see
    # publish_zone).
    ZONE_STATUS_ENTITY = "binary_sensor.home_optimizer_zone_status"
    ZONE_START_ENTITY = "sensor.home_optimizer_zone_start"
    ZONE_SETPOINT_ENTITY = "sensor.home_optimizer_zone_setpoint"

    def __init__(
        self,
        loader: DatasetLoader,
        state_manager: StateManager,
        config_repository: ConfigRepository,
        models_path: Path,
        home_assistant: HomeAssistant,
    ) -> None:
        self.loader = loader
        self.state_manager = state_manager
        self.config_repository = config_repository
        self.models_path = models_path
        self.home_assistant = home_assistant

    def run(self, optimize_config: OptimizeConfig) -> None:
        state = self.state_manager.load()
        config = self.config_repository.load()

        mpc_config = MPCConfig()

        # Fixed to optimize_config.steps so the MPC horizon is an explicit
        # choice, not whatever length the last solar prediction happened to
        # produce (which itself may have used a different PredictConfig.steps).
        # Falls back to however many steps are actually available (a shorter
        # horizon is still a valid, physically meaningful plan) rather than
        # inventing missing forecast data.
        steps = min(optimize_config.steps, len(state.predictions.solar))

        if steps < optimize_config.steps:
            logger.warning(
                "Solar prediction covers only %d of the requested %d optimize "
                "steps; planning over %d steps instead.",
                len(state.predictions.solar),
                optimize_config.steps,
                steps,
            )

        solar_forecast = [p.value for p in state.predictions.solar[:steps]]
        forecast_times = [p.time for p in state.predictions.solar[:steps]]

        # The expected-cost scenarios need the band at exactly the p50
        # prediction's own times; if it doesn't fully cover them (not
        # calibrated yet, no Solcast p10/p90), plan on p50 alone rather than
        # inventing the missing values.
        p10_by_time = {p.time: p.value for p in state.predictions.solar_p10}
        p90_by_time = {p.time: p.value for p in state.predictions.solar_p90}

        if all(t in p10_by_time and t in p90_by_time for t in forecast_times):
            solar_p10 = tuple(p10_by_time[t] for t in forecast_times)
            solar_p90 = tuple(p90_by_time[t] for t in forecast_times)
        else:
            solar_p10, solar_p90 = (), ()

        # The stored state.schedule.heat_pump.boiler.target_temperature is
        # resolved against *today's* timestamps (see StateManager.update()) - not
        # the future forecast horizon the MPC actually needs. Resolve the raw
        # config schedule against the forecast's own timestamps instead.
        target_temps = tuple(
            point.value
            for point in self.state_manager.resolve_schedule(
                config.heat_pump.boiler.target_temperature, forecast_times
            )
        )

        dynamics_forecaster = BoilerThermalIdentifier()
        dynamics_forecaster.load(path=self.models_path)
        thermal_model = dynamics_forecaster.get_model()

        # None if cop_dhw hasn't been calibrated yet (load() warns and leaves
        # it unset rather than raising) - MPCOptimizer falls back to the flat
        # boiler_electrical_power_w assumption in that case.
        cop_identifier = HeatPumpCOPIdentifier(key="dhw")
        cop_identifier.load(path=self.models_path)
        cop_model = cop_identifier.model

        dhw_state = config.heat_pump.states.dhw
        now = datetime.now(timezone.utc)
        heat_pump_changes = self._heat_pump_changes(config, now)
        boiler_on_current, run_start, idle_elapsed_hours = self.dhw_timing(
            *heat_pump_changes, dhw_state, now
        )
        compressor_elapsed_hours = (
            (now - run_start).total_seconds() / 3600.0 if run_start else 0.0
        )
        # The plan starts now, partway into the quarter its first step belongs
        # to (see MPCInput.first_step_hours).
        first_step_hours = (
            forecast_times[0] + timedelta(hours=mpc_config.step_hours) - now
        ).total_seconds() / 3600.0

        if not 0.0 < first_step_hours < mpc_config.step_hours:
            first_step_hours = None
        run_start_temp_top = run_start_temp_bottom = None
        run_start_stratification_k = None
        stratification = {
            p.time: p.value for p in state.measurements.heat_pump.boiler.stratification
        }

        if run_start is not None:
            # The tank as the run found it: the last sensor reading of a
            # measurement interval (one planning step) that ended before it
            # began - none if the run began before the day's readings do.
            reading_interval = timedelta(hours=mpc_config.step_hours)
            boiler = state.measurements.heat_pump.boiler
            before_top = [
                p
                for p in boiler.top_temperature
                if p.time + reading_interval <= run_start
            ]
            before_bottom = [
                p
                for p in boiler.bottom_temperature
                if p.time + reading_interval <= run_start
            ]

            if before_top and before_bottom:
                run_start_temp_top = before_top[-1].value
                run_start_temp_bottom = before_bottom[-1].value
                run_start_stratification_k = stratification.get(before_top[-1].time)

        # Aligned against solar's own forecast timestamps, not assumed to share
        # them: the tap forecaster is fit/predicted independently (see
        # features/tap.py) and may not have been run at all, or over a
        # different horizon - align_predictions falls back to 0.0 (no draws
        # assumed) wherever no matching point exists, the same assumption
        # implicitly made before this forecast existed.
        tap_forecast = tuple(
            self.state_manager.align_predictions(state.predictions.tap, forecast_times)
        )

        # state.forecast.open_meteo.temperature is the raw Open-Meteo forecast
        # (see StateManager._map()), not a model prediction, so it is only
        # ever missing outright (fresh install, forecast fetch not yet run) -
        # left empty in that case rather than defaulting every step to 0.0
        # deg C, which align_predictions' usual "assume none" fallback would
        # do here (a plausible default for "no tap draws", not for "outdoor
        # temperature"). MPCOptimizer falls back to the flat
        # boiler_electrical_power_w assumption when this is empty.
        outdoor_temperature_forecast = (
            tuple(
                self.state_manager.align_predictions(
                    state.forecast.open_meteo.temperature, forecast_times
                )
            )
            if state.forecast.open_meteo.temperature
            else ()
        )

        data = MPCInput(
            solar_forecast_w=solar_forecast,
            ambient_temperature=state.measurements.heat_pump.boiler.ambient_temperature[
                -1
            ].value,
            current_temp_top=state.measurements.heat_pump.boiler.top_temperature[
                -1
            ].value,
            current_temp_bottom=state.measurements.heat_pump.boiler.bottom_temperature[
                -1
            ].value,
            boiler_on_current=boiler_on_current,
            current_stratification_k=stratification.get(
                state.measurements.heat_pump.boiler.top_temperature[-1].time
            ),
            run_start_stratification_k=run_start_stratification_k,
            target_temperature_top=target_temps,
            tap_forecast_w=tap_forecast,
            outdoor_temperature_forecast=outdoor_temperature_forecast,
            solar_p10_w=solar_p10,
            solar_p90_w=solar_p90,
            compressor_elapsed_hours=compressor_elapsed_hours,
            run_start_temp_top=run_start_temp_top,
            run_start_temp_bottom=run_start_temp_bottom,
            first_step_hours=first_step_hours,
            idle_elapsed_hours=idle_elapsed_hours,
            baseload_forecast_w=tuple(
                self.state_manager.baseload_forecast(
                    state, forecast_times, datetime.now(timezone.utc)
                )
            ),
        )
        data = self._with_legionella(
            data, forecast_times, config.heat_pump.boiler, thermal_model, mpc_config
        )

        # One plan for the one heat pump: the tank and, where it can be
        # planned, the zone, sharing the compressor. Only the hot water part is
        # acted on (see publish_dhw); the zone is not ours to drive yet.
        dhw_only = MPCOptimizer(
            thermal_model=thermal_model, config=mpc_config, cop_model=cop_model
        )
        optimizer, planned = dhw_only, data
        zone = self._zone(
            state, config, forecast_times, mpc_config, optimize_config.cooling
        )

        if zone is not None:
            inputs, models = zone
            optimizer = MPCOptimizer(
                thermal_model=thermal_model,
                config=mpc_config,
                cop_model=cop_model,
                **models,
            )
            planned = dataclasses.replace(data, **inputs)

        try:
            result = optimizer.solve(planned)
        except RuntimeError as error:
            if zone is None:
                raise

            # The zone is not acted on, so it must not take the hot water plan
            # down with it: without it, that is the plan as it always was.
            logger.warning("Plan with the zone failed, hot water alone: %s", error)
            zone, optimizer, planned = None, dhw_only, data
            result = optimizer.solve(planned)

        logger.info(
            "Optimization completed: schedule=%s objective=%.3f",
            result.schedule,
            result.objective_value,
        )

        self.state_manager.update_schedule(
            schedule=result.schedule,
            temperatures=result.temperatures,
            power_w=result.electrical_power_w,
            times=forecast_times,
            heat_w=result.heat_w,
        )

        self.publish_dhw(result, forecast_times, thermal_model)
        self.log_dhw_plan(result, forecast_times, mpc_config)

        if optimize_config.explain:
            self.explain_dhw_plan(optimizer, planned, result, forecast_times)

        if zone is not None:
            # While cooling, the supply of the first planned run: the setpoint
            # the plan would give the heat pump.
            supply_c = next(
                (c for c in result.space_supply_c if not math.isnan(c)), None
            )
            logger.info(
                "Zone plan: %.1f kWh into the zone over the horizon, first run "
                "at %s degC supply",
                sum(result.space_heat_w) * mpc_config.step_hours / 1000.0,
                "unknown" if supply_c is None else f"{supply_c:.1f}",
            )

        # Empty without a zone plan, so the dashboard shows none rather than
        # an old one.
        self.state_manager.update_building_schedule(
            heat_w=result.space_heat_w,
            temperatures=result.zone_temperatures,
            times=forecast_times,
            supply_c=result.space_supply_c,
            power_w=result.space_electrical_w,
        )
        self.publish_zone(
            result,
            forecast_times,
            state,
            config,
            planned if zone is not None else None,
            heat_pump_changes[0],
            now,
        )

    def _heat_pump_changes(
        self, config: Config, now: datetime
    ) -> tuple[list[SeriesPoint], list[SeriesPoint]]:
        """The operating state's and compressor frequency's own changes since
        the previous local midnight, for dhw_timing - so a run or pause that
        spans midnight is seen.

        Their changes, not the state's quarter-hour readings: when a DHW run
        began sets how much heat it has put in so far (see
        MPCOptimizer._initial_temperature), and a quarter's reading placed that
        at the quarter's start. Replayed over 41 real runs, the tank a run had
        heated came out 2.3 K off on average from the quarter's start against
        1.6 K from the exact one, and a run begun late in its quarter 4.1 K too
        warm - up to 16 minutes of run misjudged. Both only report on change, so
        this is a few dozen points a day; the dashboard keeps its quarter hours.
        """

        frame = self.loader.load(
            DatasetBuilder()
            .timeseries(
                "state", config.heat_pump.state, aggregation=None, fill="previous"
            )
            .timeseries(
                "frequency",
                config.heat_pump.compressor_frequency,
                aggregation=None,
                fill="previous",
            )
            .build(),
            local_day_start(now, days=-1).astimezone(timezone.utc),
            now,
        )

        def changes(column: str) -> list[SeriesPoint]:
            if column not in frame:
                return []

            points = frame[["time", column]].dropna()

            return [
                SeriesPoint(time=time, value=value)
                for time, value in points.itertuples(index=False)
            ]

        return changes("state"), changes("frequency")

    @staticmethod
    def dhw_timing(
        heat_pump_state: list[SeriesPoint],
        compressor_frequency: list[SeriesPoint],
        dhw_state: str,
        now: datetime,
    ) -> tuple[bool, datetime | None, float]:
        """(whether a DHW run is under way, when its compressor run began, None
        without one, hours since the last DHW run ended - 0 while one runs),
        from the operating state's and compressor frequency's changes: each
        value holds from its own time until the next.

        The compressor run, not the operating state alone: the state stays on
        DHW while the resistive booster finishes the tank, and the booster runs
        with the compressor off at 0 Hz (see BoilerThermalIdentifier.booster_
        active), so counting that time would credit the minimum runtime with
        time the compressor did not run. A run's first minutes at 0 Hz, the
        pump circulating before the compressor comes up, are the run's own
        start and count. Without a frequency the state is all there is, and
        assuming the compressor ran keeps a real run protected. The pause is
        taken from the change that ended the last DHW run - from the first
        change loaded (the previous local midnight's, see _heat_pump_changes)
        if none shows, which errs short and keeps it.
        """

        if not heat_pump_state:
            return False, None, 0.0

        def frequency_at(time: datetime) -> float:
            value = 1.0

            for point in compressor_frequency:
                if point.time > time:
                    break
                value = point.value

            return value

        if heat_pump_state[-1].value == dhw_state:
            # The change into DHW that began the stretch under way.
            dhw_start = heat_pump_state[-1].time

            for point in reversed(heat_pump_state):
                if point.value != dhw_state:
                    break
                dhw_start = point.time

            changes = sorted(
                {dhw_start}
                | {p.time for p in compressor_frequency if dhw_start < p.time <= now}
            )
            run_start = dhw_start
            compressor_ran = False

            for time in changes:
                running = frequency_at(time) > 0.0

                if running:
                    compressor_ran = True
                elif compressor_ran:
                    # The booster has taken over; the compressor may start again.
                    run_start = None
                    continue

                if run_start is None:
                    run_start = time

            return True, run_start, 0.0

        idle_start = heat_pump_state[-1].time

        for point in reversed(heat_pump_state):
            if point.value == dhw_state:
                break
            idle_start = point.time

        return False, None, max(0.0, (now - idle_start).total_seconds() / 3600.0)

    @staticmethod
    def dhw_runs(schedule: tuple[int, ...]) -> list[tuple[int, int]]:
        """(first, last) step of each planned run."""

        runs: list[list[int]] = []

        for k, on in enumerate(schedule):
            if on and runs and runs[-1][1] == k - 1:
                runs[-1][1] = k
            elif on:
                runs.append([k, k])

        return [(first, last) for first, last in runs]

    @classmethod
    def dhw_summary(
        cls,
        result: MPCResult,
        times: list[datetime],
        mpc_config: MPCConfig,
    ) -> str:
        """What a hot water plan does and what it expects that to cost, per
        local day: each run's time and end temperature, the day's electricity
        and how much of it is expected from the grid, as the plan's own
        objective counts them (see MPCResult.electricity_kwh) - so the cheaper
        of two plans here is the one the optimizer would choose - and the total.
        The cost is the import at the price and the own sun at the export it
        forgoes; not the whole objective, which also prices starts and
        shortfalls and so is no amount of money."""

        runs = cls.dhw_runs(result.schedule)

        if not runs:
            return "no run within the horizon"

        def cost_eur(electricity_kwh: float, grid_kwh: float) -> float:
            return (
                grid_kwh * mpc_config.price_eur_per_kwh
                + (electricity_kwh - grid_kwh) * mpc_config.feed_in_price_eur_per_kwh
            )

        step = timedelta(hours=mpc_config.step_hours)
        last = len(result.temperatures) - 1
        day = [to_local_time(time).date() for time in times]
        days = []

        for date in dict.fromkeys(day[first] for first, _ in runs):
            runs_text = ", ".join(
                f"{to_local_time(times[first]):%H:%M}-"
                f"{to_local_time(times[end] + step):%H:%M} "
                f"to {result.temperatures[min(end + 1, last)]:.1f} degC"
                for first, end in runs
                if day[first] == date
            )
            electricity_kwh = sum(
                kwh
                for kwh, d in zip(result.electricity_step_kwh, day, strict=False)
                if d == date
            )
            grid_kwh = sum(
                kwh
                for kwh, d in zip(result.grid_step_kwh, day, strict=False)
                if d == date
            )
            days.append(
                f"{date:%a} {runs_text}: {electricity_kwh:.2f} kWh, of which sun "
                f"{electricity_kwh - grid_kwh:.2f} and grid {grid_kwh:.2f} "
                f"(expected), EUR {cost_eur(electricity_kwh, grid_kwh):.3f}"
            )

        return " | ".join(days) + (
            f" | total {result.electricity_kwh:.2f} kWh, "
            f"EUR {cost_eur(result.electricity_kwh, result.grid_kwh):.3f}"
        )

    @classmethod
    def log_dhw_plan(
        cls,
        result: MPCResult,
        times: list[datetime],
        mpc_config: MPCConfig,
    ) -> None:
        logger.info("DHW plan: %s", cls.dhw_summary(result, times, mpc_config))

    # How much earlier explain_dhw_plan has the first run finish (hours): the
    # question a plan usually raises is why it does not run earlier, in more sun.
    # Half an hour first: a run lasts about an hour, and the sunnier slot a plan
    # passes over is often just before it; then on to where the morning starts.
    EXPLAIN_EARLIER_HOURS = (0.5, 1, 2, 3)

    @classmethod
    def explain_dhw_plan(
        cls,
        optimizer: MPCOptimizer,
        data: MPCInput,
        result: MPCResult,
        times: list[datetime],
    ) -> None:
        """Logs the plan beside the same plan with its first run finished
        earlier, at the temperature the plan ends it at: solved again with that
        temperature as a target, so each alternative is costed with its own
        losses, COP and end temperature, not the plan's run moved in time. One
        more solve per alternative, so only on request (OptimizeConfig.explain).
        """

        runs = cls.dhw_runs(result.schedule)

        if not runs:
            return

        mpc_config = optimizer.config
        finish = runs[0][1] + 1
        end_c = result.temperatures[min(finish, len(result.temperatures) - 1)]
        lines = [f"  {'plan:':<16}{cls.dhw_summary(result, times, mpc_config)}"]

        run_steps = runs[0][1] - runs[0][0] + 1
        # The target the plan's end answers: a run planned to a target ends the
        # heat pump's overshoot above it (see MPCOptimizer), so forcing end_c
        # itself had each alternative end that overshoot higher again (real
        # data: 49.5 against the plan's 47.7 degC) and look dearer than it is.
        forced_c = end_c - (optimizer.thermal_model.setpoint_overshoot_k or 0.0)

        for hours in cls.EXPLAIN_EARLIER_HOURS:
            k = finish - round(hours / mpc_config.step_hours)

            # The run would have to start before now: no plan can reach it, and
            # the solver would only return the nearest one it can - an earlier
            # line again.
            if k - run_steps < 0:
                continue

            targets = list(data.target_temperature_top)
            targets[k] = max(targets[k], forced_c)
            forced = dataclasses.replace(data, target_temperature_top=tuple(targets))
            summary = cls.dhw_summary(optimizer.solve(forced), times, mpc_config)
            lines.append(f"  {f'{hours:g} h earlier:':<16}{summary}")

        if len(lines) == 1:
            lines.append("  no earlier finish left: the run would start before now")

        logger.info("DHW plan alternatives:\n%s", "\n".join(lines))

    def _with_legionella(
        self,
        data: MPCInput,
        times: list[datetime],
        boiler: BoilerConfig,
        thermal_model: BoilerThermalModel,
        mpc_config: MPCConfig,
    ) -> MPCInput:
        """The plan's input with a disinfection target, once one is due.

        The last day both tank sensors reached the disinfection temperature
        starts the interval: the whole tank has to be at temperature, not just
        the water around one sensor. One query per sensor for its daily maximum,
        so the period costs a row per day rather than every reading in it. The
        target goes in once the interval ends within the plan's horizon, so the
        choice is between today and tomorrow - or, once it has ended, within the
        fine-resolution horizon from now; where, disinfection_step decides. A
        day early costs the next interval a day, but over 40 real days choosing
        between the two put 24% more of the run on the sun (59.6 against 47.9
        kWh) than choosing within its last day alone: better on 16 of the 20
        days the choice differed, worse on 3.
        """

        legionella = boiler.legionella

        if legionella is None:
            return data

        now = datetime.now(timezone.utc)
        daily_max = self.loader.load(
            DatasetBuilder()
            .timeseries("top", boiler.top_temperature, interval="1d", aggregation="max")
            .timeseries(
                "bottom", boiler.bottom_temperature, interval="1d", aggregation="max"
            )
            .build(),
            now - timedelta(days=legionella.interval),
            now,
        )
        disinfected = daily_max["time"][
            daily_max[["top", "bottom"]].min(axis=1) >= legionella.temperature
        ]
        window = timedelta(hours=mpc_config.fine_horizon_hours)
        # To the end of the local day the interval ends on: the daily maxima are
        # UTC days, and a deadline at their midnight fell in the night, leaving
        # only night hours - grid power - to choose from.
        deadline = (
            local_day_start(
                disinfected.max() + timedelta(days=legionella.interval), days=1
            )
            if len(disinfected)
            else now
        )

        if deadline <= now:
            deadline = now + window
        elif deadline > times[-1] + timedelta(hours=mpc_config.step_hours):
            # The last step runs until its own end, so a deadline at midnight is
            # within a forecast whose last step starts at 23:45.
            return data

        k = self.disinfection_step(
            data, times, deadline, thermal_model, legionella.temperature, mpc_config
        )
        targets = list(data.target_temperature_top)
        targets[k] = max(targets[k], legionella.temperature)

        return dataclasses.replace(data, target_temperature_top=tuple(targets))

    @staticmethod
    def disinfection_step(
        data: MPCInput,
        times: list[datetime],
        deadline: datetime,
        thermal_model: BoilerThermalModel,
        temperature: float,
        mpc_config: MPCConfig,
    ) -> int:
        """The step before `deadline` the tank should be at `temperature` by:
        the one whose run leading up to it the expected solar surplus over the
        baseload covers most (over the calibrated scenarios, see
        SOLAR_SCENARIO_WEIGHTS). The run lasts as long as the heat pump takes to
        lift the tank to its own limit and the booster from there. Surplus beyond
        what the run draws is counted too, though it would be exported anyway: a
        simplification that only matters between two stretches that are both
        sunnier than the run needs. Not the step running now, which no plan can
        still heat for; the latest of equals, so the tank is hot no longer than
        it has to be.
        """

        heat_capacity_j_per_k = (
            thermal_model.volume_l * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K
        )
        start_c = thermal_model.mixed_temperature(
            data.current_temp_top,
            data.current_temp_bottom,
            data.current_stratification_k,
        )
        limit_c = thermal_model.heat_pump_max_tank_temperature_c or temperature
        seconds = (
            heat_capacity_j_per_k
            * max(min(temperature, limit_c) - start_c, 0.0)
            / (thermal_model.q_in_steady_w or thermal_model.q_in_nominal_w)
        )

        if thermal_model.booster_heat_w and temperature > limit_c:
            seconds += (
                heat_capacity_j_per_k
                * (temperature - max(limit_c, start_c))
                / thermal_model.booster_heat_w
            )

        run_steps = max(1, math.ceil(seconds / (mpc_config.step_hours * 3600.0)))
        scenarios, weights = (
            (
                (data.solar_p10_w, data.solar_forecast_w, data.solar_p90_w),
                SOLAR_SCENARIO_WEIGHTS,
            )
            if data.solar_p10_w
            else ((data.solar_forecast_w,), (1.0,))
        )
        baseload = np.asarray(data.baseload_forecast_w or [0.0] * len(times))
        surplus = sum(
            weight * np.clip(np.asarray(solar) - baseload, 0.0, None)
            for weight, solar in zip(weights, scenarios, strict=True)
        )
        covered = np.concatenate(([0.0], np.cumsum(surplus)))
        candidates = [k for k in range(1, len(times)) if times[k] <= deadline] or [1]

        return max(
            candidates,
            key=lambda k: (covered[k] - covered[max(k - run_steps, 0)], k),
        )

    def _zone(
        self,
        state: State,
        config: Config,
        times: list[datetime],
        mpc_config: MPCConfig,
        cooling: bool | None = None,
    ) -> tuple[dict, dict] | None:
        """The zone as a second demand on the compressor: what the plan needs
        of it (MPCInput fields) and the models it plans it with
        (MPCOptimizer arguments), or None where it cannot be planned.

        The zone is estimated here, where it is planned, rather than on every
        state update: one load and one filter pass give both the state a plan
        starts from and the measured temperature and thermal mass the dashboard
        draws. Two-node, because only that structure has the thermal mass a
        floor buffer is made of. Only planned with a comfort ceiling
        configured.

        Acted on only where configured (BuildingConfig.control_heating and
        control_cooling, see publish_zone). Elsewhere the heat pump runs the
        floor by itself, so a floor run under way is not the plan's to hold:
        it is left out (space_on_current), rather than keep the tank waiting on
        a run the plan cannot stop. The Ecodan hands over to hot water when
        asked; were it not to, hot water would wait for the floor exactly as it
        did before the zone was planned at all. Where the plan drives the
        floor, the run under way is its own and counts.

        Cooled rather than heated while the heat pump is set to cool (see
        HeatPumpConfig.mode), from the cooling runs' own models, and only with
        dew points configured: the floor may not be cooled below them.
        """

        identifier = BuildingThermalIdentifier(
            self.state_manager.latitude, self.state_manager.longitude
        )
        identifier.load(self.models_path)

        if identifier.model is None:
            return None

        now = datetime.now(timezone.utc)
        # From the previous local midnight, so the dashboard has yesterday
        # beside today, and before that the filter's own warm-up.
        start = local_day_start(now, days=-1).astimezone(timezone.utc) - timedelta(
            hours=identifier.MASS_WARMUP_HOURS
        )
        end = times[-1] + timedelta(hours=mpc_config.step_hours)
        baseload = pd.Series(
            {point.time: point.value for point in state.predictions.baseload}
        )

        try:
            zone = identifier.trajectory(
                self.loader.load(identifier.dataset(config), start, end),
                now,
                baseload if not baseload.empty else None,
            )
        except ValueError as error:
            # Normal right after a restart with no measurements yet, or before
            # the weather forecast reaches past now.
            logger.warning("No zone estimate: %s", error)
            return None

        zone.index = pd.to_datetime(zone.index, utc=True)
        self.state_manager.update_zone(
            temperature=zone["measured"].dropna(), thermal_mass=zone["mass"]
        )

        maximum = config.building.maximum_temperature

        if maximum is None:
            return None

        # The plan starts where the filter's estimate of the whole zone state
        # stands at its first step: the air node, and the mass.
        planned = zone.reindex(pd.DatetimeIndex(times))

        if planned[["air", "mass"]].iloc[0].isna().any():
            logger.info("Zone not planned: no zone estimate at %s", times[0])
            return None

        states = config.heat_pump.states
        mode = state.measurements.heat_pump.mode
        # The heat pump's own mode unless this run asks for one (see
        # OptimizeConfig.cooling).
        if cooling is None:
            cooling = bool(mode) and str(mode[-1].value).startswith(states.cooling)
        dew_point_c = self._dew_point_forecast(state, times, mpc_config)

        if cooling and dew_point_c is None:
            logger.info(
                "Zone not planned: no dew point to keep the floor above while "
                "cooling (see building.dew_points)"
            )
            return None

        inputs = dict(
            zone_temperature=float(planned["air"].iloc[0]),
            zone_mass_temperature=float(planned["mass"].iloc[0]),
            zone_target_temperature=tuple(
                point.value
                for point in self.state_manager.resolve_schedule(
                    config.building.target_temperature, times
                )
            ),
            zone_maximum_temperature=tuple(
                point.value
                for point in self.state_manager.resolve_schedule(maximum, times)
            ),
            zone_comfort_tolerance_c=config.building.comfort_tolerance,
            # No gain assumed where the estimate does not reach - the same
            # "assume none" align_predictions makes for any missing forecast.
            zone_internal_gain_w=tuple(planned["internal_gain_w"].fillna(0.0)),
            zone_solar_gain_w=tuple(planned["solar_gain_w"].fillna(0.0)),
            zone_cooling=cooling,
            zone_mass_minimum_c=(
                tuple(c + config.building.dew_point_margin for c in dew_point_c)
                if cooling
                else ()
            ),
            zone_supply_minimum_c=(
                tuple(dew_point_c)
                if cooling and not config.building.insulated_pipes
                else ()
            ),
        )

        heat_pump_state = state.measurements.heat_pump.state
        mode_state = states.cooling if cooling else states.heating

        if self.controlled(config, cooling) and heat_pump_state:
            inputs["space_on_current"] = heat_pump_state[-1].value == mode_state

        # The measurements start at local midnight, so any hot water among
        # them was today's.
        if cooling and config.building.dhw_after_cooling:
            inputs["zone_local_day"] = tuple(
                to_local_time(t).toordinal() for t in times
            )
            inputs["dhw_earlier_today"] = any(
                point.value == states.dhw for point in heat_pump_state
            )

        # How the heat pump runs the floor by itself in this mode - None until
        # its runs have shown it, and the plan may then choose the zone's heat
        # freely.
        key = "cooling" if cooling else "heating"
        floor_circuit = FloorCircuitIdentifier(
            self.state_manager.latitude,
            self.state_manager.longitude,
            self.models_path,
            key=key,
        )
        floor_circuit.load(self.models_path)
        space_cop = HeatPumpCOPIdentifier(key=key)
        space_cop.load(path=self.models_path)

        return inputs, dict(
            building_model=identifier.model,
            floor_circuit_model=floor_circuit.model,
            space_cop_model=space_cop.model,
        )

    def _dew_point_forecast(
        self, state: State, times: list[datetime], mpc_config: MPCConfig
    ) -> list[float] | None:
        """The indoor dew point at each step (deg C), None without one measured.

        From the one measured now, carried along the outdoor forecast by the
        identified moisture balance (see features.dew_point) - Open-Meteo's dew
        point, or its temperature and humidity where it has none. Held at the
        measured one without that model or forecast, as before there was one.
        Stored as a prediction, so the dashboard draws what the plan used.
        """

        measured = state.measurements.building.dew_point

        if not measured:
            return None

        now_c = float(measured[-1].value)
        identifier = DewPointIdentifier()
        identifier.load(self.models_path)
        weather = state.forecast.open_meteo

        def aligned(points: list[SeriesPoint]) -> pd.Series:
            return pd.Series(
                self.state_manager.align_predictions(points, times, default=np.nan)
            )

        outdoor_c = aligned(weather.dew_point).fillna(
            dew_point_from_humidity_c(
                aligned(weather.temperature), aligned(weather.relative_humidity)
            )
        )

        if identifier.model is None or outdoor_c.isna().all():
            forecast_c = [now_c] * len(times)
        else:
            # A gap in the forecast holds its neighbour's value.
            forecast_c = indoor_dew_point_c(
                identifier.model,
                now_c,
                outdoor_c.ffill().bfill().to_numpy(),
                mpc_config.step_hours,
            ).tolist()

        self.state_manager.update_prediction(
            "dew_point", pd.Series(forecast_c, index=times)
        )

        return forecast_c

    @staticmethod
    def controlled(config: Config, cooling: bool) -> bool:
        """Whether the plan drives the floor in this mode."""

        building = config.building

        return building.control_cooling if cooling else building.control_heating

    @staticmethod
    def running_since(changes: list[SeriesPoint], mode_state: str) -> datetime | None:
        """When the heat pump entered the mode it is in now (from its state's
        own changes), None when it is not in it."""

        if not changes or changes[-1].value != mode_state:
            return None

        since = changes[-1].time

        for point in reversed(changes):
            if point.value != mode_state:
                break
            since = point.time

        return since

    @staticmethod
    def zone_setpoint_c(
        planned_c: float,
        cooling_w: float,
        return_c: float | None,
        flow_lpm: float | None,
        minimum_c: float | None,
        previous_c: float | None,
    ) -> float:
        """The supply setpoint for a cooling run under way (deg C), to the heat
        pump's half degree.

        From the water the floor sends back, return - Q / (m_dot c_p): the
        supply that takes the planned cooling from it, set on what the floor
        does rather than on the plan's estimate of a mass no sensor measures
        (0.5-1 K off, enough to put the heat pump under its least cooling or
        the supply under the dew point). The plan's own supply without a
        return to go on - a run's first quarter hour, while the loop still
        holds the still water it stood with. Never above an earlier setpoint of
        the same run: raised over the water in the loop, the heat pump turns
        down past its least and stops. Never below the supply minimum (the dew
        point uninsulated pipes carry), which comes first - a run whose dew
        point rose past it is for the plan to stop.
        """

        setpoint_c = planned_c

        if return_c is not None and flow_lpm:
            water_w_per_k = flow_lpm / 60.0 * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K
            setpoint_c = return_c - cooling_w / water_w_per_k

        if previous_c is not None:
            setpoint_c = min(setpoint_c, previous_c)

        # The nearest half degree, as for the hot water setpoint - but never
        # rounded under the minimum.
        rounded_c = math.floor(round(2 * setpoint_c, 2) + 0.5) / 2

        if minimum_c is not None:
            rounded_c = max(rounded_c, math.ceil(round(2 * minimum_c, 2)) / 2)

        return rounded_c

    def publish_zone(
        self,
        result: MPCResult,
        times: list[datetime],
        state: State,
        config: Config,
        data: MPCInput | None,
        heat_pump_state: list[SeriesPoint],
        now: datetime,
    ) -> None:
        """Writes the plan's floor decision to Home Assistant, as publish_dhw
        does the hot water's: on/off for the quarter hour running now, the
        start of the run under way or the next one planned, and - cooling - the
        supply setpoint while a run is under way (see zone_setpoint_c).

        All three 'unknown' where the plan does not drive the floor in the mode
        the heat pump is in (BuildingConfig.control_heating/control_cooling) or
        cannot plan the zone at all: nothing then for an automation to act on.
        The setpoint 'unknown' outside a run too, and always while heating,
        where the heat pump takes its supply from its own curve.
        """

        status = start = setpoint = "unknown"
        published = None

        if (
            data is not None
            and result.space_schedule
            and self.controlled(config, data.zone_cooling)
        ):
            schedule = result.space_schedule
            running = schedule[0] == 1
            states = config.heat_pump.states
            since = self.running_since(
                heat_pump_state,
                states.cooling if data.zone_cooling else states.heating,
            )
            first = next((k for k, on in enumerate(schedule) if on), None)
            status = "on" if running else "off"

            if first is not None:
                start = (
                    since if running and since is not None else times[first]
                ).isoformat()

            if data.zone_cooling and running and result.space_supply_c:
                measured = state.measurements.heat_pump
                step = timedelta(hours=MPCConfig().step_hours)
                settled = since is not None and now - since >= step
                previous = state.schedule.building.supply_setpoint
                published = self.zone_setpoint_c(
                    planned_c=result.space_supply_c[0],
                    cooling_w=-result.space_heat_w[0],
                    return_c=(
                        measured.return_temperature[-1].value
                        if settled and measured.return_temperature
                        else None
                    ),
                    flow_lpm=(
                        measured.flow[-1].value if settled and measured.flow else None
                    ),
                    minimum_c=(
                        data.zone_supply_minimum_c[0]
                        if data.zone_supply_minimum_c
                        else None
                    ),
                    previous_c=(
                        previous.value
                        if previous is not None
                        and since is not None
                        and previous.time >= since
                        else None
                    ),
                )
                setpoint = str(published)

        self.state_manager.update_supply_setpoint(
            SeriesPoint(time=now, value=published) if published is not None else None
        )

        self.home_assistant.set_state(
            self.ZONE_STATUS_ENTITY,
            status,
            {"friendly_name": "Home Optimizer zone status"},
        )
        self.home_assistant.set_state(
            self.ZONE_START_ENTITY,
            start,
            {"friendly_name": "Home Optimizer zone start", "device_class": "timestamp"},
        )
        self.home_assistant.set_state(
            self.ZONE_SETPOINT_ENTITY,
            setpoint,
            {
                "friendly_name": "Home Optimizer zone setpoint",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
            },
        )

    def publish_dhw(
        self,
        result: MPCResult,
        times: list[datetime],
        thermal_model: BoilerThermalModel | None = None,
    ) -> None:
        """Writes the plan's hot water decision to Home Assistant: on/off for the
        quarter hour running now, and the start and SWW setpoint of the next
        planned run (the current one if it is heating now, 'unknown' without
        any).

        The heat pump heats until its setpoint and stops by itself; the heat
        left around the coil and in the loop then still mixes through, and the
        tank settles setpoint_overshoot_k above the setpoint (real data: median
        1.75 K). The plan counts that heat - a run it plans ends that much above
        what its targets need (see MPCOptimizer) - so the setpoint is the run's
        planned end less it: the setpoint its targets need, and the tank ends
        where the plan has it. That margin above the need stays on purpose: it
        absorbs a tap or loss forecast that turns out worse, where planning the
        run to end exactly on target made every replan that saw a fraction of a
        degree short start another run. Rounded to the heat pump's nearest half
        degree, at most a quarter degree below the need (replayed on 14 real
        runs, 12-26 September 2026: no afternoon top-up). Not below the heat
        pump's own limit: a run the booster finishes ends on its thermostat.
        """

        schedule = result.schedule
        heating_now = bool(schedule) and schedule[0] == 1
        start = next((k for k, on in enumerate(schedule) if on), None)
        next_start = "unknown"
        setpoint = "unknown"

        if start is not None:
            end = start
            while end + 1 < len(schedule) and schedule[end + 1]:
                end += 1

            end_temperature = result.temperatures[min(end + 1, len(schedule) - 1)]
            overshoot_k = thermal_model and thermal_model.setpoint_overshoot_k
            limit_c = thermal_model and thermal_model.heat_pump_max_tank_temperature_c

            if overshoot_k and (limit_c is None or end_temperature <= limit_c):
                end_temperature -= overshoot_k

            next_start = times[start].isoformat()
            setpoint = str(math.floor(round(2 * end_temperature, 2) + 0.5) / 2)

        self.home_assistant.set_state(
            self.DHW_STATUS_ENTITY,
            "on" if heating_now else "off",
            {"friendly_name": "Home Optimizer DHW status"},
        )
        self.home_assistant.set_state(
            self.DHW_START_ENTITY,
            next_start,
            {"friendly_name": "Home Optimizer DHW start", "device_class": "timestamp"},
        )
        self.home_assistant.set_state(
            self.DHW_SETPOINT_ENTITY,
            setpoint,
            {
                "friendly_name": "Home Optimizer DHW setpoint",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
            },
        )
