import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.state import StateManager
from domain.config import BoilerConfig, Config, Price, Tariff
from domain.jobs import OptimizeConfig
from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.physics import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    indoor_dew_point_c,
    mains_water_temperature_c,
)
from domain.sensors import SensorReference
from domain.state import BaseloadHabit, SeriesPoint, State, TapHabit, ZoneHabit
from domain.time import local_day_start, to_local_time
from features import ecodan
from features.baseload import BaseloadForecaster
from features.boiler import BoilerThermalIdentifier
from features.building import BuildingThermalIdentifier
from features.cop import HeatPumpCOPIdentifier
from features.dataset import DatasetBuilder, DatasetLoader
from features.dew_point import DewPointIdentifier
from features.floor import FloorCircuitIdentifier
from features.optimizer import SOLAR_SCENARIO_WEIGHTS, MPCOptimizer, surplus_scenarios
from features.price import (
    PRICE_PERIOD,
    SOLCAST_PERIOD,
    WIND_PERIOD,
    PriceIdentifier,
    on_quarters,
    predict_price,
    published_prices,
)
from features.tap import TapForecaster
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
    DHW_SOC_ENTITY = "sensor.home_optimizer_dhw_soc"
    # The floor's, published the same way while the plan drives it (see
    # publish_zone).
    ZONE_STATUS_ENTITY = "binary_sensor.home_optimizer_zone_status"
    ZONE_START_ENTITY = "sensor.home_optimizer_zone_start"
    ZONE_SUPPLY_ENTITY = "sensor.home_optimizer_zone_supply"
    # The auto-adaptive setpoint bias for the run (see features.ecodan), only with
    # HeatPumpConfig.auto_adaptive on.
    ZONE_BIAS_ENTITY = "sensor.home_optimizer_zone_bias"
    ZONE_SETPOINT_ENTITY = "sensor.home_optimizer_zone_setpoint"
    # The rest's thermostats' setpoint, only with BuildingConfig.rest_temperature
    # set (see publish_zone).
    ZONE_REST_SETPOINT_ENTITY = "sensor.home_optimizer_zone_rest_setpoint"

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

        mpc_config = MPCConfig(
            weight_comfort_eur_per_k2h=config.building.comfort_weight
        )

        # From the quarter hour running now to the end of tomorrow, the
        # clock's choice rather than however far the last solar prediction
        # happened to reach: the weather forecasts end there, so planning
        # beyond it would plan on guesses, and solved over 48 instead of 26
        # hours the same first run took twice as long (real data, 2 Oct). At
        # most optimize_config.steps. A forecast that falls short is extended
        # (see StateManager.extend_forecast), except the sun.
        step = timedelta(hours=mpc_config.step_hours)
        now = datetime.now(timezone.utc)
        first = pd.Timestamp(now).floor(step).to_pydatetime()
        steps = min(
            optimize_config.steps, (local_day_start(now, days=2) - first) // step
        )
        forecast_times = [first + i * step for i in range(steps)]

        # The baseload ahead (see BaseloadForecaster), into the predictions the
        # plan, the zone and the charts read.
        for name, series in self._baseload(state, config, now, forecast_times).items():
            self.state_manager.update_prediction(name, series)

        state = self.state_manager.load()

        def extended(
            points: list[SeriesPoint], fallback: float | None = None
        ) -> tuple[float, ...]:
            return tuple(
                self.state_manager.extend_forecast(
                    {p.time: p.value for p in points}, forecast_times, fallback
                )
            )

        # No sun where Solcast has none: the weather another day had is no
        # forecast, and a plan counting on sun that does not come heats on
        # the grid at the last moment.
        def solar(points: list[SeriesPoint]) -> tuple[float, ...]:
            return tuple(self.state_manager.align_predictions(points, forecast_times))

        predictions = state.predictions
        solar_forecast = list(solar(predictions.solar))
        # Within the horizon Solcast normally has every step, so a gap means
        # it has stopped updating - worth fixing at the source.
        forecast_solar = {p.time for p in predictions.solar}
        missing = [t for t in forecast_times if t not in forecast_solar]

        if missing:
            logger.warning(
                "Solar forecast misses %d of %d steps (first %s): planned "
                "without sun there; is Solcast still updating?",
                len(missing),
                len(forecast_times),
                to_local_time(missing[0]).strftime("%a %H:%M"),
            )

        if not predictions.solar_p10 or not predictions.solar_p90:
            solar_p10, solar_p90 = (), ()
        else:
            solar_p10 = solar(predictions.solar_p10)
            solar_p90 = solar(predictions.solar_p90)

        if not predictions.baseload_p10 or not predictions.baseload_p90:
            baseload_p10, baseload_p90 = (), ()
        else:
            baseload_p10, baseload_p90 = (
                tuple(
                    self.state_manager.baseload_forecast(
                        state, forecast_times, now, points
                    )
                )
                for points in (predictions.baseload_p10, predictions.baseload_p90)
            )

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

        import_price, export_price = (
            self._price(key, price, state, forecast_times)
            for key, price in (
                ("import", config.prices.import_),
                ("export", config.prices.export),
            )
        )
        heat_pump_changes = self._heat_pump_changes(config, now)
        boiler_on_current, run_start, idle_elapsed_hours = self.dhw_timing(
            *heat_pump_changes,
            dhw_state,
            now,
            state.measurements.heat_pump.defrost,
        )
        compressor_elapsed_hours = (
            (now - run_start).total_seconds() / 3600.0 if run_start else 0.0
        )
        # Handed over from the zone: the compressor running when the state
        # changed into DHW.
        frequency_before = [
            p for p in heat_pump_changes[1] if run_start and p.time < run_start
        ]
        compressor_handed_over = bool(frequency_before) and (
            frequency_before[-1].value > 0.0
        )
        # The last defrost in the run under way: the compressor rises again
        # from its end (see MPCInput.defrost_ended_hours).
        defrosts = [
            p
            for p in state.measurements.heat_pump.defrost
            if run_start is not None and p.time >= run_start
        ]
        defrost_ended_hours = (
            (0.0 if defrosts[-1].value else (now - defrosts[-1].time).total_seconds())
            / 3600.0
            if any(p.value for p in defrosts)
            else None
        )
        # The plan starts now, partway into the quarter its first step belongs
        # to (see MPCInput.first_step_hours).
        first_step_hours = (
            forecast_times[0] + timedelta(hours=mpc_config.step_hours) - now
        ).total_seconds() / 3600.0

        if not 0.0 < first_step_hours < mpc_config.step_hours:
            first_step_hours = None
        run_start_temp_top = run_start_temp_bottom = None
        run_start_stratification_k = run_start_ambient_c = None
        stratification = {
            p.time: p.value for p in state.measurements.heat_pump.boiler.stratification
        }

        run_minutes = (
            self._run_minutes(config, run_start, now) if run_start is not None else None
        )

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
            ambient = {p.time: p.value for p in boiler.ambient_temperature}

            if before_top and before_bottom:
                run_start_temp_top = before_top[-1].value
                run_start_temp_bottom = before_bottom[-1].value
                run_start_stratification_k = stratification.get(before_top[-1].time)
                run_start_ambient_c = ambient.get(before_top[-1].time)

            # Better the minute before the run began: a tap after that
            # quarter's last reading took its heat from the tank before the
            # run (real data, 31 Oct 2026: 35.5/25.0 degC at 12:29, a tap at
            # 12:33 to 33.0/22.5, the run at 12:43 - the plan 1.25 K warm).
            # The stratification is held through it, as at rest (see
            # physics.tank_stratification_k).
            before = (
                run_minutes[run_minutes["time"] + pd.Timedelta("1min") <= run_start]
                if run_minutes is not None
                else pd.DataFrame()
            )

            if not before.empty:
                run_start_temp_top = float(before["T_top"].iloc[-1])
                run_start_temp_bottom = float(before["T_bottom"].iloc[-1])
                run_start_ambient_c = float(before["T_ambient"].iloc[-1])
                run_start_stratification_k = max(
                    run_start_stratification_k or 0.0,
                    run_start_temp_top - run_start_temp_bottom,
                )

        # The quarters the run under way has ended so far, drawn by the heat
        # balance the plan starts from (see MPCOptimizer.run_temperature): its
        # sensors lag that heat (see StateManager._tank_temperature).
        step = timedelta(hours=mpc_config.step_hours)
        run_quarters = []

        if run_start is not None:
            time = forecast_times[0]

            while time > run_start:
                run_quarters.insert(0, time)
                time -= step

        # The heat the run under way has put in by those quarters and by now,
        # as the plan starts from it.
        run_heat_j = (
            self._run_heat_j(run_minutes, run_start, [*run_quarters, now])
            if run_minutes is not None
            else None
        )

        # The tap draws: the past weeks' own profile, held to what today has
        # drawn up to this plan (see TapForecaster). Kept in the state to be
        # looked at; no draws assumed beyond it, the assumption made before
        # it existed.
        tap = TapForecaster(self.models_path)
        draws = tap.prepare(
            # From a reading before midnight: the boiler's prepare() drops the
            # first, which has no interval of its own.
            self.loader.load(
                tap.dataset(config),
                local_day_start(now).astimezone(timezone.utc)
                - 2 * pd.Timedelta(BoilerThermalIdentifier.SAMPLE_INTERVAL),
                now,
            )
        )
        tap_w = tap.predict(
            draws,
            steps=len(forecast_times),
            habit=self._tap_habit(state, config, tap, now),
        )
        self.state_manager.update_prediction("tap", tap_w)
        tap_forecast = extended(
            [SeriesPoint(time=t, value=float(w)) for t, w in tap_w.items()], 0.0
        )
        run_end = now - timedelta(hours=idle_elapsed_hours)
        hot_water_c = (
            self.hot_water_c(
                state.measurements.heat_pump.boiler.top_temperature,
                run_end,
                mpc_config,
            )
            if not boiler_on_current
            else None
        )
        mains_c = mains_water_temperature_c(to_local_time(now).timetuple().tm_yday)
        day_start = local_day_start(now).astimezone(timezone.utc)
        day_end = local_day_start(now, days=1).astimezone(timezone.utc)
        # Without a DHW run in the changes loaded, run_end is only their first.
        target_temps = self.served_targets(
            target_temps,
            forecast_times,
            self.state_manager.resolve_schedule(
                config.heat_pump.boiler.target_temperature,
                [day_start + i * step for i in range((day_end - day_start) // step)],
            ),
            run_end
            if any(p.value == dhw_state for p in heat_pump_changes[0])
            else None,
            hot_water_c,
            mains_c,
            lambda since: (
                tap.drawn_j(draws, since)
                / (thermal_model.volume_l * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K)
            ),
            now,
        )

        # Empty without any Open-Meteo forecast (fresh install, fetch not yet
        # run), not 0 deg C: MPCOptimizer then falls back to the flat
        # boiler_electrical_power_w assumption.
        outdoor_temperature_forecast = (
            extended(state.forecast.open_meteo.temperature)
            if state.forecast.open_meteo.temperature
            else ()
        )

        data = MPCInput(
            solar_forecast_w=solar_forecast,
            hot_water_c=hot_water_c,
            mains_water_temperature=mains_c,
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
            defrost_ended_hours=defrost_ended_hours,
            compressor_handed_over=compressor_handed_over,
            run_start_temp_top=run_start_temp_top,
            run_start_ambient_c=run_start_ambient_c,
            run_heat_j=run_heat_j[-1] if run_heat_j else None,
            run_start_temp_bottom=run_start_temp_bottom,
            first_step_hours=first_step_hours,
            idle_elapsed_hours=idle_elapsed_hours,
            baseload_forecast_w=tuple(
                self.state_manager.baseload_forecast(state, forecast_times, now)
            ),
            baseload_p10_w=baseload_p10,
            baseload_p90_w=baseload_p90,
            previous_tank_on=tuple(
                int(on)
                for on in self.state_manager.align_predictions(
                    state.schedule.heat_pump.boiler.on, forecast_times
                )
            ),
            previous_space_on=tuple(
                int(on)
                for on in self.state_manager.align_predictions(
                    state.schedule.building.on, forecast_times
                )
            ),
            import_price_eur_per_kwh=import_price,
            export_price_eur_per_kwh=export_price,
        )
        data = self._with_legionella(
            data, forecast_times, config.heat_pump.boiler, thermal_model, mpc_config
        )

        # One plan for the one heat pump: the tank and, where it can be
        # planned, the zone, sharing the compressor (see publish_dhw and
        # publish_zone).
        dhw_only = MPCOptimizer(
            thermal_model=thermal_model, config=mpc_config, cop_model=cop_model
        )
        optimizer, planned = dhw_only, data
        estimate, zone = self._zone(
            state,
            config,
            forecast_times,
            mpc_config,
            optimize_config.cooling,
            now,
        )

        if zone is not None:
            inputs, models = zone
            optimizer = MPCOptimizer(
                thermal_model=thermal_model,
                config=mpc_config,
                cop_model=cop_model,
                **models,
            )
            states = config.heat_pump.states
            mode = states.cooling if inputs["zone_cooling"] else states.heating
            running = self.zone_running(
                *heat_pump_changes, mode, state.measurements.heat_pump.defrost
            )
            planned = dataclasses.replace(
                data,
                **inputs,
                space_on_current=running,
                space_elapsed_hours=(
                    self.zone_run_hours(
                        *heat_pump_changes,
                        mode,
                        states.dhw,
                        now,
                        state.measurements.heat_pump.defrost,
                    )
                    if running
                    else 0.0
                ),
            )

        # Stopped at its time limit, a plan with the zone is the best found by
        # then, and kept: proving a 48-hour plan with the zone takes minutes,
        # finding it seconds (real data: within 0.008 EUR of the optimum at
        # 60 s, proven only at 144 s), and the next quarter's solve starts from
        # it. Such plans were once dropped for hot water alone, after one was
        # held by every warm start after (8 Oct 2026: 0.751 EUR against
        # 0.672) - likely HiGHS reusing the warm start's instance, which
        # proved plans optimal that were not (see MPCOptimizer.solve), rather
        # than the time limit.
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

        run_tank = [
            (
                time,
                optimizer.run_temperature(
                    planned, (time - run_start).total_seconds() / 3600.0, heat_j
                ),
            )
            for time, heat_j in zip(
                # The heat by now, last, is the plan's start.
                run_quarters,
                run_heat_j or [None] * len(run_quarters),
                strict=False,
            )
        ]
        top_c, bottom_c = optimizer.sensor_temperatures(planned, result)
        self.state_manager.update_schedule(
            schedule=result.schedule,
            temperatures=result.temperatures,
            top_temperatures=top_c,
            bottom_temperatures=bottom_c,
            power_w=result.electrical_power_w,
            times=forecast_times,
            heat_w=result.heat_w,
            dhw_running=boiler_on_current,
            run_temperatures=[
                SeriesPoint(time=t, value=c) for t, c in run_tank if c is not None
            ],
        )

        self.publish_dhw(
            result,
            forecast_times,
            thermal_model,
            self.tank_level(
                # The tank the plan starts from: mixed, or by its heat balance
                # through a run under way (see MPCOptimizer._initial_temperature).
                result.temperatures[0],
                data.mains_water_temperature,
                self._dhw_setpoint_c(config, now),
                thermal_model.volume_l,
                tap.drawn_j(draws),
            ),
        )

        if optimize_config.explain:
            self.explain_dhw_plan(optimizer, planned, result, forecast_times)

        # Saved with the plan, not before the solve: a cold solve takes a
        # minute, and in it the dashboard drew this estimate's mass, the zone
        # left alone, beside the previous plan's floor runs.
        if estimate is not None:
            # The thermostat's room, which comfort is set in, and the slab
            # under it.
            self.state_manager.update_zone(
                temperature=estimate["measured"].dropna(),
                thermal_mass=estimate["slab_living"],
            )

        # Empty without a zone plan, so the dashboard shows none rather than
        # an old one.
        self.state_manager.update_building_schedule(
            heat_w=result.space_heat_w,
            temperatures=result.zone_temperatures,
            times=forecast_times,
            supply_c=result.space_supply_c,
            power_w=result.space_electrical_w,
            on=result.space_schedule,
            thermal_mass=result.zone_mass_temperatures,
            rest=result.zone_rest_temperatures,
        )
        self.publish_zone(
            result,
            forecast_times,
            state,
            config,
            planned if zone is not None else None,
            heat_pump_changes[0],
            now,
            # The compressor's least heat into the floor (see
            # features.ecodan.setpoint_bias_k), and the controller's least
            # supply.
            *(
                (
                    optimizer.floor_circuit_model.min_heat_w,
                    optimizer.floor_circuit_model.min_supply_c,
                )
                if optimizer.floor_circuit_model is not None
                else ()
            ),
        )
        # Last, so it reads after what was published.
        self.log_day_costs(
            config, state, planned, result, optimizer, forecast_times, now
        )

    def _run_minutes(
        self, config: Config, run_start: datetime, now: datetime
    ) -> pd.DataFrame:
        """The DHW run under way per minute, as the boiler model is identified
        on (see BoilerThermalIdentifier.prepare), from the minute before it
        began: the tank as the run found it and the heat it has put in."""

        boiler = BoilerThermalIdentifier()
        # prepare() drops the first reading, which has no interval of its own.
        return boiler.prepare(
            self.loader.load(
                boiler.dataset(config),
                run_start - 3 * pd.Timedelta(boiler.SAMPLE_INTERVAL),
                now,
            )
        )

    @staticmethod
    def _run_heat_j(
        df: pd.DataFrame, run_start: datetime, times: list[datetime]
    ) -> list[float] | None:
        """The heat the DHW run under way had carried to the tank by each of
        `times` (J): its calorimetry (see _run_minutes), each minute's reading
        held until the next. None where it shows none - no flow reported yet,
        say - and the plan's own heat input stands in. Taken from the minutes,
        not the state's quarter hours: the heat is flow times temperature
        rise, and the product of two quarter-hour means is not the quarter's
        heat."""

        heat_w = pd.to_numeric(df["q_in_override_w"], errors="coerce")

        if heat_w.isna().all():
            return None

        start = df["time"].clip(lower=pd.Timestamp(run_start))
        end = df["time"].shift(-1, fill_value=pd.Timestamp(times[-1]))

        return [
            float(
                (
                    heat_w.fillna(0.0)
                    * (end.clip(upper=pd.Timestamp(t)) - start)
                    .dt.total_seconds()
                    .clip(lower=0.0)
                ).sum()
            )
            for t in times
        ]

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
    def hot_water_c(
        top: list[SeriesPoint], since: datetime, mpc_config: MPCConfig
    ) -> float | None:
        """The top sensor's highest since the last DHW run ended at `since`
        (deg C, see MPCInput.hot_water_c): in each quarter that ended after
        it, labelled at its start - the quarter under way too. None without
        one."""

        step = timedelta(hours=mpc_config.step_hours)
        readings = [p.value for p in top if p.time + step > since]

        return max(readings) if readings else None

    @staticmethod
    def served_targets(
        targets: tuple[float, ...],
        times: Sequence[datetime],
        today: Sequence[SeriesPoint],
        run_end: datetime | None,
        hot_water_c: float | None,
        mains_c: float,
        drawn_k: Callable[[datetime], float],
        now: datetime,
    ) -> tuple[float, ...]:
        """The tank targets with today's lowered by what has been drawn
        (drawn_k(since), K of the whole tank) since today's last DHW run ended
        at run_end, where that run heated the tank to them (hot_water_c, see
        hot_water_c). `today`: the day's targets per quarter hour from its
        local midnight.

        A target window - quarters with a target above the mains (mains_c,
        deg C) - is only lowered by what was drawn since the previous window
        today ended: draws before that were the previous target's water
        (morning showers after a 07:00 target, then a run left for the 18:00
        one). A target at or below the mains is none: draws cannot take the
        tank below the mains (see MPCOptimizer).

        A target is the hot water made ready for the day's draws: drawn before
        its time, it was used early, not lost, and heating it again only
        heats for draws the day has already had (real plan, 31 Oct 2026: a
        run to 48 degC at noon, a shower at 16:45 local, then a run planned at
        17:15 to have 45 degC again at 18:00). What the rest of the day still
        draws stays covered: the tap forecast holds it to a heavy day less
        what has been drawn (see TapForecaster.BUDGET_DAYS), 5.6 kWh, under
        the 7.3 kWh a 200 l tank at 45 degC holds above the mains. Tomorrow's
        targets are left as they are: their water is not made yet."""

        if run_end is None or hot_water_c is None or run_end < local_day_start(now):
            return targets

        since: dict[datetime, datetime] = {}
        window_since = ended = None

        for i, point in enumerate(today):
            previous = i > 0 and today[i - 1].value > mains_c

            if point.value > mains_c and not previous:
                window_since = ended
            elif point.value <= mains_c and previous:
                ended = point.time

            since[point.time] = max(run_end, window_since or run_end)

        drawn = {start: drawn_k(start) for start in set(since.values())}

        return tuple(
            target - drawn[since[time]]
            if time in since and mains_c < target <= hot_water_c
            else target
            for time, target in zip(times, targets, strict=True)
        )

    @staticmethod
    def value_at(points: Sequence[SeriesPoint], time: datetime, value):
        """The value a series of changes holds at `time`, each from its own
        time until the next; `value` before the first."""

        for point in points:
            if point.time > time:
                break
            value = point.value

        return value

    @staticmethod
    def dhw_timing(
        heat_pump_state: list[SeriesPoint],
        compressor_frequency: list[SeriesPoint],
        dhw_state: str,
        now: datetime,
        defrost: Sequence[SeriesPoint] = (),
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
        start and count, as does a pause the compressor comes up from again,
        and a stop in a defrost (its changes in `defrost`).
        Without a frequency the state is all there is, and
        assuming the compressor ran keeps a real run protected. The pause is
        taken from the change that ended the last DHW run - from the first
        change loaded (the previous local midnight's, see _heat_pump_changes)
        if none shows, which errs short and keeps it.
        """

        if not heat_pump_state:
            return False, None, 0.0

        def frequency_at(time: datetime) -> float:
            return Optimization.value_at(compressor_frequency, time, 1.0)

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
            running = [frequency_at(time) > 0.0 for time in changes]
            # At 0 Hz after the compressor has run, the booster has taken over.
            # A compressor that came up again had only paused, and its run goes
            # on from the change into DHW (real data, 17, 23 and 26 Oct 2026: 2
            # minutes at 0 Hz, the booster off, mid-run; taken from the restart
            # the plan started 3-4 K cold, from the sensors that lag the heat).
            booster = any(running) and not running[-1]

            if booster:
                # Nor a stop in a defrost: the compressor stops there and
                # comes up again after it (real data, 31 Oct 2026: 0 Hz from
                # 12:58 to 13:00 in a defrost, the plan started from sensors
                # stirred 5 K below the run's heat).
                stop = max(i for i in range(1, len(changes)) if running[i - 1])
                booster = not Optimization.value_at(defrost, changes[stop], False)

            return True, None if booster else dhw_start, 0.0

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

    def _price(
        self,
        key: Literal["import", "export"],
        price: Price,
        state: State,
        times: list[datetime],
    ) -> tuple[float, ...]:
        """A price per step (EUR/kWh). A sensor's as published where they are
        (see config.Price), forecast beyond (see features.price). Without a
        calibrated forecast the mean of those published stands in, a flat
        price that moves nothing into the steps not known; empty without any,
        so the plan falls back to MPCConfig's flat ones."""

        if isinstance(price, Tariff):
            return tuple(
                weekend if to_local_time(t).weekday() >= 5 else weekday
                for t, weekday, weekend in zip(
                    times,
                    self._price(key, price.weekdays, state, times),
                    self._price(key, price.weekend, state, times),
                    strict=True,
                )
            )

        if not isinstance(price, SensorReference):
            return tuple(
                point.value
                for point in self.state_manager.resolve_schedule(price, times)
            )

        sensor = price

        if sensor.attribute is None:
            logger.warning(
                "Prices from %s need the attribute they are published in (e.g. "
                "data): planning on flat prices.",
                sensor.entity_id,
            )
            return ()

        # Whatever the sensor publishes, the plan still runs: on flat prices.
        try:
            published = self.loader.load(published_prices(sensor), times[0], times[-1])
        except Exception as error:
            logger.warning("No prices from %s: %s", sensor.entity_id, error)
            published = pd.DataFrame()

        def series(index, values) -> pd.Series:
            return pd.Series(
                np.asarray(values, dtype=float), index=pd.DatetimeIndex(index)
            )

        def points(points: list[SeriesPoint]) -> pd.Series:
            return series([p.time for p in points], [p.value for p in points])

        prices = on_quarters(
            series(published["time"], published["price"])
            if "price" in published.columns
            else pd.Series(dtype=float),
            times,
            PRICE_PERIOD,
        )
        missing = np.isnan(prices)

        if not missing.any():
            return tuple(prices.tolist())

        identifier = PriceIdentifier(key)
        identifier.load(self.models_path)

        if identifier.model is not None:
            forecast = predict_price(
                identifier.model,
                times,
                on_quarters(points(state.forecast.solcast.p50), times, SOLCAST_PERIOD)
                / 1000.0,
                on_quarters(
                    points(state.forecast.open_meteo.wind_speed), times, WIND_PERIOD
                ),
            )
        elif not missing.all():
            forecast = np.full(len(times), np.nanmean(prices))
        else:
            logger.warning(
                "No prices published by %s and none forecast: planning on flat prices.",
                sensor.entity_id,
            )
            return ()

        logger.info(
            "%s prices: %d of %d steps published, from %s %s",
            key.capitalize(),
            (~missing).sum(),
            len(times),
            to_local_time(times[int(np.argmax(missing))]).strftime("%a %H:%M"),
            "forecast" if identifier.model is not None else "at their mean",
        )

        return tuple(np.where(missing, forecast, prices).tolist())

    @classmethod
    def dhw_summary(
        cls,
        result: MPCResult,
        times: list[datetime],
        optimizer: MPCOptimizer,
        data: MPCInput,
    ) -> str:
        """What a hot water plan does and what it expects that to cost, per
        local day: each run's time and end temperature, the day's electricity
        and how much of it is expected from the grid, as the plan's own
        objective counts them (see MPCResult.electricity_kwh) - so the cheaper
        of two plans here is the one the optimizer would choose - and the total.
        The cost is the import at the price and the own sun at the export it
        forgoes; not the whole objective, which also prices starts and
        shortfalls and so is no amount of money. Only the tank's: the plan's
        electricity is the whole heat pump's, the zone's included."""

        runs = cls.dhw_runs(result.schedule)

        if not runs:
            return "no run within the horizon"

        # A step serves the tank or the zone, never both, so the tank's share
        # of a step's draw is all of it or none (real plan, 10 Oct 2026: 4.20
        # kWh reported for a run of 1.22, the floor's 2.98 after it counted).
        zone_w = result.space_electrical_w or (0.0,) * len(result.electrical_power_w)
        tank_share = [
            tank / (tank + zone) if tank + zone > 0.0 else 0.0
            for tank, zone in zip(result.electrical_power_w, zone_w, strict=True)
        ]
        electricity_step_kwh = [
            kwh * share
            for kwh, share in zip(result.electricity_step_kwh, tank_share, strict=False)
        ]
        grid_step_kwh = [
            kwh * share
            for kwh, share in zip(result.grid_step_kwh, tank_share, strict=False)
        ]
        cost_step_eur = [
            grid * import_ + (electricity - grid) * export
            for electricity, grid, import_, export in zip(
                electricity_step_kwh,
                grid_step_kwh,
                *optimizer.prices(data),
                strict=False,
            )
        ]
        step = timedelta(hours=optimizer.config.step_hours)
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
                for kwh, d in zip(electricity_step_kwh, day, strict=False)
                if d == date
            )
            grid_kwh = sum(
                kwh for kwh, d in zip(grid_step_kwh, day, strict=False) if d == date
            )
            cost_eur = sum(
                eur for eur, d in zip(cost_step_eur, day, strict=False) if d == date
            )
            days.append(
                f"{date:%a} {runs_text}: {electricity_kwh:.2f} kWh, of which sun "
                f"{electricity_kwh - grid_kwh:.2f} and grid {grid_kwh:.2f} "
                f"(expected), EUR {cost_eur:.3f}"
            )

        return " | ".join(days) + (
            f" | total {sum(electricity_step_kwh):.2f} kWh, "
            f"EUR {sum(cost_step_eur):.3f}"
        )

    # What a day's line adds up, per row (kWh, the cost EUR).
    DAY_COLUMNS = (
        "import",
        "export",
        "solar",
        "dhw",
        "heating",
        "cooling",
        "standby",
        "house",
        "cost",
    )

    def log_day_costs(
        self,
        config: Config,
        state: State,
        data: MPCInput,
        result: MPCResult,
        optimizer: MPCOptimizer,
        times: list[datetime],
        now: datetime,
    ) -> None:
        """Logs per local day the grid's import and export, the solar, the heat
        pump's electricity per mode, the rest of the house and their cost:
        today measured up to now and planned on from it, tomorrow planned."""

        measured = self.measured_energy(config, state, now)
        planned = self.planned_energy(result, data, times, optimizer)
        step = timedelta(hours=optimizer.config.step_hours)
        logger.info(
            "Planned runs (kWh):\n%s",
            "\n".join(self.planned_runs(result, data, planned, times, step)),
        )

        # The table's columns: header, and what each day's cell sums.
        columns = (
            ("eur", "cost", None),
            ("meas", "cost", "measured"),
            ("plan", "cost", "planned"),
            ("in", "import", None),
            ("out", "export", None),
            ("solar", "solar", None),
            ("hp", ("dhw", "heating", "cooling", "standby"), None),
            ("dhw", "dhw", None),
            ("heat", "heating", None),
            ("cool", "cooling", None),
            ("stby", "standby", None),
            ("house", "house", None),
        )
        lines = [f"{'day':<10}" + "".join(f"{h:>7}" for h, _, _ in columns)]

        for date in dict.fromkeys(to_local_time(t).date() for t in times):
            parts = {
                name: frame[[to_local_time(t).date() == date for t in frame["time"]]]
                for name, frame in (("measured", measured), ("planned", planned))
            }
            cells = []

            for _, column, part in columns:
                rows = parts[part] if part else pd.concat(parts.values())
                value = float(rows[list(np.atleast_1d(column))].to_numpy().sum())
                cells.append(f"{value:>7.2f}" if abs(value) >= 0.005 else f"{'-':>7}")

            lines.append(f"{date:%a %d %b}" + "".join(cells))

        logger.info("Day costs (EUR, kWh):\n%s", "\n".join(lines))

    @staticmethod
    def planned_runs(
        result: MPCResult,
        data: MPCInput,
        planned: pd.DataFrame,
        times: list[datetime],
        step: timedelta,
    ) -> list[str]:
        """The plan's runs as table lines, in order of their start: which, its
        day, start and end and the electricity it draws (see
        planned_energy)."""

        modes = [("dhw", "dhw", result.schedule)]

        if result.space_schedule:
            zone = "cool" if data.zone_cooling else "heat"
            modes.append(
                (
                    zone,
                    "cooling" if data.zone_cooling else "heating",
                    result.space_schedule,
                )
            )

        runs = []

        for label, column, schedule in modes:
            k = 0

            while k < len(schedule):
                if not schedule[k]:
                    k += 1
                    continue

                end = k

                while end + 1 < len(schedule) and schedule[end + 1]:
                    end += 1

                runs.append(
                    (
                        times[k],
                        label,
                        times[end] + step,
                        float(planned[column].iloc[k : end + 1].sum()),
                    )
                )
                k = end + 1

        lines = [f"{'day':<12}{'start':<7}{'end':<7}{'run':<6}{'kwh':>5}"]

        for start, label, end, kwh in sorted(runs):
            lines.append(
                f"{to_local_time(start):%a %d %b}  {to_local_time(start):%H:%M}  "
                f"{to_local_time(end):%H:%M}  {label:<6}{kwh:>5.2f}"
            )

        return lines if runs else [*lines, "none"]

    def measured_energy(
        self, config: Config, state: State, now: datetime
    ) -> pd.DataFrame:
        """Today so far per quarter hour (see DAY_COLUMNS), from the meters'
        minutes over time: import and export apart per minute, as the grid
        meter counts them - over a quarter hour's mean they would net."""

        frame = self.loader.load(
            DatasetBuilder()
            .timeseries("grid", config.grid, aggregation="time_mean", interval="1m")
            .timeseries("solar", config.solar, aggregation="time_mean", interval="1m")
            .timeseries(
                "power", config.heat_pump.power, aggregation="time_mean", interval="1m"
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                aggregation="last",
                interval="1m",
                fill="previous",
            )
            .build(),
            local_day_start(now).astimezone(timezone.utc),
            now,
        )
        frame = frame.dropna(subset=["grid", "solar", "power"])

        if frame.empty:
            return pd.DataFrame(columns=["time", *self.DAY_COLUMNS])

        kwh = 1.0 / 60.0 / 1000.0
        grid = frame["grid"].astype(float)
        power = frame["power"].astype(float)
        states = config.heat_pump.states
        mode = frame["state"]
        minutes = pd.DataFrame(
            {
                "import": grid.clip(lower=0.0) * kwh,
                "export": (-grid).clip(lower=0.0) * kwh,
                "solar": frame["solar"].astype(float).clip(lower=0.0) * kwh,
                "dhw": power.where(mode == states.dhw, 0.0) * kwh,
                "heating": power.where(mode == states.heating, 0.0) * kwh,
                "cooling": power.where(mode == states.cooling, 0.0) * kwh,
                "standby": power.where(
                    ~mode.isin([states.dhw, states.heating, states.cooling]), 0.0
                )
                * kwh,
                "house": (grid + frame["solar"].astype(float) - power) * kwh,
                "time": pd.to_datetime(frame["time"]).dt.floor("15min"),
            }
        )
        quarters = minutes.groupby("time").sum().reset_index()
        quarter_times = [t.to_pydatetime() for t in quarters["time"]]
        prices = config.prices
        quarters["cost"] = quarters["import"] * np.array(
            self._price("import", prices.import_, state, quarter_times)
        ) - quarters["export"] * np.array(
            self._price("export", prices.export, state, quarter_times)
        )

        return quarters

    @staticmethod
    def planned_energy(
        result: MPCResult,
        data: MPCInput,
        times: list[datetime],
        optimizer: MPCOptimizer,
    ) -> pd.DataFrame:
        """The plan per step from now (see DAY_COLUMNS): the heat pump's draw
        split between the tank and the zone, and the house's import and export
        at the median forecasts."""

        step_hours = optimizer.config.step_hours
        hours = np.array(
            [
                data.first_step_hours
                if j == 0 and data.first_step_hours
                else step_hours
                for j in range(len(result.electricity_step_kwh))
            ]
        )
        drawn = np.array(result.electricity_step_kwh, dtype=float)
        tank_w = np.abs(np.array(result.electrical_power_w, dtype=float))
        zone_w = np.abs(
            np.array(result.space_electrical_w or (0.0,) * len(drawn), dtype=float)
        )
        # A step serves the tank or the zone (see dhw_summary).
        tank_share = np.divide(
            tank_w, tank_w + zone_w, out=np.zeros_like(drawn), where=tank_w + zone_w > 0
        )
        # The house's import and export at the median sun and mean baseload, the
        # day a forecast expects: the plan's own scenarios weight each
        # quarter's p90 baseload - its peaks - as if a third of the day drew
        # it (7.1 kWh for 7 Nov 2026 against 5.55 at p50; 6 Nov drew 5.2).
        baseload_w = np.array(
            data.baseload_forecast_w or (0.0,) * len(drawn), dtype=float
        )[: len(drawn)]
        solar_w = np.array(data.solar_forecast_w[: len(drawn)], dtype=float)
        grid_w = drawn / hours * 1000.0 + baseload_w - solar_w

        import_price, export_price = optimizer.prices(data)
        zone = drawn * (1.0 - tank_share)
        frame = pd.DataFrame(
            {
                "time": times[: len(drawn)],
                "import": np.clip(grid_w, 0.0, None) * hours / 1000.0,
                "export": np.clip(-grid_w, 0.0, None) * hours / 1000.0,
                "solar": solar_w * hours / 1000.0,
                "dhw": drawn * tank_share,
                "heating": 0.0 if data.zone_cooling else zone,
                "cooling": zone if data.zone_cooling else 0.0,
                "standby": 0.0,
                "house": baseload_w * hours / 1000.0,
            }
        )
        frame["cost"] = frame["import"] * np.array(import_price[: len(drawn)]) - frame[
            "export"
        ] * np.array(export_price[: len(drawn)])

        return frame

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
        lines = [f"  {'plan:':<16}{cls.dhw_summary(result, times, optimizer, data)}"]

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
            summary = cls.dhw_summary(optimizer.solve(forced), times, optimizer, forced)
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
        window = timedelta(hours=mpc_config.legionella_overdue_hours)
        # To the end of the local day the interval ends on: the daily maxima are
        # UTC days, and a deadline at their midnight fell in the night, leaving
        # only night hours - grid power - to choose from. Counted in calendar
        # days: seven times 24 hours across a change to winter time ended an
        # hour early, on the day before.
        deadline = (
            local_day_start(disinfected.max(), days=legionella.interval + 1)
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
        surplus_scenarios). The run lasts as long as the heat pump takes to
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
            data.ambient_temperature,
            data.current_stratification_k,
        )
        limit_c = thermal_model.heat_pump_max_tank_temperature_c or temperature
        seconds = (
            heat_capacity_j_per_k
            * max(min(temperature, limit_c) - start_c, 0.0)
            / (
                (thermal_model.q_in_steady_w or thermal_model.q_in_nominal_w)
                * (thermal_model.calorimetric_gain or 1.0)
            )
        )

        if thermal_model.booster_heat_w and temperature > limit_c:
            seconds += (
                heat_capacity_j_per_k
                * (temperature - max(limit_c, start_c))
                / thermal_model.booster_heat_w
            )

        run_steps = max(1, math.ceil(seconds / (mpc_config.step_hours * 3600.0)))
        surplus = sum(weight * surplus for weight, surplus in surplus_scenarios(data))
        covered = np.concatenate(([0.0], np.cumsum(surplus)))
        candidates = [k for k in range(1, len(times)) if times[k] <= deadline] or [1]

        return max(
            candidates,
            key=lambda k: (covered[k] - covered[max(k - run_steps, 0)], k),
        )

    def _baseload(
        self, state: State, config: Config, now: datetime, times: list[datetime]
    ) -> pd.DataFrame:
        """The baseload at `times` from its habit (see BaseloadForecaster.habit),
        made once a day and kept in the state: its twelve weeks load in a
        second or so, and a day moves the median one week in eight."""

        baseload = BaseloadForecaster()
        today = to_local_time(now).date()
        habit = state.predictions.baseload_habit

        if habit is None or habit.day != today:
            p10, p50, p90 = baseload.habit(
                self.loader.load(
                    baseload.dataset(config),
                    now - timedelta(weeks=BaseloadForecaster.BAND_WEEKS, days=1),
                    now,
                )
            )
            habit = BaseloadHabit(day=today, p10_w=p10, p50_w=p50, p90_w=p90)
            self.state_manager.update_habit("baseload_habit", habit)

        return baseload.predict(
            (habit.p10_w, habit.p50_w, habit.p90_w), pd.DatetimeIndex(times)
        )

    def _tap_habit(
        self, state: State, config: Config, tap: TapForecaster, now: datetime
    ) -> tuple[pd.Series, float | None]:
        """The tap draws' habit (see TapForecaster.habit), made once a day and
        kept in the state: its four weeks of the boiler's minutes load in some
        1 s a plan."""

        today = to_local_time(now).date()
        habit = state.predictions.tap_habit

        if habit is None or habit.day != today:
            profile, heavy_day_w = tap.habit(
                self.loader.load(
                    tap.dataset(config),
                    now - timedelta(days=TapForecaster.PROFILE_DAYS + 1),
                    now,
                )
            )
            habit = TapHabit(
                day=today,
                draw_w_per_k=profile.tolist(),
                heavy_day_w=heavy_day_w,
            )
            self.state_manager.update_habit("tap_habit", habit)

        return pd.Series(habit.draw_w_per_k), habit.heavy_day_w

    def _room_heat_habit(
        self,
        state: State,
        config: Config,
        identifier: BuildingThermalIdentifier,
        now: datetime,
    ) -> pd.DataFrame:
        """The rooms' habit (see BuildingThermalIdentifier.room_heat_habit),
        made once a day and kept in the state: a median over three weeks moves
        little in a day (the forecast as good, within 0.002 K), and its weeks
        load in some 2 s against the zone's own 0.6 s. The rooms' rise in a
        run (see BuildingThermalIdentifier.room_run_rise) from the same weeks,
        into state.predictions.zone_habit."""

        today = to_local_time(now).date()
        habit = state.predictions.zone_habit

        # Made again where it lacks the rooms' rise (made before it was kept).
        if habit is None or habit.day != today or not habit.run_rise:
            weeks = self.loader.load(
                identifier.dataset(config),
                now - timedelta(days=identifier.HABIT_HISTORY_DAYS),
                now,
            )
            frame = identifier.room_heat_habit(weeks, now)
            prepared = identifier.prepare(weeks)
            loop_open, loop_flow_lpm = identifier.loop_flows(prepared, now)
            habit = ZoneHabit(
                day=today,
                living_w=frame["living"].tolist(),
                rest_w=frame["rest"].tolist(),
                run_rise=identifier.room_run_rise(prepared, now),
                loop_open=loop_open,
                loop_flow_lpm=loop_flow_lpm,
            )
            self.state_manager.update_habit("zone_habit", habit)
            state.predictions.zone_habit = habit

        return pd.DataFrame(
            {"living": habit.living_w, "rest": habit.rest_w},
            index=range(0, 24 * 60, 24 * 60 // len(habit.living_w)),
        )

    def _zone(
        self,
        state: State,
        config: Config,
        times: list[datetime],
        mpc_config: MPCConfig,
        cooling: bool | None = None,
        now: datetime | None = None,
    ) -> tuple[pd.DataFrame | None, tuple[dict, dict] | None]:
        """The zone as a second demand on the compressor: the filter's
        estimate of it (None without one), and what the plan needs of it
        (MPCInput fields) with the models it plans it with (MPCOptimizer
        arguments), or None where it cannot be planned.

        The zone is estimated here, where it is planned, rather than on every
        state update: one load and one filter pass give both the state a plan
        starts from and the measured temperature and thermal mass the dashboard
        draws. Two-node, because only that structure has the thermal mass a
        floor buffer is made of. Only planned with a comfort schedule for
        the mode (see BuildingConfig.comfort).

        Published for Home Assistant (see publish_zone). A floor run under way
        is the plan's to continue or stop (space_on_current, set by the
        caller): left out, the plan saw the compressor off, stopped a run it
        needed again half an hour later and paid a start for nothing. The tank
        then waits at most a step, as for any hand-over from the floor (see
        MPCOptimizer._add_space_heating).

        Cooled rather than heated while the heat pump is set to cool (see
        HeatPumpConfig.mode), from the cooling runs' own models, and only with
        dew points configured: the floor may not be cooled below them.
        """

        identifier = BuildingThermalIdentifier(
            self.state_manager.latitude, self.state_manager.longitude
        )
        identifier.load(self.models_path)

        if identifier.model is None:
            return None, None

        now = datetime.now(timezone.utc)
        # From the previous local midnight, so the dashboard has yesterday
        # beside today, and before that the filter's own warm-up.
        start = local_day_start(now, days=-1).astimezone(timezone.utc) - timedelta(
            hours=identifier.MASS_WARMUP_HOURS
        )
        end = times[-1] + timedelta(hours=mpc_config.step_hours)
        # Heat adds up, so the zone needs the load to expect, not the median the
        # energy plan works with: an evening's cooking falls on either side of
        # it, and the median picks the side without (backtest over 27 days,
        # 17-19 h: median 254 W against 578 W measured on average, the band's
        # expectation by Swanson's weights 551 W; all day 174 and 257 against
        # 257 W). The median alone where there is no band.
        predictions = state.predictions
        bands = [
            pd.Series({point.time: point.value for point in points}, dtype=float)
            for points in (
                predictions.baseload_p10,
                predictions.baseload,
                predictions.baseload_p90,
            )
        ]
        baseload = (
            sum(w * band for w, band in zip(SOLAR_SCENARIO_WEIGHTS, bands, strict=True))
            if all(not band.empty for band in bands)
            else bands[1]
        )

        # The sun on the facade ahead, from the array's own forecast (see
        # BuildingThermalIdentifier.forecast()).
        pv = pd.Series(
            {point.time: point.value for point in predictions.solar}, dtype=float
        )

        # The heat pump's own mode unless this run asks for one (see
        # OptimizeConfig.cooling).
        if cooling is None:
            cooling = self.state_manager.cooling(state, config)

        # How the heat pump runs the floor by itself in this mode - None until
        # its runs have shown it, and the plan may then choose the zone's heat
        # freely. Its pipes read the screed in the water as well (see
        # BuildingThermalIdentifier._water_reading).
        key = "cooling" if cooling else "heating"
        floor_circuit = FloorCircuitIdentifier(
            self.state_manager.latitude,
            self.state_manager.longitude,
            self.models_path,
            key=key,
        )
        floor_circuit.load(self.models_path)

        try:
            zone = identifier.trajectory(
                self.loader.load(identifier.dataset(config), start, end),
                now,
                baseload if not baseload.empty else None,
                pv if not pv.empty else None,
                self.loader.load(
                    identifier.habit_dataset(),
                    now - timedelta(days=identifier.HABIT_HISTORY_DAYS),
                    now,
                ),
                self._room_heat_habit(state, config, identifier, now),
                (
                    floor_circuit.model.pipe_ua_w_per_k
                    if floor_circuit.model is not None
                    else None
                ),
            )
        except ValueError as error:
            # Normal right after a restart with no measurements yet, or before
            # the weather forecast reaches past now.
            logger.warning("No zone estimate: %s", error)
            return None, None

        zone.index = pd.to_datetime(zone.index, utc=True)

        comfort = config.building.comfort(cooling)

        if comfort is None:
            return zone, None

        # The plan starts where the filter's estimate of the whole zone state
        # stands at its first step: both rooms and their slabs.
        planned = zone.reindex(pd.DatetimeIndex(times))
        state_names = list(identifier.state_names(identifier.model))

        if planned[state_names].iloc[0].isna().any():
            logger.info("Zone not planned: no zone estimate at %s", times[0])
            return zone, None

        dew_point_c = self._dew_point_forecast(state, times, mpc_config)

        if cooling and dew_point_c is None:
            logger.info(
                "Zone not planned: no dew point to keep the floor above while "
                "cooling (see building.dew_points)"
            )
            return zone, None

        def gain(column: str) -> tuple[float, ...]:
            return tuple(
                self.state_manager.extend_forecast(
                    zone[column].dropna().to_dict(), times, 0.0
                )
            )

        target_c, minimum_c, maximum_c = (
            tuple(
                point.value
                for point in self.state_manager.resolve_schedule(schedule, times)
            )
            for schedule in comfort
        )
        # Every loop, at the flow its runs ran at, without a run in the frame.
        loops = identifier.last_run_loops(zone)
        floor_open = ((loops[0], loops[1]),) * len(times) if loops else ()
        floor_flow = (loops[2],) * len(times) if loops else ()
        rest_minimum_c = config.building.rest_temperature

        # Set to the rest's minimum while a heating run is served (see
        # publish_zone), each room's own thermostat opens its loops while the
        # room is below it and closes them once it gets there: per step, the
        # plan takes the rooms open that are below it then - the room now,
        # risen as the previous plan had the rest rise since, in its runs at
        # the room's own rate (see _planned_rest_rise_k). Held at the room now,
        # rooms 0.01-0.4 K below it stayed open all run though they closed
        # within the hour, and tomorrow's rooms - cooled overnight below it -
        # closed because they are above it now (9 Nov 2026: 12.6 kWh planned
        # for tomorrow's floor against 17.9 with its upstairs loops open).
        if rest_minimum_c is not None and not cooling:
            rooms = config.building.rooms
            total_m2 = sum(room.area_m2 for room in rooms)
            habit = state.predictions.zone_habit
            run_rise = (
                habit.run_rise
                if habit is not None and len(habit.run_rise) == len(rooms)
                else [1.0] * len(rooms)
            )
            rest = np.zeros(len(times))

            for room, ratio in zip(rooms, run_rise, strict=True):
                if room.temperature == config.building.thermostat.temperature:
                    continue

                reading = self._last_reading(room.temperature, now)

                if reading is not None:
                    read_at, room_c = reading
                    rise_k = self._planned_rest_rise_k(
                        state.schedule.building.rest_temperatures,
                        state.schedule.building.on,
                        times,
                        read_at,
                        timedelta(hours=mpc_config.step_hours),
                        ratio,
                    )
                    rest += (room_c + rise_k < rest_minimum_c) * (
                        room.area_m2 / total_m2
                    )

            living = identifier.model.living_area_fraction

            # At the flow the runs had through as much open loop area, between
            # two such between them; the last run's without any.
            def flow_lpm(share: float) -> float | None:
                if habit is not None and habit.loop_open:
                    return float(
                        np.interp(living + share, habit.loop_open, habit.loop_flow_lpm)
                    )

                return loops[2] if loops else None

            floor_open = tuple((living, float(share)) for share in rest)
            floor_flow = tuple(flow_lpm(float(share)) for share in rest)
        inputs = dict(
            zone_state=tuple(float(c) for c in planned[state_names].iloc[0]),
            zone_target_temperature=target_c,
            zone_minimum_temperature=minimum_c,
            zone_maximum_temperature=maximum_c,
            # Extended where the weather forecast behind the estimate does not
            # reach; no gain assumed without any. The sun through the glazing
            # is not: as for the solar forecast, none where none is forecast.
            zone_internal_gain_w=gain("internal_gain_w"),
            zone_living_gain_w=tuple(planned["living_gain_w"].fillna(0.0)),
            zone_rest_gain_w=tuple(planned["rest_gain_w"].fillna(0.0)),
            zone_living_air_gain_w=gain("living_air_gain_w"),
            zone_living_sun_w=tuple(planned["living_sun_w"].fillna(0.0)),
            zone_rest_sun_w=tuple(planned["rest_sun_w"].fillna(0.0)),
            # Beyond the forecast the envelope a day earlier, else the nearest.
            zone_envelope_w_per_k=tuple(
                self.state_manager.extend_forecast(
                    zone["envelope_w_per_k"].dropna().to_dict(), times
                )
            ),
            zone_floor_open=floor_open,
            zone_floor_flow_lpm=floor_flow,
            zone_heat_may_rise=config.heat_pump.auto_adaptive,
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

        # The measurements start at local midnight, so any hot water among
        # them was today's.
        if cooling and config.building.dhw_after_cooling:
            inputs["zone_local_day"] = tuple(
                to_local_time(t).toordinal() for t in times
            )
            inputs["dhw_earlier_today"] = any(
                point.value == config.heat_pump.states.dhw
                for point in state.measurements.heat_pump.state
            )

        space_cop = HeatPumpCOPIdentifier(key=key)
        space_cop.load(path=self.models_path)

        return zone, (
            inputs,
            dict(
                building_model=identifier.model,
                floor_circuit_model=floor_circuit.model,
                space_cop_model=space_cop.model,
            ),
        )

    @staticmethod
    def _planned_rest_rise_k(
        planned: list[SeriesPoint],
        on: list[SeriesPoint],
        times: list[datetime],
        since: datetime,
        step: timedelta,
        run_rise: float = 1.0,
    ) -> np.ndarray:
        """How far a rest room rises from its reading at since to each of
        times (K) as the previous plan had the rest's thermostats rise
        (planned), the
        steps that plan heated the zone in (on) run_rise times as fast (see
        BuildingThermalIdentifier.room_run_rise): what a replan, made every
        few minutes, takes the room to do meanwhile. Held beyond that plan's
        end; zero throughout where it was made over a step before this one's
        start - from another state, or none at all."""

        if not planned or planned[0].time < times[0] - step:
            return np.zeros(len(times))

        seconds = np.array([point.time.timestamp() for point in planned])
        values = np.array([point.value for point in planned])
        # np.interp holds the last value beyond the end.
        level = np.interp([time.timestamp() for time in times], seconds, values)
        increments = np.diff(
            level, prepend=np.interp(since.timestamp(), seconds, values)
        )
        # Each increment over the step it ends: the first from since.
        heated = {point.time for point in on if point.value}
        spans = [times[0]] + times[:-1]
        rate = np.array([run_rise if time in heated else 1.0 for time in spans])

        return np.cumsum(rate * increments)

    def _dew_point_forecast(
        self, state: State, times: list[datetime], mpc_config: MPCConfig
    ) -> list[float] | None:
        """The indoor dew point at each step (deg C), None without one measured.

        From the one measured now, carried along the outdoor forecast by the
        identified moisture balance (see features.dew_point) - Open-Meteo's dew
        point. Held at the measured one without that model or forecast, as
        before there was one.
        Stored as a prediction, so the dashboard draws what the plan used.
        """

        measured = state.measurements.building.dew_point

        if not measured:
            return None

        now_c = float(measured[-1].value)
        identifier = DewPointIdentifier()
        identifier.load(self.models_path)
        outdoor_c = pd.Series(
            self.state_manager.extend_forecast(
                {p.time: p.value for p in state.forecast.open_meteo.dew_point}, times
            )
        )

        if identifier.model is None or outdoor_c.isna().all():
            forecast_c = [now_c] * len(times)
        else:
            forecast_c = indoor_dew_point_c(
                identifier.model,
                now_c,
                outdoor_c.to_numpy(),
                mpc_config.step_hours,
            ).tolist()

        self.state_manager.update_prediction(
            "dew_point", pd.Series(forecast_c, index=times)
        )

        return forecast_c

    @staticmethod
    def zone_running(
        heat_pump_state: list[SeriesPoint],
        compressor_frequency: list[SeriesPoint],
        mode_state: str,
        defrost: Sequence[SeriesPoint] = (),
    ) -> bool:
        """Whether a floor run is under way: the compressor running in the
        zone's mode now, so a plan that goes on with it makes no start. The
        state alone does not say so - it stays on heating at 0 Hz between
        runs. Without a frequency the state is all there is, as in
        dhw_timing. A stop in a defrost (its changes in `defrost`) is no end:
        the compressor comes up again after it in the same run, as in
        zone_run_hours (7 Nov 2026: 0 Hz in a defrost 16 minutes into a run,
        and the plan went over to hot water)."""

        if not heat_pump_state or heat_pump_state[-1].value != mode_state:
            return False

        if not compressor_frequency or compressor_frequency[-1].value > 0.0:
            return True

        return bool(
            Optimization.value_at(defrost, compressor_frequency[-1].time, False)
        )

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
    def zone_run_hours(
        heat_pump_state: list[SeriesPoint],
        compressor_frequency: list[SeriesPoint],
        mode_state: str,
        dhw_state: str,
        now: datetime,
        defrost: Sequence[SeriesPoint] = (),
    ) -> float:
        """How long the floor run under way has run since the compressor
        started it on the zone (hours, see MPCInput.space_elapsed_hours): from
        the change into the zone's mode, or from the compressor's last start
        within it - the state stays in that mode at 0 Hz between runs. 0 for a
        run the tank handed over, the compressor running on from hot water.

        A stop in a defrost (its changes in `defrost`) is no new start: the
        compressor comes up again after it in the same run, as in dhw_timing.
        Nor is a change from DHW at 0 Hz a hand-over - the booster finishing
        the tank or a DHW run that had ended - but a start from off, counted
        from the compressor's start."""

        since = Optimization.running_since(heat_pump_state, mode_state)

        if since is None:
            return 0.0

        before = [p for p in heat_pump_state if p.time < since]
        handed_over = bool(before) and before[-1].value == dhw_state
        frequency_before = [p for p in compressor_frequency if p.time < since]
        stops = [since] if frequency_before and frequency_before[-1].value <= 0 else []
        stops += [
            p.time
            for p in compressor_frequency
            if p.time >= since
            and p.value <= 0
            and not Optimization.value_at(defrost, p.time, False)
        ]

        if stops:
            starts = [
                p.time
                for p in compressor_frequency
                if p.time >= stops[-1] and p.value > 0
            ]

            if not starts:
                return 0.0

            since, handed_over = starts[0], False

        return 0.0 if handed_over else max((now - since).total_seconds() / 3600.0, 0.0)

    @staticmethod
    def zone_supply_c(
        planned_c: float,
        heat_w: float,
        return_c: float | None,
        flow_lpm: float | None,
        minimum_c: float | None,
        previous_c: float | None,
        cooling: bool,
    ) -> float:
        """The supply setpoint for a run under way (deg C), to the heat pump's
        half degree: heat_w the planned heat into the floor, negative while
        cooling.

        From the water the floor sends back, return + Q / (m_dot c_p): the
        supply that brings the planned heat into it, set on what the floor
        does rather than on the plan's estimate of a mass no sensor measures
        (0.5-1 K off, enough to put the heat pump under its least heat or a
        cooling supply under the dew point). The plan's own supply without a
        return to go on - a run's first quarter hour, while the loop still
        holds the still water it stood with. Never above an earlier setpoint of
        the same cooling run, nor below one of the same heating run: set past
        the water in the loop, the heat pump turns down past its least and
        stops. Never below the supply minimum (the dew point uninsulated pipes
        carry), which comes first - a run whose dew point rose past it is for
        the plan to stop.
        """

        setpoint_c = planned_c

        if return_c is not None and flow_lpm:
            water_w_per_k = flow_lpm / 60.0 * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K
            setpoint_c = return_c + heat_w / water_w_per_k

        if previous_c is not None:
            setpoint_c = (min if cooling else max)(setpoint_c, previous_c)

        # The nearest half degree, as for the hot water setpoint - but never
        # rounded under the minimum.
        rounded_c = math.floor(round(2 * setpoint_c, 2) + 0.5) / 2

        if minimum_c is not None:
            rounded_c = max(rounded_c, math.ceil(round(2 * minimum_c, 2)) / 2)

        return rounded_c

    # The resolution the zone's room setpoints are published at (K): half a
    # degree, the step thermostats take (Danfoss Icon: target_temp_step 0.5,
    # never another value in 60 days). Between steps the thermostat rounds,
    # and rounded down a run's setpoint could fall under the room.
    ZONE_SETPOINT_STEP_C = 0.5

    @classmethod
    def zone_setpoint_c(
        cls,
        schedule: tuple[int, ...],
        first: int,
        temperatures: tuple[float, ...],
        measured_c: float | None,
        limit_c: tuple[float, ...] | None,
        cooling: bool,
        tank: tuple[int, ...] = (),
    ) -> str:
        """The room setpoint (deg C) for the zone run starting at step first:
        the least step of ZONE_SETPOINT_STEP_C past the warmest (coolest)
        temperature the plan has the room at from that run's start to the next
        run's, and a whole step past measured_c, the room now; no further than
        limit_c where the run starts (the comfort band's edge, None for no
        limit). temperatures are at each step's start. The run is the
        compressor's: a tank block it hands over to and back (tank) does not
        end it, the floor goes on after it.

        A whole step past the room, so a room the sun warms faster than the
        plan has it stays below the setpoint until the next replan raises it:
        the thermostat reads the air, which the sun warms within minutes and
        which cools as fast once it is gone, while the plan sees the floor and
        the mass too (7 Nov 2026: +0.4 K in 25 minutes; three runs in Oct-Nov
        2026 ended at the thermostat, the room 0.03-0.05 K under its setpoint,
        five minutes before the plan's end)."""

        end = first

        while end + 1 < len(schedule) and (
            schedule[end + 1] or (end + 1 < len(tank) and tank[end + 1])
        ):
            end += 1

        following = next(
            (k for k in range(end + 1, len(schedule)) if schedule[k]), len(schedule)
        )
        window = temperatures[first : following + 1]
        step = cls.ZONE_SETPOINT_STEP_C

        # Strictly past the plan's: at the room's own temperature a thermostat
        # asks for nothing.
        if cooling:
            steps = math.ceil(round(min(window) / step, 6)) - 1

            if measured_c is not None:
                steps = min(steps, math.floor(round(measured_c / step, 6)) - 1)

            setpoint_c = steps * step
            setpoint_c = max(setpoint_c, limit_c[first]) if limit_c else setpoint_c
        else:
            steps = math.floor(round(max(window) / step, 6)) + 1

            if measured_c is not None:
                steps = max(steps, math.ceil(round(measured_c / step, 6)) + 1)

            setpoint_c = steps * step
            setpoint_c = min(setpoint_c, limit_c[first]) if limit_c else setpoint_c

        return f"{setpoint_c:.1f}"

    # The plan's heat is held: a new one only past this from the held one, at
    # the controller's target: near the plan's heats the controller moves
    # ~3-4 kW of heat per K of error (5 Nov 2026, 17.5 L/min, 8 degC), so a
    # 0.1 K step is ~0.4 kW. One step: at two a plan one step off was never
    # followed, ~1 kWh over a 3-hour run (11 Nov 2026). Successive plans' heats
    # differ by less (2.96-3.46 kW over 80 replayed solves, 10-11 Nov 2026);
    # a 30-minute hold on top changed the bias at 4 of them (24 against 20
    # steps), so there is none.
    ZONE_BIAS_DEADBAND_K = 0.1
    # The surplus the grid meter measures (its reading without the heat pump's
    # own draw) caps the heat - the same as holding the meter to the plan's
    # import: lowered once a shortfall lasted one decision interval (the plan
    # runs every 5 minutes; its best minute, so a kettle or a passing cloud
    # does not) and raised again once the surplus held for
    # ZONE_BIAS_RAISE_WINDOW (its worst minute), both at once. Over 128
    # sunny days (Jun-Nov 2026, 1-minute P1 and heat pump power, the plan's
    # heat then held 30 minutes) that took 5.8 bias steps a day, 0.6 of them
    # back up within half an hour, the import and export it costs within
    # 0.2 ct a day; raised at once it took 13.5 at the same.
    ZONE_BIAS_LOWER_WINDOW = timedelta(minutes=5)
    ZONE_BIAS_RAISE_WINDOW = timedelta(minutes=10)

    def _zone_bias(
        self,
        config: Config,
        data: MPCInput,
        result: MPCResult,
        state: State,
        k: int,
        on: bool,
        since: datetime | None,
        now: datetime,
        run_setpoint_c: float | None = None,
        least_heat_w: float = 0.0,
        min_supply_c: float = 0.0,
    ) -> tuple[float, str]:
        """The bias to publish now, and why (for the log): for a heating run
        under way or handed over (step k its first), the bias at which the
        controller asks for the plan's mean heat over its next hour; outside
        one, or without the readings, the bias at which it asks for its
        least, at most 0; 0 - the controller left to its setpoint - while
        cooling or without the room or its setpoint.

        What is held within a run under way since `since` is the heat, not
        the bias: the bias follows the room and the setpoint at once, so the
        controller's error - and its heat - stays the plan's when the sun
        warms the room or the blueprint raises the setpoint. The plan's heat
        itself is held within the deadband; a run's first quarter hour and
        the 0 after it are not held.

        Against the thermostat's setpoint, which the controller follows -
        before the heat pump heats, run_setpoint_c, the room setpoint the
        blueprint gives the run as it starts: the one measured is still the
        one before (6 Nov 2026: 15 degC, the bias clamped at 2.5 K).

        The heat is at most what the measured solar surplus plus the plan's
        own expected import buys at the plan's COP: the plan chose that
        import, more is the forecast's error (sun or load). That cap moves
        at once, see ZONE_BIAS_LOWER_WINDOW."""

        outdoor = config.heat_pump.outdoor_temperature
        assert outdoor is not None
        building = state.measurements.building
        measured = state.measurements.heat_pump

        if data.zone_cooling:
            return 0.0, " | cooling, thermostat follows its setpoint"

        setpoint_c = run_setpoint_c if on and since is None else None

        if setpoint_c is None and building.setpoint:
            setpoint_c = building.setpoint[-1].value

        if setpoint_c is None or not building.zone_temperature:
            return 0.0, " | no room temperature or setpoint"

        room_c = building.zone_temperature[-1].value
        # Outside a run, or within one without the readings, the least, at
        # most 0. Not 0 with the setpoint above the room - still the run's,
        # left to it the controller heats on hard until the thermostat is
        # lowered (11 Nov 2026: 0 at the plan's end, the living room still at
        # 20.5, supply 28 -> 30.8 degC). Never above 0: at or below the room
        # the setpoint alone asks for the least, and a positive bias would add
        # to a thermostat raised by hand.
        least_k = min(ecodan.least_bias_k(room_c, setpoint_c), 0.0)
        readings = f"room {room_c:.1f}, setpoint {setpoint_c:.1f}"
        least = (
            " = minimum" if least_k == ecodan.least_bias_k(room_c, setpoint_c) else ""
        )

        if not on or not result.space_heat_w:
            return least_k, f"{least} | no heating run | {readings}"

        schedule = result.space_schedule
        step_hours = MPCConfig().step_hours
        step = timedelta(hours=step_hours)
        steps_per_hour = round(1.0 / step_hours)
        steps = [
            j
            for j in range(k, min(k + steps_per_hour, len(schedule)))
            if schedule[j] == 1
        ] or [k]
        plan_w = float(np.mean([result.space_heat_w[j] for j in steps]))

        # The flow measured once the heat pump has heated this run for a
        # quarter hour, as zone_supply_c takes it: before, the quarter's mean
        # holds the hot water's circuit or the pump still off. Till then the
        # plan's, the floor's last (or this) run's.
        settled = since is not None and now - since >= step
        flow_lpm = (
            measured.flow[-1].value
            if settled and measured.flow and measured.flow[-1].value > 0
            else data.zone_floor_flow_lpm[k]
            if data.zone_floor_flow_lpm
            else None
        )
        outdoor_c = self._latest(outdoor, now)

        if not (flow_lpm and outdoor_c is not None and data.zone_maximum_temperature):
            return least_k, f"{least} | no flow or outdoor reading | {readings}"

        readings += f", flow {flow_lpm:.0f} L/min, outdoor {outdoor_c:.1f} °C"

        # The controller holds its supply at its floor (min_supply_c) however
        # low the bias asks, so over a return below it the least bias already
        # gives the floor's heat (12 Nov 2026: supply 24.0-24.5 over a 20.5
        # degC return at -0.8 and -1.0 K alike, which asked 23.9 and 23.3).
        # From the run's settled return, as the flow above.
        if settled and min_supply_c and measured.return_temperature:
            least_heat_w = max(
                least_heat_w,
                ecodan.water_w_per_k(flow_lpm)
                * (min_supply_c - measured.return_temperature[-1].value),
            )

        # Logged as the heat pump's draw at the plan's COP, to set beside the
        # grid meter and its power sensor; the heat where the plan has none.
        cop = (
            result.space_heat_w[k] / result.space_electrical_w[k]
            if result.space_electrical_w and result.space_electrical_w[k] > 0
            else None
        )

        def kw(heat_w: float) -> str:
            return (
                f"{heat_w / cop / 1000:.2f} kW"
                if cop
                else f"{heat_w / 1000:.1f} kW heat"
            )

        if cop:
            readings += f", COP {cop:.1f}"

        def bias_k(heat_w: float) -> float:
            return ecodan.setpoint_bias_k(
                heat_w=heat_w,
                flow_lpm=flow_lpm,
                outdoor_c=outdoor_c,
                room_c=room_c,
                setpoint_c=setpoint_c,
                # Within a run under way the plan's heat, wherever the sun
                # has taken the air (see zone_setpoint_c).
                maximum_c=(
                    data.zone_maximum_temperature[k] if since is None else math.inf
                ),
                least_heat_w=least_heat_w,
            )

        previous = state.schedule.building.bias_heat
        # Held only from a heat set once the run has settled - its first
        # quarter hour, as the flow above: before it the compressor is not yet
        # up and the plan may not count the run as under way.
        held = (
            previous
            if previous is not None
            and since is not None
            and previous.time >= since + step
            else None
        )
        heat_w = plan_w
        plan_kw = f"plan {kw(plan_w)}"
        why = "plan"

        if (
            held is not None
            and abs(bias_k(plan_w) - bias_k(held.value))
            < self.ZONE_BIAS_DEADBAND_K - 1e-9
        ):
            heat_w = held.value

            if heat_w != plan_w:
                why = f"held, {plan_kw} too close (< {self.ZONE_BIAS_DEADBAND_K:.1f} K)"

        cap_w = state.schedule.building.bias_cap_w if held is not None else None
        surplus = self._surplus_w(config, now)
        # What set the cap: kept until the sun holds, or this surplus.
        sun_note = " (kept until the sun holds 10 min)"

        if surplus is not None and result.grid_step_kwh and cop:
            # Export the plan expected is no budget: what the heat pump draws
            # within the measured surplus imports nothing.
            planned_import_w = max(
                float(
                    np.mean(
                        [
                            result.grid_step_kwh[j]
                            * 1000.0
                            / (
                                data.first_step_hours
                                if j == 0 and data.first_step_hours
                                else step_hours
                            )
                            for j in steps
                        ]
                    )
                ),
                0.0,
            )
            # At least 0, as the plan's (see optimizer.surplus_scenarios): its
            # import is the heat pump's own draw beyond the sun, the rest of
            # the house's draw beyond it is not the heat pump's to make up.
            # Below 0 it took that draw off the heat at night (a 300 W house:
            # a 4.5 kW plan capped at 3 kW).
            best_w, worst_w = (max(surplus_w, 0.0) for surplus_w in surplus)
            lowered_w = (best_w + planned_import_w) * cop
            raised_w = (worst_w + planned_import_w) * cop
            capped_w = heat_w if cap_w is None else min(heat_w, cap_w)

            def sun(surplus_w: float) -> str:
                return (
                    f" (surplus {surplus_w / 1000:.2f} + planned grid "
                    f"{planned_import_w / 1000:.2f})"
                )

            if bias_k(lowered_w) < bias_k(capped_w):
                cap_w = lowered_w
                sun_note = sun(best_w)
            # The cap lifted at once as far as the surplus held covers.
            elif (
                cap_w is not None
                and bias_k(min(plan_w, raised_w))
                >= bias_k(capped_w) + self.ZONE_BIAS_DEADBAND_K - 1e-9
            ):
                heat_w = plan_w
                cap_w = raised_w
                why = "plan"
                sun_note = sun(worst_w)
        else:
            cap_w = None

        if cap_w is not None and cap_w >= heat_w:
            cap_w = None

        if cap_w is not None:
            why = f"{plan_kw}, sun allows {kw(cap_w)}{sun_note}"

        # Its time is when the plan's heat last moved.
        point = (
            previous
            if previous is not None and previous.value == heat_w
            else SeriesPoint(time=now, value=heat_w)
        )

        if point is not previous or cap_w != state.schedule.building.bias_cap_w:
            self.state_manager.update_zone_bias(point, cap_w)

        asked_w = heat_w if cap_w is None else cap_w
        bias = bias_k(asked_w)

        # Less than the controller can ask for: at the least bias it gives the
        # base rise, or the compressor's lowest speed where that is more (see
        # ecodan.setpoint_bias_k) - more than the heat asked.
        if bias == ecodan.least_bias_k(room_c, setpoint_c):
            given_w = max(
                ecodan.MIN_RISE_K * ecodan.water_w_per_k(flow_lpm), least_heat_w
            )
            asked = f"minimum (~{kw(given_w)})"
            why = plan_kw if why == "plan" else why
        else:
            asked = kw(asked_w)

        return bias, f" = {asked} | {why} | {readings}"

    def publish_zone(
        self,
        result: MPCResult,
        times: list[datetime],
        state: State,
        config: Config,
        data: MPCInput | None,
        heat_pump_state: list[SeriesPoint],
        now: datetime,
        least_heat_w: float = 0.0,
        min_supply_c: float = 0.0,
    ) -> None:
        """Writes the plan's floor decision to Home Assistant, as publish_dhw
        does the hot water's: on/off for the quarter hour running now, the
        start of the run under way or the next one planned, the room setpoint
        the thermostat is to have now - the run's while one is under way (see
        zone_setpoint_c), the comfort band's edge between runs - and the
        supply setpoint while a run is under way (see zone_supply_c).

        All four 'unknown' where the zone cannot be planned at all: nothing
        then for an automation to act on.
        The supply 'unknown' outside a run too.
        """

        status = start = setpoint = room = "unknown"
        # The rest's thermostats' setpoint (see
        # BuildingConfig.rest_temperature), only with that minimum set: the
        # minimum while a heating run is served, the idle setpoint otherwise -
        # also where the zone cannot be planned.
        minimum_c = config.building.rest_temperature
        rest_c = config.building.rest_idle_temperature
        published = None
        bias: float | None = None
        bias_reason = " | zone not planned"

        if data is not None and result.space_schedule:
            schedule = result.space_schedule
            running = schedule[0] == 1
            states = config.heat_pump.states
            since = self.running_since(
                heat_pump_state,
                states.cooling if data.zone_cooling else states.heating,
            )
            first = next((k for k, on in enumerate(schedule) if on), None)
            # The tank's block under way and the zone's run planned straight
            # after it: on from the tank's start, so the heat pump goes from
            # the hot water straight on to the floor - the hot water comes
            # first, it is not cut short for heat (19 hand-overs 18 Oct-6 Nov
            # 2026 with the thermostat asking throughout, runs to 74 minutes).
            # Off until the tank's last quarter hour, a tank done sooner than
            # planned stopped the heat pump and started it again for the zone,
            # a start the plan does not make: it counts the hand-over as one
            # run. Heating only: the heat pump stops to reverse between cooling
            # and the tank anyway.
            tank = result.schedule
            tank_end = next((j for j, on in enumerate(tank) if not on), len(tank))
            hand_over = (
                not running
                and not data.zone_cooling
                and bool(tank)
                and tank[0] == 1
                and tank_end < len(schedule)
                and schedule[tank_end] == 1
            )
            # The step whose supply the run is given: its first, after the
            # tank's when handed over. Handed over the run is set up - its
            # setpoints and bias - but on only from its own start, which the
            # status and start say.
            k = tank_end if hand_over else 0
            serving = running or hand_over
            status = "on" if running else "off"

            if minimum_c is not None and serving and not data.zone_cooling:
                rest_c = minimum_c

            if first is not None:
                start = (
                    since if running and since is not None else times[first]
                ).isoformat()

            # The room setpoint the thermostat is given for that run, which
            # asks the heat pump for heat while the room is below it: what the
            # run warms the room to in the plan, up to the next run - after it
            # too, as the screed gives its heat to the room for hours - and,
            # for a run under way, above the room as measured. Without that a
            # run that heats the floor for later, the room already as warm as
            # the plan has it, asked for nothing (real plan, 25 Oct 2026: 19.8
            # degC against a room at 19.82). The least published step past
            # either, never past the comfort band: status says when it ends.
            # Between runs the comfort band's edge now, so the thermostat asks
            # for heat (cold) only where the room passed what the plan allows:
            # left at a run's setpoint the heat pump ran on 20-75 minutes past
            # the plan's end (3-5 Nov 2026). Inside the band, on its step.
            band = (
                data.zone_maximum_temperature
                if data.zone_cooling
                else data.zone_minimum_temperature
            )

            if serving and first is not None and result.zone_temperatures:
                room = self.zone_setpoint_c(
                    schedule,
                    first,
                    result.zone_temperatures,
                    (
                        state.measurements.building.zone_temperature[-1].value
                        if state.measurements.building.zone_temperature
                        else None
                    ),
                    # Within a run under way the plan ends it (status), not
                    # the comfort band at the thermostat: the sun takes the air
                    # past the band's edge, not the floor the plan charges.
                    (
                        None
                        if running and since is not None
                        else data.zone_minimum_temperature
                        if data.zone_cooling
                        else data.zone_maximum_temperature
                    ),
                    data.zone_cooling,
                    tank,
                )
                # Held within a run under way: never short of the thermostat's
                # setpoint, which the blueprint only raises within a run - a
                # replan lowering it changes nothing there (7 Nov 2026: 20.0
                # and 20.5 in turn, the thermostat at 21).
                measured = state.measurements.building.setpoint

                if running and since is not None and measured:
                    held = (min if data.zone_cooling else max)(
                        float(room), measured[-1].value
                    )
                    room = f"{held:.1f}"
            elif not serving and band:
                step = self.ZONE_SETPOINT_STEP_C
                within = math.ceil if data.zone_cooling else math.floor
                room = f"{within(round(band[0] / step, 6)) * step:.1f}"

            if config.heat_pump.auto_adaptive:
                bias, bias_reason = self._zone_bias(
                    config,
                    data,
                    result,
                    state,
                    k,
                    serving,
                    since if running else None,
                    now,
                    float(room) if room != "unknown" else None,
                    least_heat_w,
                    min_supply_c,
                )

            if serving and result.space_supply_c:
                measured = state.measurements.heat_pump
                step = timedelta(hours=MPCConfig().step_hours)
                settled = since is not None and now - since >= step
                previous = state.schedule.building.supply_setpoint
                published = self.zone_supply_c(
                    planned_c=result.space_supply_c[k],
                    heat_w=result.space_heat_w[k],
                    return_c=(
                        measured.return_temperature[-1].value
                        if settled and measured.return_temperature
                        else None
                    ),
                    flow_lpm=(
                        measured.flow[-1].value if settled and measured.flow else None
                    ),
                    minimum_c=(
                        data.zone_supply_minimum_c[k]
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
                    cooling=data.zone_cooling,
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
            room,
            {
                "friendly_name": "Home Optimizer zone setpoint",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
            },
        )

        if minimum_c is not None:
            self.home_assistant.set_state(
                self.ZONE_REST_SETPOINT_ENTITY,
                f"{rest_c:.1f}",
                {
                    "friendly_name": "Home Optimizer zone rest setpoint",
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                },
            )
        self.home_assistant.set_state(
            self.ZONE_SUPPLY_ENTITY,
            setpoint,
            {
                "friendly_name": "Home Optimizer zone supply",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
            },
        )

        if config.heat_pump.auto_adaptive:
            text = f"{bias:.1f}" if bias is not None else "unknown"
            self.home_assistant.set_state(
                self.ZONE_BIAS_ENTITY,
                text,
                {
                    "friendly_name": "Home Optimizer zone bias",
                    "unit_of_measurement": "°C",
                },
            )
            # After the sensors, before the plan's runs (see log_day_costs).
            logger.info("Zone bias %s K%s", text, bias_reason)

    def _dhw_setpoint_c(self, config: Config, now: datetime) -> float | None:
        """The heat pump's DHW setpoint now (deg C), None without a reading."""

        return self._latest(config.heat_pump.boiler.setpoint, now)

    def _last_reading(
        self, reference: SensorReference, now: datetime
    ) -> tuple[datetime, float] | None:
        """A room thermometer's last reading in the hour before now and when
        it was taken, None without: the rooms' thermostats report every 30
        minutes, so a reading may be half an hour old - some 0.07 K of a
        run's rise."""

        frame = self.loader.load(
            DatasetBuilder()
            .timeseries("value", reference, aggregation="last", interval="1m")
            .build(),
            now - timedelta(hours=1),
            now,
        )

        if frame.empty or "value" not in frame:
            return None

        readings = frame.assign(
            value=pd.to_numeric(frame["value"], errors="coerce")
        ).dropna(subset=["value"])

        if readings.empty:
            return None

        last = readings.iloc[-1]

        return pd.Timestamp(last["time"]).to_pydatetime(), float(last["value"])

    def _latest(self, reference: SensorReference, now: datetime) -> float | None:
        """A sensor's last reading in the quarter hour before now, None without."""

        frame = self.loader.load(
            DatasetBuilder()
            .timeseries(
                "value", reference, aggregation="last", interval="15m", fill="previous"
            )
            .build(),
            now - timedelta(minutes=15),
            now,
        )
        readings = pd.to_numeric(frame.get("value"), errors="coerce").dropna()

        return float(readings.iloc[-1]) if not readings.empty else None

    def _surplus_w(self, config: Config, now: datetime) -> tuple[float, float] | None:
        """The solar surplus the heat pump had (W): its own draw less the grid
        meter's, per minute - solar less the rest of the house - at its best
        minute of ZONE_BIAS_LOWER_WINDOW and its worst of
        ZONE_BIAS_RAISE_WINDOW. None without readings."""

        frame = self.loader.load(
            DatasetBuilder()
            .timeseries(
                "grid", config.grid, aggregation="mean", interval="1m", fill="previous"
            )
            .timeseries(
                "heat_pump",
                config.heat_pump.power,
                aggregation="mean",
                interval="1m",
                fill="previous",
            )
            .build(),
            now - self.ZONE_BIAS_RAISE_WINDOW,
            now,
        )
        surplus = pd.to_numeric(
            frame.get("heat_pump"), errors="coerce"
        ) - pd.to_numeric(frame.get("grid"), errors="coerce")
        recent = surplus[
            pd.to_datetime(frame["time"], utc=True) >= now - self.ZONE_BIAS_LOWER_WINDOW
        ].dropna()

        if recent.empty:
            return None

        return float(recent.max()), float(surplus.min())

    @staticmethod
    def tank_level(
        tank_c: float,
        mains_c: float,
        setpoint_c: float | None,
        volume_l: float,
        drawn_today_j: float,
    ) -> tuple[float | None, dict[str, float]]:
        """What the tank holds, for its SoC sensor: the share of a tank at its
        setpoint (%, at most 100; None without a setpoint above the mains),
        and as attributes the heat above the mains water (kWh) - what tapping
        can take out, every litre replaced from the mains - and the heat drawn
        today (kWh). Heat, not litres: litres of mixed tap water rest on a tap
        temperature assumed, and the tank's own water leaves at a temperature
        that falls as it empties."""

        heat_capacity_j_per_k = RHO_WATER_KG_PER_L * volume_l * CP_WATER_J_PER_KG_K
        above_mains_k = max(tank_c - mains_c, 0.0)
        soc = (
            round(min(100.0 * above_mains_k / (setpoint_c - mains_c), 100.0), 1)
            if setpoint_c is not None and setpoint_c > mains_c
            else None
        )

        return soc, {
            "available_kwh": round(above_mains_k * heat_capacity_j_per_k / 3.6e6, 2),
            "drawn_today_kwh": round(drawn_today_j / 3.6e6, 2),
        }

    def publish_dhw(
        self,
        result: MPCResult,
        times: list[datetime],
        thermal_model: BoilerThermalModel | None = None,
        tank_level: tuple[float | None, dict[str, float]] | None = None,
    ) -> None:
        """Writes the plan's hot water decision to Home Assistant: on/off for the
        quarter hour running now, and the start and SWW setpoint of the next
        planned run (the current one if it is heating now, 'unknown' without
        any); and the tank's level (see tank_level()) where given.

        The heat pump heats until its setpoint and stops by itself; the heat
        left in the plate exchanger and the loops then still mixes through, and the
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

        if tank_level is not None:
            soc, attributes = tank_level
            self.home_assistant.set_state(
                self.DHW_SOC_ENTITY,
                "unknown" if soc is None else str(soc),
                {
                    "friendly_name": "Home Optimizer DHW SoC",
                    "unit_of_measurement": "%",
                    "state_class": "measurement",
                    **attributes,
                },
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
