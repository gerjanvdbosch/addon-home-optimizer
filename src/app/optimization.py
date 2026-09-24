import dataclasses
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.state import StateManager
from domain.config import Config
from domain.jobs import OptimizeConfig
from domain.mpc import MPCConfig, MPCInput, MPCResult
from domain.state import State
from domain.time import local_day_start
from features.boiler import BoilerThermalIdentifier
from features.building import BuildingThermalIdentifier
from features.cop import HeatPumpCOPIdentifier
from features.dataset import DatasetLoader
from features.optimizer import MPCOptimizer
from features.space_heating import SpaceHeatingIdentifier
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
        heat_pump_state = state.measurements.heat_pump.state
        boiler_on_current = bool(heat_pump_state) and (
            heat_pump_state[-1].value == dhw_state
        )

        # From the first quarter hour of the trailing run of COMPRESSOR
        # readings. The operating state alone is not enough: it stays on DHW
        # while the resistive booster finishes the tank, and the booster runs
        # with the compressor off (see BoilerThermalIdentifier.booster_active),
        # so counting those quarter hours would credit the minimum runtime with
        # time the compressor did not run. The frequency reports 0 Hz exactly
        # then, which is the compressor's own account of itself.
        #
        # Each reading is the state at its quarter hour's start, so the run may
        # have begun up to a quarter hour earlier: the elapsed time errs short,
        # and the run in progress is protected that much longer rather than too
        # briefly.
        frequency = {
            point.time: point.value
            for point in state.measurements.heat_pump.compressor_frequency
        }

        def compressor_running(point) -> bool:
            if point.value != dhw_state:
                return False

            # Without a frequency reading the state is all there is, and
            # assuming the compressor ran keeps a real run protected.
            return frequency.get(point.time, 1.0) > 0.0

        compressor_elapsed_hours = 0.0

        if boiler_on_current:
            run_start = heat_pump_state[-1].time

            for point in reversed(heat_pump_state):
                if not compressor_running(point):
                    break
                run_start = point.time

            compressor_elapsed_hours = max(
                0.0, (datetime.now(timezone.utc) - run_start).total_seconds() / 3600.0
            )

        # The other way round for the pause between runs (see
        # MPCConfig.boiler_min_off_steps): how long ago the last run ended. The
        # first reading that is not DHW is the earliest the run can have ended,
        # so this too errs short and the pause is kept rather than cut.
        idle_elapsed_hours = 0.0

        if not boiler_on_current:
            idle_start = heat_pump_state[-1].time if heat_pump_state else None

            for point in reversed(heat_pump_state):
                if point.value == dhw_state:
                    break
                idle_start = point.time

            idle_elapsed_hours = (
                max(
                    0.0,
                    (datetime.now(timezone.utc) - idle_start).total_seconds() / 3600.0,
                )
                if idle_start is not None
                else 0.0
            )

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
            target_temperature_top=target_temps,
            tap_forecast_w=tap_forecast,
            outdoor_temperature_forecast=outdoor_temperature_forecast,
            solar_p10_w=solar_p10,
            solar_p90_w=solar_p90,
            compressor_elapsed_hours=compressor_elapsed_hours,
            idle_elapsed_hours=idle_elapsed_hours,
            baseload_forecast_w=tuple(
                self.state_manager.baseload_forecast(
                    state, forecast_times, datetime.now(timezone.utc)
                )
            ),
        )

        optimizer = MPCOptimizer(
            thermal_model=thermal_model, config=mpc_config, cop_model=cop_model
        )
        result = optimizer.solve(data)

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
        )

        self.publish_dhw(result, forecast_times)
        self._plan_space_heating(optimizer, data, state, config, forecast_times)

    def _plan_space_heating(
        self,
        optimizer: MPCOptimizer,
        data: MPCInput,
        state: State,
        config: Config,
        times: list[datetime],
    ) -> None:
        """The building's side of a plan: the zone's state, and a second plan
        with the zone as a second demand on the compressor.

        The zone is estimated here, where it is planned, rather than on every
        state update: one load and one filter pass give both the state a plan
        starts from and the measured temperature and thermal mass the dashboard
        draws. Two-node, because only that structure has the thermal mass a
        floor buffer is made of.

        The plan itself is a shadow: stored for the dashboard and never acted
        on. Before a plan may drive the thermostats it has to be seen to make
        sense against what they actually do, over a heating season. It is
        solved apart from the hot water plan acted on above, so that plan stays
        exactly what it was - sharing the compressor, the zone would otherwise
        move it. Only planned with a comfort ceiling configured.
        """

        identifier = BuildingThermalIdentifier(
            self.state_manager.latitude, self.state_manager.longitude
        )
        identifier.load(self.models_path)

        if identifier.model is None:
            return

        now = datetime.now(timezone.utc)
        # From the previous local midnight, so the dashboard has yesterday
        # beside today, and before that the filter's own warm-up.
        start = local_day_start(now, days=-1).astimezone(timezone.utc) - timedelta(
            hours=identifier.MASS_WARMUP_HOURS
        )
        end = times[-1] + timedelta(hours=optimizer.config.step_hours)
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
            return

        zone.index = pd.to_datetime(zone.index, utc=True)
        self.state_manager.update_zone(
            temperature=zone["measured"].dropna(), thermal_mass=zone["mass"]
        )

        maximum = config.building.maximum_temperature

        if maximum is None:
            return

        # The plan starts where the filter's estimate of the whole zone state
        # stands at its first step: the air node, and the mass.
        planned = zone.reindex(pd.DatetimeIndex(times))

        if planned[["air", "mass"]].iloc[0].isna().any():
            logger.info(
                "Shadow space-heating plan skipped: no zone estimate at %s", times[0]
            )
            return

        heat_pump_state = state.measurements.heat_pump.state

        zone_data = dataclasses.replace(
            data,
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
            space_on_current=bool(heat_pump_state)
            and heat_pump_state[-1].value == config.heat_pump.states.heating,
        )

        # How the heat pump runs the floor by itself - None until heating runs
        # have shown it, and the plan may then choose the zone's heat freely.
        space_heating = SpaceHeatingIdentifier(
            self.state_manager.latitude, self.state_manager.longitude, self.models_path
        )
        space_heating.load(self.models_path)
        heating_cop = HeatPumpCOPIdentifier(key="heating")
        heating_cop.load(path=self.models_path)

        try:
            result = MPCOptimizer(
                thermal_model=optimizer.thermal_model,
                config=optimizer.config,
                cop_model=optimizer.cop_model,
                building_model=identifier.model,
                space_heating_model=space_heating.model,
                heating_cop_model=heating_cop.model,
            ).solve(zone_data)
        except RuntimeError as error:
            # Nothing acts on this plan, so a failure here must not take the
            # hot water plan above down with it.
            logger.warning("Shadow space-heating plan failed: %s", error)
            return

        logger.info(
            "Shadow space-heating plan: %.1f kWh into the zone over the horizon",
            sum(result.space_heat_w) * optimizer.config.step_hours / 1000.0,
        )

        self.state_manager.update_building_schedule(
            heat_w=result.space_heat_w,
            temperatures=result.zone_temperatures,
            times=times,
        )

    def publish_dhw(self, result: MPCResult, times: list[datetime]) -> None:
        """Writes the plan's hot water decision to Home Assistant: on/off for the
        quarter hour running now, and the start and SWW setpoint of the next
        planned run (the current one if it is heating now, 'unknown' without
        any).

        The heat pump heats until its setpoint and stops by itself, so the
        setpoint is the run's planned end temperature, rounded up to the heat
        pump's half degree. The heat left in the coil and loop then still flows
        into the tank, which settles above the setpoint (real data: 0.9-2.8 K,
        median 1.75 K). That is deliberately not subtracted: it is the margin
        that absorbs a tap or loss forecast that turns out worse, where
        planning the run to end exactly on target made every replan that saw a
        fraction of a degree short start another run.
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
            next_start = times[start].isoformat()
            setpoint = str(math.ceil(round(2 * end_temperature, 2)) / 2)

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
