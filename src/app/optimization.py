import dataclasses
import logging
import math
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
    # The floor's, published the same way while the plan drives it (see
    # publish_zone).
    ZONE_STATUS_ENTITY = "binary_sensor.home_optimizer_zone_status"
    ZONE_START_ENTITY = "sensor.home_optimizer_zone_start"
    ZONE_SUPPLY_ENTITY = "sensor.home_optimizer_zone_supply"
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
        run_start_stratification_k = run_start_ambient_c = None
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
            ambient = {p.time: p.value for p in boiler.ambient_temperature}

            if before_top and before_bottom:
                run_start_temp_top = before_top[-1].value
                run_start_temp_bottom = before_bottom[-1].value
                run_start_stratification_k = stratification.get(before_top[-1].time)
                run_start_ambient_c = ambient.get(before_top[-1].time)

        # The heat the run under way has put in by now, as the plan starts
        # from it.
        run_heat_j = (
            self._run_heat_j(config, run_start, [now])
            if run_start is not None
            else None
        )

        # The tap draws: the past weeks' own profile, held to what today has
        # drawn up to this plan (see TapForecaster). Kept in the state to be
        # looked at; no draws assumed beyond it, the assumption made before
        # it existed.
        tap = TapForecaster(self.models_path)
        tap_w = tap.predict(
            # From a reading before midnight: the boiler's prepare() drops the
            # first, which has no interval of its own.
            self.loader.load(
                tap.dataset(config),
                local_day_start(now).astimezone(timezone.utc)
                - 2 * pd.Timedelta(BoilerThermalIdentifier.SAMPLE_INTERVAL),
                now,
            ),
            steps=len(forecast_times),
            habit=self._tap_habit(state, config, tap, now),
        )
        self.state_manager.update_prediction("tap", tap_w)
        tap_forecast = extended(
            [SeriesPoint(time=t, value=float(w)) for t, w in tap_w.items()], 0.0
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
            mains_water_temperature=mains_water_temperature_c(
                to_local_time(now).timetuple().tm_yday
            ),
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
            states = config.heat_pump.states
            planned = dataclasses.replace(
                data,
                **inputs,
                space_on_current=self.zone_running(
                    *heat_pump_changes,
                    states.cooling if inputs["zone_cooling"] else states.heating,
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

        self.state_manager.update_schedule(
            schedule=result.schedule,
            temperatures=result.temperatures,
            power_w=result.electrical_power_w,
            times=forecast_times,
            heat_w=result.heat_w,
            now=now,
        )

        self.publish_dhw(result, forecast_times, thermal_model)
        self.log_dhw_plan(result, forecast_times, optimizer, planned)

        if optimize_config.explain:
            self.explain_dhw_plan(optimizer, planned, result, forecast_times)

        if zone is not None:
            # The supply of the first planned run: the setpoint the plan would
            # give the heat pump.
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
            on=result.space_schedule,
            thermal_mass=result.zone_mass_temperatures,
            screed=result.zone_screed_temperatures,
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

    def _run_heat_j(
        self, config: Config, run_start: datetime, times: list[datetime]
    ) -> list[float] | None:
        """The heat the DHW run under way had carried to the tank by each of
        `times` (J): its calorimetry as the boiler model is identified on (see
        BoilerThermalIdentifier.prepare), each minute's reading held until the
        next. None where it shows none - no flow reported yet, say - and the
        plan's own heat input stands in. Taken from the minutes, not the
        state's quarter hours: the heat is flow times temperature rise, and
        the product of two quarter-hour means is not the quarter's heat."""

        boiler = BoilerThermalIdentifier()
        # From one reading before the run: prepare() drops the first, which has
        # no interval of its own.
        df = boiler.prepare(
            self.loader.load(
                boiler.dataset(config),
                run_start - 2 * pd.Timedelta(boiler.SAMPLE_INTERVAL),
                times[-1],
            )
        )
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

    @classmethod
    def log_dhw_plan(
        cls,
        result: MPCResult,
        times: list[datetime],
        optimizer: MPCOptimizer,
        data: MPCInput,
    ) -> None:
        logger.info("DHW plan: %s", cls.dhw_summary(result, times, optimizer, data))

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
        load in some 2 s against the zone's own 0.6 s."""

        today = to_local_time(now).date()
        habit = state.predictions.zone_habit

        if habit is None or habit.day != today:
            frame = identifier.room_heat_habit(
                self.loader.load(
                    identifier.dataset(config),
                    now - timedelta(days=identifier.HABIT_HISTORY_DAYS),
                    now,
                ),
                now,
            )
            habit = ZoneHabit(
                day=today,
                living_w=frame["living"].tolist(),
                rest_w=frame["rest"].tolist(),
            )
            self.state_manager.update_habit("zone_habit", habit)

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
    ) -> tuple[dict, dict] | None:
        """The zone as a second demand on the compressor: what the plan needs
        of it (MPCInput fields) and the models it plans it with
        (MPCOptimizer arguments), or None where it cannot be planned.

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
            return None

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
            )
        except ValueError as error:
            # Normal right after a restart with no measurements yet, or before
            # the weather forecast reaches past now.
            logger.warning("No zone estimate: %s", error)
            return None

        zone.index = pd.to_datetime(zone.index, utc=True)
        # The thermostat's room, which comfort is set in, and the slab and
        # screed under it.
        self.state_manager.update_zone(
            temperature=zone["measured"].dropna(),
            thermal_mass=zone["slab_living"],
            screed=zone.get("screed_living"),
        )

        # The heat pump's own mode unless this run asks for one (see
        # OptimizeConfig.cooling).
        if cooling is None:
            cooling = self.state_manager.cooling(state, config)

        comfort = config.building.comfort(cooling)

        if comfort is None:
            return None

        # The plan starts where the filter's estimate of the whole zone state
        # stands at its first step: both rooms and their slabs.
        planned = zone.reindex(pd.DatetimeIndex(times))
        state_names = list(identifier.state_names(identifier.model))

        if planned[state_names].iloc[0].isna().any():
            logger.info("Zone not planned: no zone estimate at %s", times[0])
            return None

        dew_point_c = self._dew_point_forecast(state, times, mpc_config)

        if cooling and dew_point_c is None:
            logger.info(
                "Zone not planned: no dew point to keep the floor above while "
                "cooling (see building.dew_points)"
            )
            return None

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
            # Beyond the forecast the envelope a day earlier, else the nearest.
            zone_envelope_w_per_k=tuple(
                self.state_manager.extend_forecast(
                    zone["envelope_w_per_k"].dropna().to_dict(), times
                )
            ),
            zone_floor_open=loops[:2] if loops else (),
            zone_floor_flow_lpm=loops[2] if loops else None,
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
    ) -> bool:
        """Whether a floor run is under way: the compressor running in the
        zone's mode now, so a plan that goes on with it makes no start. The
        state alone does not say so - it stays on heating at 0 Hz between
        runs. Without a frequency the state is all there is, as in
        dhw_timing."""

        return (
            bool(heat_pump_state)
            and heat_pump_state[-1].value == mode_state
            and (not compressor_frequency or compressor_frequency[-1].value > 0.0)
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

    # The resolution the zone's room setpoint is published at (K).
    ZONE_SETPOINT_STEP_C = 0.1

    @classmethod
    def zone_setpoint_c(
        cls,
        schedule: tuple[int, ...],
        first: int,
        temperatures: tuple[float, ...],
        measured_c: float | None,
        limit_c: tuple[float, ...] | None,
        cooling: bool,
    ) -> str:
        """The room setpoint (deg C) for the zone run starting at step first:
        the least step of ZONE_SETPOINT_STEP_C past the warmest (coolest)
        temperature the plan has the room at from that run's start to the next
        run's, and past measured_c, the room now; no further than the comfort
        band's edge where the run starts. temperatures are at each step's
        start."""

        end = first

        while end + 1 < len(schedule) and schedule[end + 1]:
            end += 1

        following = next(
            (k for k in range(end + 1, len(schedule)) if schedule[k]), len(schedule)
        )
        window = list(temperatures[first : following + 1])

        if measured_c is not None:
            window.append(measured_c)

        step = cls.ZONE_SETPOINT_STEP_C
        # Strictly past it: at the room's own temperature a thermostat asks for
        # nothing.
        if cooling:
            setpoint_c = (math.ceil(round(min(window) / step, 6)) - 1) * step
            setpoint_c = max(setpoint_c, limit_c[first]) if limit_c else setpoint_c
        else:
            setpoint_c = (math.floor(round(max(window) / step, 6)) + 1) * step
            setpoint_c = min(setpoint_c, limit_c[first]) if limit_c else setpoint_c

        return f"{setpoint_c:.1f}"

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
        start of the run under way or the next one planned, the room setpoint
        for that run (see zone_setpoint_c), and the supply setpoint while a run
        is under way (see zone_supply_c).

        All four 'unknown' where the zone cannot be planned at all: nothing
        then for an automation to act on.
        The supply 'unknown' outside a run too.
        """

        status = start = setpoint = room = "unknown"
        published = None

        if data is not None and result.space_schedule:
            schedule = result.space_schedule
            running = schedule[0] == 1
            states = config.heat_pump.states
            since = self.running_since(
                heat_pump_state,
                states.cooling if data.zone_cooling else states.heating,
            )
            first = next((k for k, on in enumerate(schedule) if on), None)
            # The tank's block ending in this quarter and the zone's run
            # planned from the next: on already, so the heat pump goes from the
            # hot water straight on to the floor - it finishes the tank first.
            # Off, it stops once the hot water is done and starts again for the
            # zone a few minutes later, a start the plan does not make: it
            # counts the hand-over as one run. Heating only: the heat pump
            # stops to reverse between cooling and the tank anyway.
            hand_over = (
                not running
                and not data.zone_cooling
                and len(schedule) > 1
                and schedule[1] == 1
                and result.schedule[:2] == (1, 0)
            )
            # The step whose supply the run is given: the next one's when
            # handed over, the step now being the tank's.
            k = 1 if hand_over else 0
            status = "on" if running or hand_over else "off"

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
            if first is not None and result.zone_temperatures:
                room = self.zone_setpoint_c(
                    schedule,
                    first,
                    result.zone_temperatures,
                    (
                        state.measurements.building.zone_temperature[-1].value
                        if status == "on"
                        and state.measurements.building.zone_temperature
                        else None
                    ),
                    (
                        data.zone_minimum_temperature
                        if data.zone_cooling
                        else data.zone_maximum_temperature
                    ),
                    data.zone_cooling,
                )

            if (running or hand_over) and result.space_supply_c:
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
        self.home_assistant.set_state(
            self.ZONE_SUPPLY_ENTITY,
            setpoint,
            {
                "friendly_name": "Home Optimizer zone supply",
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
