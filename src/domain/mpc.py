"""What the planner is given and what it hands back."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MPCConfig:
    step_hours: float = 0.25
    # Fallback electrical-power assumption for costing, used only when no
    # calibrated HeatPumpCOPModel or outdoor-temperature forecast is
    # available (see MPCOptimizer._electrical_power_w) - otherwise superseded
    # by the calibrated, outdoor-temperature-dependent COP model.
    boiler_electrical_power_w: float = 3000.0
    boiler_min_runtime_steps: int = 2
    # The heat pump stays off this many steps after a run (0.5 hour). A start is not
    # free: real runs put an estimated 0.4-0.5 kWh into reheating the loop and
    # coil before the tank gains anything (estimated from how far the tank
    # settles above the setpoint after a run - see Optimization.publish_dhw), so
    # topping the tank up by a fraction of a degree right after a run costs far
    # more than it stores. It also makes a run end high enough by itself: the
    # plan knows it cannot top up afterwards.
    heat_pump_min_off_steps: int = 2
    # Flat price for now - will become a per-installation config option later.
    price_eur_per_kwh: float = 0.23
    weight_switching: float = 0.1
    weight_temperature_slack: float = 1000.0
    # Decisions within this many hours of now keep the full step_hours
    # resolution; steps beyond that are aggregated into coarse_step_hours
    # blocks purely to shrink the MILP's own binary-variable count (see
    # MPCOptimizer._build_step_plan) - confirmed on real data: solve time is
    # dominated by proving optimality across many binaries, not by finding a
    # good solution, so halving the variable count for the *look-ahead-only*
    # portion of the horizon (re-solved at full precision before it is ever
    # acted on) is a real, low-risk speedup. 12h covers a run and the
    # target it serves at full precision; a deadline further away is only
    # planned coarsely until it comes within reach, by which time it is
    # re-solved finely. Measured on real data against 24h: median solve
    # 6.5 -> 2.9 s, worst 24 -> 9 s, the first decision the same in 59 of 60
    # solves.
    fine_horizon_hours: float = 24.0
    coarse_step_hours: float = 1.0
    # A fallback, not the usual case (s): on real data a plan is proven optimal
    # in 1-8 s. Should a harder one come up, the best plan found by then is
    # used - better than a plan that never comes.
    solve_time_limit_s: float = 30.0


@dataclass(frozen=True)
class MPCInput:
    solar_forecast_w: list[float]
    # Held constant across the horizon: the boiler's local ambient sensor has no
    # forecast (unlike outdoor temperature, which has Open-Meteo) and is indoors,
    # where conditions change slowly relative to a typical MPC horizon.
    ambient_temperature: float
    current_temp_top: float
    current_temp_bottom: float
    boiler_on_current: bool
    target_temperature_top: tuple[float, ...] = ()
    # Forecasted expected additional heat-sink power (W) from tap draws (see
    # features/tap.py's TapForecaster), on top of the passive UA loss already in
    # the dynamics - empty means "no forecast available", treated as no draws
    # (the same assumption implicitly made before this field existed), not a
    # claim that none will occur. This forecast has known, real but modest
    # accuracy (see TapForecaster's own backtest) - weight_temperature_slack
    # absorbs the resulting forecast error, same as it already does for solar.
    # It also targets a quantity that is a mix of real tap draws and a known,
    # uncorrected temperature-dependent heat-transfer gap in the passive-loss
    # model (see BoilerThermalIdentifier.excess_loss_w's docstring) - not tap
    # draws alone.
    tap_forecast_w: tuple[float, ...] = ()
    # Open-Meteo outdoor-temperature forecast (deg C), aligned to the
    # horizon - the evaporator's heat source for an air-water heat pump (see
    # HeatPumpCOPModel.cop()), distinct from `ambient_temperature` above
    # (the boiler's own indoor location). Empty means "no forecast
    # available", falling back to the flat boiler_electrical_power_w
    # assumption for costing (see MPCOptimizer._electrical_power_w) rather
    # than inventing a temperature.
    outdoor_temperature_forecast: tuple[float, ...] = ()
    # Calibrated p10/p90 solar forecasts (W) aligned to the horizon, around
    # solar_forecast_w as p50 - grid import is then costed as an expectation
    # over these three scenarios (see optimizer.SOLAR_SCENARIO_WEIGHTS)
    # instead of assuming p50 comes true. Both empty means "no band
    # available": plan on solar_forecast_w alone.
    solar_p10_w: tuple[float, ...] = ()
    solar_p90_w: tuple[float, ...] = ()
    # Baseload forecast (W) aligned to the horizon: the rest of the house draws
    # this first, so only solar beyond it is available to the heat pump. Empty
    # means no forecast - all solar counts as available.
    baseload_forecast_w: tuple[float, ...] = ()
    # How long the COMPRESSOR in the run in progress has been running (hours),
    # 0 when it is not - the run keeps going until its minimum runtime has
    # passed (see MPCOptimizer). Compressor time, not run time: the resistive
    # booster finishes a run with the compressor off, and that does not protect
    # the compressor from short cycling.
    # Space heating. All empty or None means none is planned, and the model is
    # exactly the domestic-hot-water one it was before - the heat pump serves
    # one demand at a time, so adding the second only ever constrains it.
    #
    # The zone's own temperature now, which the plan starts from.
    zone_temperature: float | None = None
    # The thermal mass's temperature now (deg C). Nothing measures it, so it
    # comes from the Kalman filter's estimate (see building.kalman_states). A
    # plan needs it: starting the screed at the air temperature would claim a
    # cold floor is as ready to heat as a charged one.
    zone_mass_temperature: float | None = None
    # Comfort floor per step, as a schedule rather than one number.
    zone_target_temperature: tuple[float, ...] = ()
    # Comfort ceiling per step: how warm buffering heat may make the zone.
    # Empty means none.
    zone_maximum_temperature: tuple[float, ...] = ()
    # How far below the target the zone may dip before it counts (K).
    zone_comfort_tolerance_c: float = 0.0
    # Heat entering the zone that no decision can change (W), split by where it
    # physically lands: appliances, lighting and people warm the air directly,
    # while shortwave through the glazing is absorbed by floor and furnishings,
    # so the split has to be carried rather than summed away.
    zone_internal_gain_w: tuple[float, ...] = ()
    zone_solar_gain_w: tuple[float, ...] = ()
    # Whether the heat pump is serving the zone right now, the space-heating
    # counterpart of boiler_on_current.
    space_on_current: bool = False
    compressor_elapsed_hours: float = 0.0
    # How long ago the last run ended (hours), 0 while heating - no new run
    # starts until MPCConfig.heat_pump_min_off_steps have passed since then.
    idle_elapsed_hours: float = 0.0


@dataclass(frozen=True)
class MPCResult:
    schedule: tuple[int, ...]
    temperatures: tuple[float, ...]
    # The electrical power (W) assumed for each step, averaged over the step
    # (a partly used last step of a run draws only for its part) - from
    # the calibrated COP model where available, otherwise the flat
    # boiler_electrical_power_w fallback (see
    # MPCOptimizer._electrical_power_w) - reported alongside the schedule so
    # StateManager.update_schedule() can log the actual assumed consumption
    # per step, not a single flat number.
    electrical_power_w: tuple[float, ...]
    # Planned heat into the tank per step (W), by either source.
    heat_w: tuple[float, ...]
    objective_value: float
    solver_status: str
    termination_condition: str
    # Space heating, empty when none was planned: whether the heat pump serves
    # the zone in each step, the heat it delivers there (W), and the zone
    # temperature that results.
    space_schedule: tuple[int, ...] = ()
    space_heat_w: tuple[float, ...] = ()
    zone_temperatures: tuple[float, ...] = ()
