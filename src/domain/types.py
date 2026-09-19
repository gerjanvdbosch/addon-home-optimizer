import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

HeatPumpMode = Literal["heat", "cool"]

ForecasterType = Literal["baseload", "tap"]

IdentificationType = Literal[
    "boiler",
    "cop_dhw",
    "cop_heating",
    "solar",
    "building",
    "building_lumped",
]


class JobType(str, Enum):
    CONFIG = "config"
    UPDATE = "update"
    FIT = "fit"
    PREDICT = "predict"
    TUNE = "tune"
    BACKTEST = "backtest"
    CALIBRATE = "calibrate"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"


class WorkerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"


JsonType = dict[str, Any] | list[Any]


class Settings(BaseModel):
    influx_host: str = Field(
        default="homeassistant.local",
        description="InfluxDB host",
    )
    influx_port: int = Field(
        default=8086,
        description="InfluxDB port",
    )
    influx_username: str = Field(
        default="",
        description="InfluxDB username",
    )
    influx_password: str = Field(
        default="",
        description="InfluxDB password",
    )
    influx_database: str = Field(
        default="home_assistant",
        description="InfluxDB database",
    )
    data_path: Path = Field(
        default=Path("data"),
        description="Data path",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    # Required, no default: a wrong location silently misplaces every reading's
    # solar elevation band (see features.solar), so a missing one must fail.
    latitude: float = Field(description="Installation latitude (degrees)")
    longitude: float = Field(description="Installation longitude (degrees)")


class InfluxSensor(BaseModel):
    measurement: str
    entity_id: str
    field: str
    value_type: str | None = None


class SensorReference(BaseModel):
    entity_id: str = Field()
    attribute: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attribute": None,
            }

        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attribute": value[1],
            }

        return value


T = TypeVar("T")


class SensorAttributesReference(BaseModel, Generic[T]):
    entity_id: str = Field()
    attributes: T

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, str):
            return {
                "entity_id": value,
                "attributes": {},
            }

        if isinstance(value, (list, tuple)):
            return {
                "entity_id": value[0],
                "attributes": value[1],
            }

        return value


class BoilerConfig(BaseModel):
    setpoint: SensorReference = Field()
    top_temperature: SensorReference = Field()
    bottom_temperature: SensorReference = Field()
    ambient_temperature: SensorReference = Field()
    volume: int = Field(default=200)
    target_temperature: float | list[tuple[time, float]] = Field()


class HeatPumpConfig(BaseModel):
    state: SensorReference = Field()
    power: SensorReference = Field()
    supply_temperature: SensorReference = Field()
    return_temperature: SensorReference = Field()
    compressor_frequency: SensorReference = Field()
    flow: SensorReference = Field()
    # Optional: not every installation reports this separately, and its
    # absence should not break anything - see
    # HeatPumpCOPIdentifier.BOOSTER_ACTIVE_STATE for why it matters when
    # present (a resistive backup heater, not the compressor, so its
    # electrical draw follows entirely different physics and must not be
    # mixed into the heat pump's own COP calibration).
    booster: SensorReference | None = Field(default=None)
    # The unit's own outdoor air temperature sensor, if it really measures
    # outdoor air. Optional, and deliberately unset on this installation: the
    # heat pump stands in a shed, so its sensor reads shed air, which runs on
    # average 1.8 K warmer than the 2 m air temperature and carries a strongly
    # diurnal bias (+2.8 K at night against +0.9 K at midday) because the shed
    # retains heat and damps the real outdoor swing. Driving an envelope loss
    # with that would fold a sensor artifact into the building's own dynamics.
    # Without it, Open-Meteo's temperature attribute is used instead - which
    # planning has to use for the future in any case.
    outdoor_temperature: SensorReference | None = Field(default=None)
    boiler: BoilerConfig = Field()


class SouthGlazing(BaseModel):
    """One group of south-facing windows and the shutter in front of it.

    Areas are per group rather than one house total because shading has to be
    weighted by them: the shaded fraction of the glazing is
    sum(area_i * open_i) / sum(area_i), and a plain average over shutters would
    give a small bedroom window the same say as a large living room screen.
    Confirmed on this installation's data, where the four south shutters move
    almost independently (pairwise correlations of 0.07 to 0.23, one of them
    shut continuously for 90 days), so an unweighted mean differs from the
    area-weighted one by a median of 8.8 percentage points.
    """

    glass_m2: float = Field()
    # The cover entity in front of this glazing, reporting current_position
    # (100 = fully open). None means the glass is never shaded.
    cover: SensorReference | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, (int, float)):
            return {"glass_m2": value, "cover": None}

        if isinstance(value, (list, tuple)):
            return {"glass_m2": value[0], "cover": value[1]}

        return value


class ZoneSensor(BaseModel):
    """One room sensor and the floor area it stands for.

    The area is what makes the zone temperature a weighted mean rather than a
    plain average over sensors. Without it the average is weighted by how many
    thermostats a floor happens to have: on this installation four of the five
    Danfoss zones are upstairs, so the ground floor counted for 20% of a
    dwelling temperature while being half its floor area. With warm air
    collecting upstairs that biased the modelled temperature by +0.085 K on
    average (p95 0.42 K); weighting by area brings that to +0.008 K (p95
    0.11 K).
    """

    area_m2: float = Field()
    sensor: SensorReference = Field()

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, (list, tuple)):
            return {"area_m2": value[0], "sensor": value[1]}

        return value


class ClimateConfig(BaseModel):
    temperature: SensorReference = Field()
    setpoint: SensorReference = Field()
    target_temperature: float | list[tuple[time, float]] = Field()
    # Every room sensor that belongs to the modelled zone. The building model
    # averages these into one representative zone temperature, which is what a
    # whole-dwelling energy balance needs: the delivered heat and the baseload
    # it is weighed against are both house-wide, so a single room's thermometer
    # would be an arbitrary sample of the zone. Confirmed on this
    # installation's data, where the five Danfoss room sensors sit within a
    # median 0.33 K of each other - one zone, sampled five times - while the
    # attic runs 3.5 K warmer and tracks outdoor temperature, and so must be
    # left out. Averaging also suppresses the sensors' 0.1 K reporting
    # quantisation, which is a real limit on identifying a building whose daily
    # indoor swing is around 1 K. Empty falls back to `temperature`.
    zone_temperatures: list[ZoneSensor] = Field(default_factory=list)
    # Net floor-to-ceiling height of the conditioned zone. The air volume is
    # derived from it and the zone_temperatures areas, rather than configured
    # separately: those areas are already required for the weighting, so a
    # hand-computed volume would be a second place for the same fact to be
    # wrong. It covers only the rooms that have a sensor, so it slightly
    # undercounts hall, landing and stairwell - acceptable because the volume
    # only sets bounds on a heat capacity (and for the single-node model does
    # not bind at all, since air capacity stays far below MIN_C_J_PER_K for
    # any dwelling-sized zone).
    ceiling_height: float = Field(default=2.6)
    # The zone's south-facing glazing, window group by window group. Not the
    # solar gain itself: the total area is the physical upper bound on the
    # identified effective aperture a_eff_m2 = area * g_value *
    # incidence/soiling factor, all of which are <= 1 (see
    # BuildingThermalIdentifier.calibrate).
    south_glazing: list[SouthGlazing] = Field(default_factory=list)


class SolcastAttributes(BaseModel):
    p10: str = Field(default="pv_estimate10", description="10e percentile")
    p50: str = Field(default="pv_estimate", description="50e percentile")
    p90: str = Field(default="pv_estimate90", description="90e percentile")

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class SolcastConfig(SensorAttributesReference[SolcastAttributes]): ...


class OpenMeteoAttributes(BaseModel):
    temperature: str = Field(default="temperature_2m")
    is_day: str = Field(default="is_day")
    global_tilted_irradiance: str = Field(default="global_tilted_irradiance")
    direct_radiation: str = Field(default="direct_radiation")
    direct_normal_irradiance: str = Field(default="direct_normal_irradiance")
    diffuse_radiation: str = Field(default="diffuse_radiation")
    cloud_cover_low: str = Field(default="cloud_cover_low")
    cloud_cover_mid: str = Field(default="cloud_cover_mid")
    cloud_cover_high: str = Field(default="cloud_cover_high")
    wind_direction: str = Field(default="wind_direction_10m")
    wind_speed: str = Field(default="wind_speed_10m")
    precipitation: str = Field(default="precipitation")

    def items(self):
        return (
            ("temperature", self.temperature),
            ("is_day", self.is_day),
            ("global_tilted_irradiance", self.global_tilted_irradiance),
            ("direct_radiation", self.direct_radiation),
            ("direct_normal_irradiance", self.direct_normal_irradiance),
            ("diffuse_radiation", self.diffuse_radiation),
            ("cloud_cover_low", self.cloud_cover_low),
            ("cloud_cover_mid", self.cloud_cover_mid),
            ("cloud_cover_high", self.cloud_cover_high),
            ("wind_direction", self.wind_direction),
            ("wind_speed", self.wind_speed),
            ("precipitation", self.precipitation),
        )


class OpenMeteoConfig(SensorAttributesReference[OpenMeteoAttributes]): ...


class ForecastConfig(BaseModel):
    solcast: SolcastConfig = Field()
    open_meteo: OpenMeteoConfig = Field()


class Config(BaseModel):
    solar: SensorReference = Field()
    baseload: SensorReference = Field()
    heat_pump: HeatPumpConfig = Field()
    climate: ClimateConfig = Field()
    forecast: ForecastConfig = Field()
    presence: list[SensorReference] = Field(default_factory=list)


class UpdateConfig(BaseModel): ...


class FitConfig(BaseModel):
    target: ForecasterType | None = Field(default=None)
    days: int = Field(default=90)


class PredictConfig(BaseModel):
    target: ForecasterType | None = Field(default=None)
    steps: int = Field(default=192)


class BacktestConfig(BaseModel):
    target: ForecasterType
    days: int = Field(default=90)
    # Same horizon as PredictConfig.steps, so backtest and tune measure the
    # forecast the optimizer actually receives, not a shorter, easier one.
    steps: int = Field(default=192)


class TuneConfig(BacktestConfig):
    trails: int = Field(default=10)


class CalibrateConfig(BaseModel):
    target: IdentificationType | None = Field(default=None)
    days: int = Field(default=90)


class ValidateConfig(BaseModel):
    target: IdentificationType | None = Field(default=None)
    days: int = Field(default=90)
    # Anchors the `days`-long window to this instant instead of "now" - so
    # repeated validations (e.g. checking a model change, or the solar bias
    # identifier's own multi-window pooling) evaluate the exact same
    # historical period rather than one that drifts forward every time the
    # job runs, which otherwise makes two runs' numbers incomparable
    # regardless of what actually changed between them.
    end: datetime | None = Field(default=None)


class OptimizeConfig(BaseModel):
    # MPC horizon in 15-minute steps (same convention as PredictConfig.steps) -
    # fixed here so the optimizer always plans over this many steps, rather
    # than incidentally following however many points the last solar
    # prediction happened to produce.
    steps: int = Field(default=192)


P = TypeVar("P")


class SeriesPoint(BaseModel, Generic[P]):
    time: datetime
    value: P


class BoilerMeasurement(BaseModel):
    top_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    bottom_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    ambient_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class HeatPumpMeasurement(BaseModel):
    mode: HeatPumpMode = "heat"
    state: list[SeriesPoint[str]] = Field(default_factory=list)
    power: list[SeriesPoint[float]] = Field(default_factory=list)
    supply_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    return_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    compressor_frequency: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: BoilerMeasurement = Field(default_factory=BoilerMeasurement)


class ClimateMeasurement(BaseModel):
    temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    setpoint: list[SeriesPoint[float]] = Field(default_factory=list)
    # The average over config.climate.zone_temperatures - what the building
    # model predicts, as opposed to `temperature`, which is the single
    # thermostat the climate setpoint refers to. Kept apart so the dashboard
    # compares the model against the quantity it actually models.
    zone_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class Measurements(BaseModel):
    solar: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload: list[SeriesPoint[float]] = Field(default_factory=list)
    heat_pump: HeatPumpMeasurement = Field(default_factory=HeatPumpMeasurement)
    climate: ClimateMeasurement = Field(default_factory=ClimateMeasurement)


class SolcastForecast(BaseModel):
    p10: list[SeriesPoint[float]] = Field(default_factory=list)
    p50: list[SeriesPoint[float]] = Field(default_factory=list)
    p90: list[SeriesPoint[float]] = Field(default_factory=list)

    def items(self):
        return (
            ("p10", self.p10),
            ("p50", self.p50),
            ("p90", self.p90),
        )


class OpenMeteoForecast(BaseModel):
    temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    global_tilted_irradiance: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_low: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_mid: list[SeriesPoint[float]] = Field(default_factory=list)
    cloud_cover_high: list[SeriesPoint[float]] = Field(default_factory=list)
    wind_direction: list[SeriesPoint[float]] = Field(default_factory=list)
    wind_speed: list[SeriesPoint[float]] = Field(default_factory=list)
    precipitation: list[SeriesPoint[float]] = Field(default_factory=list)

    def items(self):
        return (
            ("temperature", self.temperature),
            ("global_tilted_irradiance", self.global_tilted_irradiance),
            ("cloud_cover_low", self.cloud_cover_low),
            ("cloud_cover_mid", self.cloud_cover_mid),
            ("cloud_cover_high", self.cloud_cover_high),
            ("wind_direction", self.wind_direction),
            ("wind_speed", self.wind_speed),
            ("precipitation", self.precipitation),
        )


class Forecast(BaseModel):
    solcast: SolcastForecast = Field(default_factory=SolcastForecast)
    open_meteo: OpenMeteoForecast = Field(default_factory=OpenMeteoForecast)


class Predictions(BaseModel):
    solar: list[SeriesPoint[float]] = Field(default_factory=list)
    # Calibrated p10/p90 band around `solar` (see features.solar.predict_solar_band),
    # used as the MPC's pessimistic/optimistic solar scenarios.
    solar_p10: list[SeriesPoint[float]] = Field(default_factory=list)
    solar_p90: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload: list[SeriesPoint[float]] = Field(default_factory=list)
    tap: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: list[SeriesPoint[float]] = Field(default_factory=list)
    # The filter's estimate of the building's thermal mass - screed and
    # internal walls - which no sensor measures. Continuous by construction,
    # because the filter corrects every step rather than restarting each
    # horizon, and lagging and damped relative to the air as a heavy mass
    # should be. This is also the state an MPC must start a plan from.
    thermal_mass: list[SeriesPoint[float]] = Field(default_factory=list)


class BoilerSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    temperatures: list[SeriesPoint[float]] = Field(default_factory=list)


class HeatPumpSchedule(BaseModel):
    power: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: BoilerSchedule = Field(default_factory=BoilerSchedule)


class ClimateSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


class Schedule(BaseModel):
    heat_pump: HeatPumpSchedule = Field(default_factory=HeatPumpSchedule)
    climate: ClimateSchedule = Field(default_factory=ClimateSchedule)


class State(BaseModel):
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    measurements: Measurements = Field(default_factory=Measurements)
    forecast: Forecast = Field(default_factory=Forecast)
    predictions: Predictions = Field(default_factory=Predictions)
    schedule: Schedule = Field(default_factory=Schedule)


class BacktestPoint(BaseModel):
    label: str
    points: list[dict[str, object]]
    color: str | None = None
    group: str | None = None


class BacktestResult(BaseModel):
    name: ForecasterType
    label: str
    unit: str
    mae: float
    rmse: float
    r2: float
    points: list[BacktestPoint]


@dataclass
class Job:
    type: JobType
    config: (
        Config
        | UpdateConfig
        | FitConfig
        | PredictConfig
        | TuneConfig
        | BacktestConfig
        | CalibrateConfig
        | ValidateConfig
        | OptimizeConfig
    )
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class BoilerThermalModel:
    volume_l: float
    ua_top_w_per_k: float
    ua_bottom_w_per_k: float
    ua_mix_idle_w_per_k: float
    ua_mix_active_w_per_k: float
    q_in_nominal_w: float
    # Identified from heating runs where the booster heater took over (see
    # BoilerThermalIdentifier._identify_booster): the highest tank temperature
    # the heat pump reaches on its own, the tank's maximum (where the thermostat
    # cuts the booster out - the setpoint those runs used), and the booster's
    # heat input (W). None until such a run has been observed - the optimizer
    # then plans without the booster, and with a default tank maximum (see
    # optimizer.DEFAULT_MAX_TANK_TEMPERATURE_C).
    heat_pump_max_tank_temperature_c: float | None = None
    max_tank_temperature_c: float | None = None
    booster_heat_w: float | None = None
    # How far the settled tank ends above the SWW setpoint of a heat pump run
    # that stopped by itself on it (see
    # BoilerThermalIdentifier._identify_setpoint_overshoot): the setpoint for a
    # planned run is its planned end temperature minus this. None until such a
    # run has been observed.
    setpoint_overshoot_k: float | None = None


@dataclass
class BuildingThermalModel:
    """Two-node (2R2C) grey-box model of one thermal zone.

    x = [T_air, T_mass]:

        C_air  dT_air/dt  = UA_env (T_out - T_air) + UA_am (T_mass - T_air) + Q_int
        C_mass dT_mass/dt = UA_am  (T_air - T_mass) + Q_sol + Q_floor

    Q_floor and Q_sol enter the mass node rather than the air node because that
    is where the physics puts them: the floor circuit runs inside the screed,
    and air is effectively transparent to shortwave radiation, which is
    absorbed by the floor and furnishings. Q_int (metabolic and appliance
    heat) is released convectively into the air.

    One model serves both heating and cooling: none of these parameters
    describes the heat pump. Q_floor is a measured calorimetric input carrying
    its own sign, so cooling is simply a negative Q. Mode-dependence lives in
    the COP model and in the condensation limit, not in this balance.

    Simplification, stated explicitly: the mass node has no direct path to
    outdoors, so envelope mass is lumped with the air node rather than given a
    third node. With a single measured zone temperature a third capacity is not
    identifiable, and inventing one would be a fit term without evidence.
    """

    ua_envelope_w_per_k: float
    ua_air_mass_w_per_k: float
    c_air_j_per_k: float
    c_mass_j_per_k: float
    # Effective solar aperture of the zone's south glazing (m2): glass area
    # times g-value times an incidence/soiling factor. Identified as one
    # lumped parameter because those three factors only ever appear as their
    # product in the heat balance, and the g-value is not separately measured.
    a_eff_m2: float
    # Fraction of the house-wide baseload electrical power that is released as
    # heat inside this zone. The baseload sensor measures the whole house; the
    # modelled zone is only part of it.
    internal_gain_fraction: float


@dataclass
class BuildingLumpedModel:
    """Single-node (1R1C) model of the same zone: C dT/dt = UA (T_out - T) +
    Q_internal + Q_solar + Q_floor.

    One capacity covering everything that stores heat - air, furnishings,
    screed, internal walls together - and one conductance to outdoors. Its
    parameters are NOT a reduction of BuildingThermalModel's: both are fitted
    to the same data and land on genuinely different values, because a single
    node has to account for the whole response with one time constant.

    It exists because the two-node model's mass node is never measured, and on
    cooling-season data that hidden state was the largest single error source:
    dropping it took the six-hour rollout error from 0.36 K to 0.11 K and
    turned an effective solar aperture of 6% of the glass area - which no
    glazing can have - into a perfectly ordinary 50%. Here the state IS the
    measurement, so nothing has to be inferred.

    The cost is real and physical: this cannot represent the floor being warmer
    than the air, so it cannot describe charging the screed as thermal storage.
    That is what the two-node model is for, once data with daily floor-circuit
    transitions can actually identify it - see BuildingLumpedIdentifier.
    """

    ua_w_per_k: float
    c_j_per_k: float
    a_eff_m2: float
    internal_gain_fraction: float


# Exact by definition of the Kelvin scale (0 degC = 273.15 K) - used
# wherever a Celsius temperature must enter a formula (like COP) that is
# only valid on an absolute temperature scale.
KELVIN_OFFSET_C = 273.15


@dataclass
class HeatPumpCOPModel:
    eta_carnot: float
    delta_t_cond: float
    delta_t_evap: float
    # The 95th percentile of this mode's own observed supply temperature
    # (see HeatPumpCOPIdentifier.calibrate()) - planning's stand-in for the
    # heat pump's actual supply temperature, which it has no forecast for
    # (see MPCOptimizer._power_line_coefficients). Not read by cop() itself
    # (default 0.0 is harmless there, e.g. for a trial fit's parameter
    # vector - see HeatPumpCOPIdentifier._predict_cop).
    reference_supply_temperature_c: float = 0.0
    # Thermal output (W) at HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C/HIGH_C from
    # a line fitted to real calorimetric data so that it reproduces measured
    # electrical power (see HeatPumpCOPIdentifier._fit_q_th_line()) - used
    # by MPCOptimizer instead of BoilerThermalModel's fixed q_in_nominal_w
    # when estimating electrical power: real data confirmed Q_th is not
    # constant across a compressor run (it rises from a low start, peaks
    # mid-cycle, then falls as the compressor modulates down approaching
    # setpoint), so q_in_nominal_w (calibrated for the tank's temperature
    # *trajectory*, a different purpose) understated real electrical draw
    # through the middle of a cycle. Not read by cop() itself (default 0.0
    # is harmless there, e.g. for a trial fit's parameter vector).
    q_th_at_power_fit_low_w: float = 0.0
    q_th_at_power_fit_high_w: float = 0.0

    def cop(self, T_outdoor: float, T_supply: float) -> float:
        """Carnot COP scaled by eta_carnot - see
        HeatPumpCOPIdentifier's class docstring (features/cop.py) for the
        physical derivation. Kept on the model itself, not just the
        identifier, so planning code (MPCOptimizer) can evaluate a
        calibrated model directly without depending on the identifier that
        produced it. Works equally on scalars or numpy arrays (T_outdoor,
        T_supply is plain arithmetic, no numpy-specific code needed) - used
        both by MPCOptimizer (scalars, one step at a time) and by
        HeatPumpCOPIdentifier._predict_cop (arrays, one call per fit
        iteration).
        """

        T_cond_K = T_supply + self.delta_t_cond + KELVIN_OFFSET_C
        T_evap_K = T_outdoor - self.delta_t_evap + KELVIN_OFFSET_C

        return self.eta_carnot * T_cond_K / (T_cond_K - T_evap_K)


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
    # coil before the tank gains anything (see
    # BoilerThermalIdentifier._identify_setpoint_overshoot's own data), so
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
    # acted on) is a real, low-risk speedup. 24h keeps at least the next
    # daily target deadline at full precision regardless of what time of day
    # this solves, given this installation's targets recur roughly daily.
    fine_horizon_hours: float = 24.0
    coarse_step_hours: float = 1.0


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
    # Comfort floor per step, as a schedule rather than one number.
    zone_target_temperature: tuple[float, ...] = ()
    # Heat entering the zone that no decision can change (W): solar through the
    # glazing plus internal gains. Passed as one series because a single-node
    # zone cannot tell them apart - they enter the same node with the same
    # coefficient - and the split would be a distinction the model does not
    # make.
    zone_gain_w: tuple[float, ...] = ()
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
