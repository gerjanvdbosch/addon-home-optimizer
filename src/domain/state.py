"""What the system has measured, predicted, and planned.

The running picture the dashboard draws and the optimizer starts from, as
plain series of timestamped points.
"""

from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from domain.models import ForecasterType

P = TypeVar("P")


class SeriesPoint(BaseModel, Generic[P]):
    time: datetime
    value: P


class BoilerMeasurement(BaseModel):
    top_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    bottom_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    ambient_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    # Top less bottom, held since the tank was last mixed (K, see
    # physics.tank_stratification_k): what the cold layer is read from.
    stratification: list[SeriesPoint[float]] = Field(default_factory=list)


class HeatPumpMeasurement(BaseModel):
    state: list[SeriesPoint[str]] = Field(default_factory=list)
    power: list[SeriesPoint[float]] = Field(default_factory=list)
    supply_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    return_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    flow: list[SeriesPoint[float]] = Field(default_factory=list)
    compressor_frequency: list[SeriesPoint[float]] = Field(default_factory=list)
    # The operating mode it is set to (see HeatPumpConfig.mode).
    mode: list[SeriesPoint[str]] = Field(default_factory=list)
    boiler: BoilerMeasurement = Field(default_factory=BoilerMeasurement)


class BuildingMeasurement(BaseModel):
    setpoint: list[SeriesPoint[float]] = Field(default_factory=list)
    # The thermostat's room, which the setpoint refers to and the building
    # model plans (see Optimization._zone).
    zone_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    # The highest of config.building.dew_points: the air most likely to
    # condense on a cooled surface.
    dew_point: list[SeriesPoint[float]] = Field(default_factory=list)


class Measurements(BaseModel):
    solar: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload: list[SeriesPoint[float]] = Field(default_factory=list)
    heat_pump: HeatPumpMeasurement = Field(default_factory=HeatPumpMeasurement)
    building: BuildingMeasurement = Field(default_factory=BuildingMeasurement)


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
    dew_point: list[SeriesPoint[float]] = Field(default_factory=list)

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
            ("dew_point", self.dew_point),
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
    # The band around `baseload` (see BaseloadForecaster.band_quantiles), the
    # MPC's low and high baseload scenarios.
    baseload_p10: list[SeriesPoint[float]] = Field(default_factory=list)
    baseload_p90: list[SeriesPoint[float]] = Field(default_factory=list)
    tap: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: list[SeriesPoint[float]] = Field(default_factory=list)
    # The filter's estimate of the building's thermal mass - screed and
    # internal walls - which no sensor measures. Continuous by construction,
    # because the filter corrects every step rather than restarting each
    # horizon, and lagging and damped relative to the air as a heavy mass
    # should be. This is also the state an MPC must start a plan from.
    thermal_mass: list[SeriesPoint[float]] = Field(default_factory=list)
    # The indoor dew point over the plan's horizon (see features.dew_point):
    # what a cooled floor must stay above.
    dew_point: list[SeriesPoint[float]] = Field(default_factory=list)


class BoilerSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    temperatures: list[SeriesPoint[float]] = Field(default_factory=list)
    # Whether the plan heats the tank (1/0): the next plan starts from it (see
    # MPCInput.previous_tank_on).
    on: list[SeriesPoint[int]] = Field(default_factory=list)


class HeatPumpSchedule(BaseModel):
    # One machine, one plan: the electricity it draws (W) and the heat it
    # delivers (W), each for the tank and the floor together - the floor's
    # heat as a magnitude, so cooling counts positive too.
    power: list[SeriesPoint[float]] = Field(default_factory=list)
    heat: list[SeriesPoint[float]] = Field(default_factory=list)
    boiler: BoilerSchedule = Field(default_factory=BoilerSchedule)


class BuildingSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    # The zone's part of the plan, for comparison with what the thermostats
    # do - not acted on yet (see Optimization._zone). Its heat and power are
    # the heat pump's (see HeatPumpSchedule).
    temperatures: list[SeriesPoint[float]] = Field(default_factory=list)
    # The supply each planned step runs the floor at: while cooling, the
    # setpoint the plan would give the heat pump. Only steps it runs.
    supply: list[SeriesPoint[float]] = Field(default_factory=list)
    # Whether the plan serves the zone (1/0), as BoilerSchedule.on.
    on: list[SeriesPoint[int]] = Field(default_factory=list)
    # The supply setpoint last published during a cooling run, which a later
    # one in the same run may not exceed (see Optimization.zone_setpoint_c).
    supply_setpoint: SeriesPoint[float] | None = None


class Schedule(BaseModel):
    heat_pump: HeatPumpSchedule = Field(default_factory=HeatPumpSchedule)
    building: BuildingSchedule = Field(default_factory=BuildingSchedule)


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
