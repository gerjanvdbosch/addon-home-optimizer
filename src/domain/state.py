"""What the system has measured, predicted, and planned.

The running picture the dashboard draws and the optimizer starts from, as
plain series of timestamped points.
"""

from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from domain.models import ForecasterType, HeatPumpMode

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


class BuildingMeasurement(BaseModel):
    temperature: list[SeriesPoint[float]] = Field(default_factory=list)
    setpoint: list[SeriesPoint[float]] = Field(default_factory=list)
    # The average over config.building.rooms - what the building model
    # predicts, as opposed to `temperature`, which is the single thermostat the
    # setpoint refers to. Kept apart so the dashboard compares the model
    # against the quantity it actually models.
    zone_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


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
    # Where the zone temperature goes if nothing is done to it, from each of
    # the two building structures. Both are kept on purpose: they disagree
    # mainly about how much sun gets in, and a forecast is the one test that
    # can settle that without waiting for a heating season - the measurement
    # catches up to them a day later and says which was right.
    zone_forecast_lumped: list[SeriesPoint[float]] = Field(default_factory=list)
    zone_forecast_two_node: list[SeriesPoint[float]] = Field(default_factory=list)
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


class BuildingSchedule(BaseModel):
    target_temperature: list[SeriesPoint[float]] = Field(default_factory=list)


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
