"""The installation: which sensors it has, and what it is asked to reach.

One section per physical subsystem, each carrying its own sensors, its own
physical constants and its own target schedule.
"""

from datetime import time
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from domain.sensors import SensorAttributesReference, SensorReference


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


class BoilerConfig(BaseModel):
    setpoint: SensorReference = Field()
    top_temperature: SensorReference = Field()
    bottom_temperature: SensorReference = Field()
    ambient_temperature: SensorReference = Field()
    volume: int = Field(default=200)
    target_temperature: float | list[tuple[time, float]] = Field()


class HeatPumpStates(BaseModel):
    """The values heat_pump.state reports for each operating mode, which differ
    per brand and language. The defaults are this Ecodan's Dutch labels.

    Each is matched exactly, as a whitelist: supply, return and flow are shared
    by the tank and the floor circuit, so a mode that is not recognised must
    count as neither rather than be taken for one of them.
    """

    # The only value trusted as "compressor definitely off", where a stale
    # flow or power reading is reset to 0.
    off: str = Field(default="Uit")
    # Heat goes to the hot water tank.
    dhw: str = Field(default="SWW")
    # Heat goes to (or, cooling, is taken from) the floor circuit.
    heating: str = Field(default="Verwarmen")
    cooling: str = Field(default="Koelen")


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
    states: HeatPumpStates = Field(default_factory=HeatPumpStates)


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


class Room(BaseModel):
    """One room of the modelled zone: its floor area and its thermometer.

    Rooms are the parts; the zone is the whole. The model has a single zone
    with one air temperature, and these are the measurements it is averaged
    from - which is why this is a room rather than a zone of its own.

    The area is what makes the zone temperature a weighted mean rather than a
    plain average over sensors. Without it the average is weighted by how many
    thermostats a floor happens to have: on this installation four of the five
    Danfoss rooms are upstairs, so the ground floor counted for 20% of a
    dwelling temperature while being half its floor area. With warm air
    collecting upstairs that biased the modelled temperature by +0.085 K on
    average (p95 0.42 K); weighting by area brings that to +0.008 K (p95
    0.11 K).
    """

    area_m2: float = Field()
    temperature: SensorReference = Field()

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, value):
        if isinstance(value, (list, tuple)):
            return {"area_m2": value[0], "temperature": value[1]}

        return value


class Thermostat(BaseModel):
    """The one thermostat the occupants actually set.

    Its reading and its setpoint are one device, which is why they are grouped
    rather than sitting loose beside `rooms`: that list is what the model
    averages over, this is what a person turns up.
    """

    temperature: SensorReference = Field()
    setpoint: SensorReference = Field()


class BuildingConfig(BaseModel):
    thermostat: Thermostat = Field()
    target_temperature: float | list[tuple[time, float]] = Field()
    # The warmest the zone may be heated to (deg C), a schedule like the target.
    # Buffering heat in the floor means heating above the target while heat is
    # cheap, and this is the ceiling of that: without one, cheap heat would be
    # stored without limit. None plans no space heating at all.
    maximum_temperature: float | list[tuple[time, float]] | None = Field(default=None)
    # How far below the target the zone may dip without counting as a shortfall
    # (K). A comfort choice, not physics: 0 holds the target exactly, down to a
    # predicted dip of hundredths of a degree that no one would notice.
    comfort_tolerance: float = Field(default=0.0, ge=0.0)
    # Every room that belongs to the modelled zone. The building model averages
    # their sensors into one representative zone temperature, which is what a
    # whole-dwelling energy balance needs: the delivered heat and the baseload
    # it is weighed against are both house-wide, so a single room's thermometer
    # would be an arbitrary sample of the zone. Confirmed on this
    # installation's data, where the five Danfoss room sensors sit within a
    # median 0.33 K of each other - one zone, sampled five times - while the
    # attic runs 3.5 K warmer and tracks outdoor temperature, and so must be
    # left out. Averaging also suppresses the sensors' 0.1 K reporting
    # quantisation, which is a real limit on identifying a building whose daily
    # indoor swing is around 1 K. Empty falls back to the thermostat.
    rooms: list[Room] = Field(default_factory=list)
    # Net floor-to-ceiling height of the conditioned zone. The air volume is
    # derived from it and the room areas, rather than configured separately:
    # those areas are already required for the weighting, so a hand-computed
    # volume would be a second place for the same fact to be wrong. It covers
    # only the rooms that have a sensor, so it slightly undercounts hall,
    # landing and stairwell - acceptable because the volume only sets bounds on
    # a heat capacity.
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
    building: BuildingConfig = Field()
    forecast: ForecastConfig = Field()
    presence: list[SensorReference] = Field(default_factory=list)
