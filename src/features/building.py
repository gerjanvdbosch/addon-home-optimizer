"""Identification of the building zone's thermal parameters.

The balances themselves live in domain/physics.py; this module is how their
parameters are obtained from measurements - the dataset, the rollout windows,
the fit and the diagnostics that say whether the result may be believed.

One model serves both space heating and space cooling. The envelope physics -
transmission to outdoors, thermal mass, solar gain through south glazing,
internal gains - is the same in both modes, and the heat pump's contribution
enters as a *measured* calorimetric term carrying its own sign, so cooling is
simply a negative heat input. Nothing in BuildingThermalModel describes the
heat pump, so there is no parameter that could differ per mode. Mode-dependence
belongs in the COP model (already one instance per mode, see features/cop.py),
not in this heat balance. Floor cooling is additionally limited by condensation
on the floor surface, which is a constraint on how the zone may be cooled rather
than a term in its energy balance - not modelled here, and not yet anywhere.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pvlib import irradiance, solarposition
from scipy.optimize import least_squares

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.dynamics import discretize_zoh, kalman_states
from domain.models import BuildingThermalModel
from domain.physics import (
    CP_AIR_J_PER_KG_K,
    RHO_AIR_KG_PER_M3,
    floor_heat_w,
    internal_gain_w,
    solar_gain_w,
    two_node_zone_state_space,
    zone_observation,
)
from domain.sensors import Aggregation, FillMethod, SensorReference
from features.dataset import DatasetBuilder
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)

# The modelled glazing faces due south on a vertical facade. Both are geometry
# of this installation, expressed in pvlib's convention (azimuth clockwise from
# north, tilt from horizontal).
SOUTH_FACADE_AZIMUTH_DEG = 180.0
VERTICAL_FACADE_TILT_DEG = 90.0


def facade_irradiance_w_per_m2(
    interval_midpoints: pd.Series,
    direct_normal: np.ndarray,
    diffuse_horizontal: np.ndarray,
    global_horizontal: np.ndarray,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """Plane-of-array irradiance on the vertical south facade (W/m2).

    The Open-Meteo `global_tilted_irradiance` attribute already in the config is
    computed for the PV array's own tilt and azimuth, so it does not describe
    this facade; the transposition is redone here from the three components
    Open-Meteo reports independently of any surface (DNI, DHI, GHI).

    The irradiance values are means over a step, so the transposition uses the
    sun's position at that step's MIDPOINT rather than its start - over 15
    minutes the sun moves nearly four degrees in azimuth, which a vertical
    facade sees directly in its angle of incidence.

    Uses pvlib's isotropic sky-diffuse model: the simplest transposition with no
    free parameters. A Perez-type model would be more accurate for diffuse on a
    vertical surface, but its extra empirical coefficients cannot be checked
    against anything measured at this installation, and the identified effective
    aperture a_eff_m2 would silently absorb the difference.
    """

    position = solarposition.get_solarposition(
        pd.DatetimeIndex(interval_midpoints), latitude=latitude, longitude=longitude
    )

    total = irradiance.get_total_irradiance(
        surface_tilt=VERTICAL_FACADE_TILT_DEG,
        surface_azimuth=SOUTH_FACADE_AZIMUTH_DEG,
        solar_zenith=position["apparent_zenith"].to_numpy(),
        solar_azimuth=position["azimuth"].to_numpy(),
        dni=direct_normal,
        ghi=global_horizontal,
        dhi=diffuse_horizontal,
        model="isotropic",
    )

    return np.nan_to_num(np.asarray(total["poa_global"], dtype=float), nan=0.0)


def _rollout(
    a: np.ndarray,
    b: np.ndarray,
    initial_state: np.ndarray,
    inputs: np.ndarray,
    dt_seconds: np.ndarray,
) -> np.ndarray:
    """Forward-simulate a window from one initial state, for either structure.

    `inputs` holds u = [T_outdoor, Q_internal, Q_solar, Q_floor] per sample, and
    only the initial state is taken from measurements - no mid-window room
    temperature is used - so this measures genuine forward-simulation accuracy
    rather than one-step curve fitting. Zero-order hold: the input at the START
    of each interval governs that interval, matching the discretization.
    """

    n = len(dt_seconds)

    simulated = np.empty((n, len(initial_state)))
    simulated[0] = initial_state

    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for i in range(1, n):
        dt = float(dt_seconds[i])
        key = round(dt, 3)

        if key not in cache:
            cache[key] = discretize_zoh(a, b, dt)

        a_d, b_d = cache[key]

        simulated[i] = a_d @ simulated[i - 1] + b_d @ inputs[i - 1]

    return simulated


class BuildingThermalIdentifier(SystemIdentifier[BuildingThermalModel]):
    """Identifies the zone's envelope parameters from measured room temperature.

    Both the heating and the cooling season feed the same fit: Q_floor is a
    measured calorimetric input (see floor_heat_w), so a cooling run is just a
    negative heat input and no parameter here is mode-specific.

    Parameters are identified from multi-hour rollouts, not one-step residuals.
    This installation's room sensor reports on change, roughly every half hour
    at 0.1 K resolution, so the true one-step temperature change over a 15
    minute step is largely inside the quantisation and interpolation noise - the
    same reason BoilerThermalIdentifier scores its passive-loss parameters over
    2 hour decay windows rather than single steps.

    Windows where the floor circuit is running are deliberately kept in: during
    free float the air and mass nodes drift together, so the coupling between
    them (ua_air_mass_w_per_k) is only really excited when heat is injected into
    the floor. Free-float windows in turn carry almost all the information about
    the envelope loss. The fit needs both, and validate() reports each regime
    separately so one cannot hide a poor result in the other.
    """

    # A flow reading must be strictly positive to mean anything physically.
    MIN_FLOW_LPM = 0.0

    TRAIN_RATIO = 0.80

    # Excludes transitions spanning a data gap far larger than the nominal
    # sampling interval, which would test steady-state convergence rather than
    # the dynamics - same role and value as in BoilerThermalIdentifier.
    MAX_DT_SECONDS_MULTIPLE = 6.0

    # Length of a scored rollout. Long enough for the building's own time
    # constants (hours, set by c_mass over ua_envelope) to produce a temperature
    # change well clear of the 0.1 K sensor resolution, short enough that
    # unmeasured disturbances - an opened window, a wood stove, an unmodelled
    # zone exchanging air - are unlikely to span a whole window.
    ROLLOUT_HORIZON_HOURS = 6.0

    # T_mass is never measured, so a simulation has to start it somewhere: each
    # contiguous run starts it equal to the air temperature and scores nothing
    # until this much has elapsed. It must outlast the mass node's own time
    # constant c_mass / ua_air_mass, which for a floor-heated dwelling is around
    # half a day (tens of MJ/K over a few hundred W/K); a day of lead-in leaves
    # a few percent of the initial error. A shorter warm-up was tried and
    # rejected: at 6 hours the fit traded the unresolved initial mass
    # temperature against the envelope conductance and recovered it several
    # times too small on synthetic data with known parameters.
    MASS_WARMUP_HOURS = 24.0

    # The zone sensors report on a 0.01 K grid - verified against 30 days of
    # raw readings, which sit on it exactly and do not fit a coarser one. A
    # uniform quantisation error of that width has variance step^2/12, which is
    # what the filter is told about the measurement: taken from the instrument,
    # not tuned.
    SENSOR_RESOLUTION_K = 0.01

    # Standard deviation of the heat flow the model does not account for, which
    # is what the filter's process noise stands in for. Its scale is what an
    # unmodelled ventilation path actually moves: three air changes per hour
    # through a 200 m3 zone - a window wide open - shifts about 1 kW at a 5 K
    # indoor-outdoor difference and some 3 kW at the 15 K of a winter day.
    # Measured unexplained flows on this installation are the same order (about
    # 1.3 kW during cooling runs).
    #
    # Stating it as a heat flow rather than an abstract covariance is what makes
    # that argument possible at all. It also has to be large relative to the
    # 0.03 K sensor noise, because the model is demonstrably the less reliable
    # of the two: below roughly 1 kW the filter starts trusting the model over
    # the thermometer and the result gets worse. Above it the result is flat
    # across two orders of magnitude, so nothing here hinges on the exact
    # number - the tests check that.
    PROCESS_NOISE_W = 3000.0

    # Standard MAD-to-std conversion for a normal distribution, used to set the
    # robust loss scale from the data's own residual spread (not physical).
    MAD_TO_STD = 1.4826
    MIN_F_SCALE = 1e-6

    # Envelope conductance of a whole dwelling of this size, from a very well
    # insulated new build to a poorly insulated older one. A sanity range, not
    # an expected value.
    MIN_UA_ENVELOPE_W_PER_K = 30.0
    MAX_UA_ENVELOPE_W_PER_K = 500.0
    INITIAL_UA_ENVELOPE_W_PER_K = 150.0

    # Convective/radiative coupling between the internal mass surfaces (floor,
    # internal walls) and room air: a combined surface coefficient of roughly
    # 3-8 W/m2K over an internal surface area of tens to a few hundred m2.
    MIN_UA_AIR_MASS_W_PER_K = 50.0
    MAX_UA_AIR_MASS_W_PER_K = 2000.0
    INITIAL_UA_AIR_MASS_W_PER_K = 400.0

    # The air node's capacity is at least the zone's own air (rho*V*cp) and at
    # most that many times more, covering the light furnishings and surface
    # layers that follow the air almost instantly. Both bounds are derived from
    # the configured volume, never asserted in Joules. Ten times air is about
    # what furnishings can physically account for: roughly 10 kg/m2 of wood over
    # the floor area is 2.1 MJ/K against 0.3 MJ/K of air, so about 8x.
    #
    # On real data this pins at the upper bound, which is reported rather than
    # tuned away. Raising a ceiling a parameter is chasing is not evidence for a
    # larger true value - the same trap already documented for
    # HeatPumpCOPIdentifier.MAX_DELTA_T_EVAP. The likely physical cause is that
    # the zone temperature comes from wall-mounted thermostats, whose own
    # response lags the air they measure, so the fit needs a slower "air" node
    # than air actually is.
    MIN_AIR_CAPACITY_MULTIPLE = 1.0
    MAX_AIR_CAPACITY_MULTIPLE = 10.0
    INITIAL_AIR_CAPACITY_MULTIPLE = 3.0

    # Screed, floor slab and internal walls. 2 MJ/K is about a tonne of
    # concrete (cp ~ 880 J/kgK), the least a floor-heated dwelling can have;
    # 50 MJ/K is tens of tonnes, more structure than a house of this size has.
    MIN_C_MASS_J_PER_K = 2.0e6
    MAX_C_MASS_J_PER_K = 50.0e6
    INITIAL_C_MASS_J_PER_K = 15.0e6

    # Fraction of the configured south glass area used as the starting guess
    # for the effective aperture: a typical double-glazing g-value times a
    # yearly-average incidence/soiling factor lands near half the geometric
    # area. The bounds themselves are [0, glass area] - see calibrate().
    INITIAL_APERTURE_FRACTION = 0.5

    # A wall thermostat exchanges longwave radiation with the surfaces around
    # it, so it reads an operative temperature between air and mass rather than
    # air alone (see BuildingThermalModel.sensor_mass_fraction). 0.5 is that
    # textbook average in still air, and the ceiling here: the sensor sits in
    # moving room air, so it cannot follow the surfaces more closely than
    # evenly. On this installation's cooling data the fit runs into that
    # ceiling, which says the thermostats follow the structure at least that
    # closely - reported rather than tuned away, like every other pinned bound
    # here, and one for the heating season to settle.
    MAX_SENSOR_MASS_FRACTION = 0.5
    INITIAL_SENSOR_MASS_FRACTION = 0.25

    # An unmodelled heat flow can enter either node: the air through
    # ventilation, a stove or a visitor, the mass through heat that was
    # measured into the floor circuit but lost in the pipe run before the
    # screed (this heat pump stands in a shed). Q_internal and Q_floor are the
    # input channels that land there, so the filter carries one disturbance per
    # node instead of claiming the mass is driven exactly as measured.
    DISTURBANCE_INPUTS = (1, 3)

    # Share of the house-wide baseload dissipated inside the modelled zone.
    # Physically a fraction, hence [0, 1]; the living zone is a substantial but
    # not dominant part of the house, so the fit starts mid-range.
    INITIAL_INTERNAL_GAIN_FRACTION = 0.5

    def __init__(self, latitude: float, longitude: float) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        # Overwritten by dataset() from ClimateConfig once available; the
        # defaults only keep prepare()/calibrate() callable on their own.
        self.volume_m3: float = 0.0
        self.glazing_areas_m2: list[float] = []
        self.shutter_areas_m2: list[float] = []
        self.room_temperature_columns: list[str] = []
        self.room_areas_m2: list[float] = []
        self.shutter_columns: list[str] = []
        self.presence_columns: list[str] = []
        self.states = HeatPumpStates()
        self.parameter_std_errors: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return "building"

    @property
    def label(self) -> str:
        return "Room temperature"

    @property
    def unit(self) -> str:
        return "°C"

    # Home Assistant device_tracker convention, and the numeric form InfluxDB
    # exports these trackers as. Whitelist, not a blacklist of "anything that
    # isn't not_home": "unknown"/"unavailable" or a missing reading means the
    # tracker is unreliable right now and must not be counted as a person
    # sitting in the room radiating 75 W.
    HOME_PRESENCE_STATE = "home"
    HOME_PRESENCE_VALUE = 1.0

    # Home Assistant cover convention: current_position is a percentage with
    # 100 meaning fully open.
    FULLY_OPEN_POSITION = 100.0

    # a_eff_m2 = glass area * g-value * frame factor * soiling. The geometric
    # angle of incidence is NOT in there - the transposition to the facade
    # already accounts for it - so from the glazing alone the product has a
    # floor: even solar-control glass has a g-value around 0.25, and a frame
    # factor below 0.7 would mean more frame than glass, giving about 0.17.
    #
    # The threshold sits deliberately BELOW that floor, because a_eff also
    # absorbs facade shading the transposition knows nothing about - an
    # overhang, a neighbouring building, a tree - which can legitimately push
    # the effective aperture under what the glazing alone allows. That makes
    # this a test for values no combination of glass and shade could produce,
    # not a test for good glass. A result between this threshold and 0.17 is
    # therefore not an endorsement: it passes only if there is real shading to
    # point at.
    MIN_PLAUSIBLE_APERTURE_FRACTION = 0.15

    # A thermal model has to beat holding the last measurement for the length of
    # a rollout. That baseline is free, needs no parameters, and is genuinely
    # hard to beat indoors over a few hours: room temperature is a slow random
    # walk. Anything at or below zero skill means the model's own dynamics are
    # adding error rather than information, however small its MAE looks next to
    # the sensor resolution - which is exactly what happened here on
    # cooling-season data, where the model scored 0.36 K against persistence's
    # 0.16 K.
    MIN_SKILL_VS_PERSISTENCE = 0.0

    # Samples inside one rollout are highly correlated - they come from a single
    # forward simulation - so the independent unit is the WINDOW, not the
    # sample. Below this many windows containing floor activity, the difference
    # between two mean absolute errors cannot be told from the spread between
    # individual rollouts, and the active-regime skill is reported as
    # insufficient evidence rather than as a verdict. Same principle as
    # BoilerThermalIdentifier.MIN_CALORIMETRIC_Q_IN_SAMPLES.
    MIN_ACTIVE_WINDOWS = 10

    # Below this many windows a slope fitted through them says more about which
    # windows happened to occur than about the model.
    MIN_WINDOWS_FOR_A_TREND = 20

    # Below this the sun is doing nothing a 0.1 K sensor could notice over a
    # rollout: a watt per square metre of facade through the whole configured
    # aperture is a few watts into a dwelling.
    NEGLIGIBLE_SOLAR_GAIN_W = 1.0

    # Two standard errors: the ordinary convention for "not zero", applied to
    # the fitted slope rather than to a single measurement.
    SIGNIFICANT_SLOPE_STD_ERRORS = 2.0

    # Open-Meteo reports wind speed in km/h unless asked otherwise; exact by
    # definition of both units.
    KMH_PER_M_PER_S = 3.6

    PARAMETER_NAMES = (
        "ua_envelope_w_per_k",
        "ua_air_mass_w_per_k",
        "c_air_j_per_k",
        "c_mass_j_per_k",
        "a_eff_m2",
        "sensor_mass_fraction",
        "internal_gain_fraction",
    )

    @staticmethod
    def _state_space(model) -> tuple[np.ndarray, np.ndarray]:
        """The structure this identifier fits."""

        return two_node_zone_state_space(model)

    @staticmethod
    def _parameters(model: BuildingThermalModel) -> np.ndarray:
        return np.array(
            [
                model.ua_envelope_w_per_k,
                model.ua_air_mass_w_per_k,
                model.c_air_j_per_k,
                model.c_mass_j_per_k,
                model.a_eff_m2,
                model.sensor_mass_fraction,
                model.internal_gain_fraction,
            ]
        )

    def _model_from_parameters(self, x: np.ndarray) -> BuildingThermalModel:
        return BuildingThermalModel(
            ua_envelope_w_per_k=float(x[0]),
            ua_air_mass_w_per_k=float(x[1]),
            c_air_j_per_k=float(x[2]),
            c_mass_j_per_k=float(x[3]),
            a_eff_m2=float(x[4]),
            sensor_mass_fraction=float(x[5]),
            internal_gain_fraction=float(x[6]),
        )

    def _shutter_open_fraction(self, df: pd.DataFrame) -> pd.Series:
        """Area-weighted unshaded fraction of the zone's south glazing.

        sum(area_i * open_i) / sum(area_i): shading is an area effect, so a
        small bedroom window may not carry the same weight as a large living
        room screen. Confirmed necessary on this installation's data - the four
        south shutters move almost independently, and an unweighted mean differs
        from this by a median of 8.8 percentage points.

        Glazing without a cover contributes its full area as permanently
        unshaded, which is what keeps this a fraction of the TOTAL south glass
        that a_eff_m2 is bounded by.
        """

        total_area = sum(self.glazing_areas_m2)

        if total_area <= 0.0:
            return pd.Series(1.0, index=df.index)

        unshaded_area = total_area - sum(self.shutter_areas_m2)
        weighted = pd.Series(unshaded_area, index=df.index)

        for column, area in zip(
            self.shutter_columns, self.shutter_areas_m2, strict=True
        ):
            position = pd.to_numeric(df[column], errors="coerce")
            # A missing position reading must not silently mean "shut": an
            # absent cover reading says nothing about the glass, and assuming
            # full shading would attribute real solar gain to the envelope.
            open_fraction = (position / self.FULLY_OPEN_POSITION).fillna(1.0)
            weighted = weighted + area * open_fraction.clip(0.0, 1.0)

        return (weighted / total_area).clip(0.0, 1.0)

    def _occupants(self, df: pd.DataFrame) -> pd.Series:
        if not self.presence_columns:
            return pd.Series(0.0, index=df.index)

        at_home = [
            (df[column] == self.HOME_PRESENCE_STATE)
            | (pd.to_numeric(df[column], errors="coerce") == self.HOME_PRESENCE_VALUE)
            for column in self.presence_columns
        ]

        return pd.concat(at_home, axis=1).sum(axis=1).astype(float)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turns the loaded frame into the model's inputs on one regular grid.

        The weather frame is a series of forecast snapshots, so each valid time
        appears once per snapshot that covered it. Only the freshest snapshot
        that still looked forward is kept - the closest thing to a measurement
        of irradiance this installation has, since no pyranometer is installed.
        """

        df = df.copy()

        # One representative temperature for the zone, weighted by floor area.
        # Averaging rather than picking a room: the heat this is weighed against
        # (Q_floor, the baseload) is delivered to the whole dwelling, and the
        # sensors' own 0.1 K reporting steps partly cancel across them. The
        # weighting matters because thermostats are not spread evenly over a
        # dwelling - see Room. A room that has dropped out is left out of
        # that step's mean, and its weight with it, instead of voiding the step.
        if self.room_temperature_columns:
            rooms = df[self.room_temperature_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            weights = pd.Series(self.room_areas_m2, index=self.room_temperature_columns)
            available = rooms.notna() * weights
            df["T_air"] = (rooms * weights).sum(axis=1) / available.sum(axis=1)

        # Without a configured outdoor sensor, Open-Meteo's own temperature is
        # the only outdoor air temperature available (see
        # HeatPumpConfig.outdoor_temperature for why the measurement is
        # preferred when there is one).
        if "T_out" not in df.columns and "temperature" in df.columns:
            df["T_out"] = df["temperature"]

        # Every one of these is requested unconditionally by dataset(), so a
        # missing column means a misconfigured sensor, not an optional input -
        # better to say which one than to quietly model it as zero.
        required_columns = [
            "time",
            "target_time",
            "T_air",
            "T_out",
            "state",
            "flow_lpm",
            "T_supply",
            "T_return",
            "baseload_w",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
        ]
        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for building identification: "
                f"{missing_columns}"
            )

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600.0

        df = df[df["lead_time_hours"] >= 0.0]
        df = df.sort_values(["target_time", "lead_time_hours"])
        df = df.drop_duplicates(subset="target_time", keep="first")

        df = df.drop(columns=["time", "lead_time_hours"])
        df = df.rename(columns={"target_time": "time"})
        df = df.sort_values("time").reset_index(drop=True)

        # Rows past the last measurement survive this, and forecast() relies on
        # it: the weather forecast still covers them with real irradiance and
        # outdoor temperature, and every sensor column carries its last reading
        # forward, so nothing here is missing. Which rows those are cannot be
        # read off the frame for the same reason - forecast() is told where now
        # is instead of guessing.
        df = df.dropna(subset=["T_air", "T_out"]).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No overlapping room temperature, outdoor temperature and "
                "weather data in the requested range."
            )

        numeric_columns = [
            "T_air",
            "T_out",
            "flow_lpm",
            "T_supply",
            "T_return",
            "baseload_w",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # flow_lpm is rate-like and fetched without an InfluxDB fill, so a
        # reporting gap arrives as NaN. The compressor's own state settles what
        # a gap means - identical reasoning to
        # BoilerThermalIdentifier._bridge_flow_reporting_gaps.
        flow = df["flow_lpm"].ffill()
        df["flow_lpm"] = flow.where(df["state"] != self.states.off, 0.0).fillna(0.0)

        # Heat only reaches the floor circuit while the heat pump is actually
        # serving the space; during SWW the same flow goes to the tank instead.
        measurable = (
            df["state"].isin((self.states.heating, self.states.cooling))
            & (df["flow_lpm"] > self.MIN_FLOW_LPM)
            & df["T_supply"].notna()
            & df["T_return"].notna()
        )

        df["Q_floor_w"] = np.where(
            measurable,
            floor_heat_w(
                df["flow_lpm"].to_numpy(dtype=float),
                df["T_supply"].to_numpy(dtype=float),
                df["T_return"].to_numpy(dtype=float),
            ),
            0.0,
        )

        # dt_seconds[i] is the step ENDING at row i, matching _rollout, which
        # advances from i-1 to i over it.
        df["dt_seconds"] = df["time"].diff().dt.total_seconds()
        df.loc[df.index[0], "dt_seconds"] = df["dt_seconds"].median()

        # The step DRIVEN by row i is the one starting there, so its midpoint
        # uses the forward gap, not the backward one.
        forward_dt = df["dt_seconds"].shift(-1)
        forward_dt = forward_dt.fillna(df["dt_seconds"].median())

        df["I_facade_w_per_m2"] = facade_irradiance_w_per_m2(
            df["time"] + pd.to_timedelta(forward_dt / 2.0, unit="s"),
            direct_normal=df["direct_normal_irradiance"].to_numpy(dtype=float),
            diffuse_horizontal=df["diffuse_radiation"].to_numpy(dtype=float),
            # Open-Meteo reports the direct component on the horizontal plane
            # separately from the diffuse one, and GHI is their sum by
            # definition of global horizontal irradiance.
            global_horizontal=(
                df["direct_radiation"].fillna(0.0) + df["diffuse_radiation"].fillna(0.0)
            ).to_numpy(dtype=float),
            latitude=self.latitude,
            longitude=self.longitude,
        )

        df["shutter_open_fraction"] = self._shutter_open_fraction(df)
        df["occupants"] = self._occupants(df)
        df["baseload_w"] = df["baseload_w"].fillna(0.0)

        if "wind_speed" in df.columns:
            df["wind_m_per_s"] = (
                pd.to_numeric(df["wind_speed"], errors="coerce") / self.KMH_PER_M_PER_S
            )

        return df

    def _inputs(self, model, df: pd.DataFrame) -> np.ndarray:
        """u = [T_outdoor, Q_internal, Q_solar, Q_floor] per sample.

        Rebuilt per fit iteration because two of the four depend on identified
        parameters (the effective aperture and the in-zone baseload fraction).
        Takes the model rather than the raw parameter vector.
        """

        q_solar = solar_gain_w(
            a_eff_m2=model.a_eff_m2,
            shutter_open_fraction=df["shutter_open_fraction"].to_numpy(dtype=float),
            facade_irradiance=df["I_facade_w_per_m2"].to_numpy(dtype=float),
        )

        q_internal = internal_gain_w(
            baseload_w=df["baseload_w"].to_numpy(dtype=float),
            internal_gain_fraction=model.internal_gain_fraction,
            occupants=df["occupants"].to_numpy(dtype=float),
        )

        return np.column_stack(
            [
                df["T_out"].to_numpy(dtype=float),
                q_internal,
                q_solar,
                df["Q_floor_w"].to_numpy(dtype=float),
            ]
        )

    def _rollout_plan(
        self,
        df: pd.DataFrame,
        median_dt: float,
    ) -> tuple[list[tuple[int, int]], int, int]:
        """(contiguous runs, warm-up samples, horizon samples).

        A run is a stretch with no sampling gap. Within a run, T_air is
        re-anchored to the measurement every horizon so errors cannot accumulate
        without bound, while T_mass is carried straight through - it is never
        measured, so there is nothing to re-anchor it to, and restarting it from
        the air temperature at every window would inject a fresh error each time.
        """

        # Zero is meaningful, not a degenerate case: a structure whose only
        # state is the measurement has nothing to settle, so every window
        # scores and a six-hour gap-free stretch is already usable.
        warmup_samples = max(int(round(self.MASS_WARMUP_HOURS * 3600.0 / median_dt)), 0)
        horizon_samples = max(
            int(round(self.ROLLOUT_HORIZON_HOURS * 3600.0 / median_dt)), 2
        )

        contiguous = (
            df["dt_seconds"].to_numpy(dtype=float)
            <= self.MAX_DT_SECONDS_MULTIPLE * median_dt
        )

        runs: list[tuple[int, int]] = []
        run_start = 0

        for i in range(1, len(df) + 1):
            if i < len(df) and contiguous[i]:
                continue

            if i - run_start >= warmup_samples + horizon_samples:
                runs.append((run_start, i))

            run_start = i

        if not runs:
            raise ValueError(
                "No usable rollout windows: the data has no gap-free stretch of "
                f"{self.MASS_WARMUP_HOURS + self.ROLLOUT_HORIZON_HOURS:.0f} hours."
            )

        return runs, warmup_samples, horizon_samples

    def _simulate_windows(
        self,
        x: np.ndarray,
        df: pd.DataFrame,
        plan: tuple[list[tuple[int, int]], int, int],
        include_partial: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward-simulate every scored window; returns (predicted, measured,
        active, row indices).

        `active` marks the samples where the floor circuit was actually serving
        the space, so calibration diagnostics and validation can report the
        free-floating and actively conditioned regimes apart instead of letting
        a good result in one hide a poor one in the other.

        `include_partial` keeps a run's trailing window even when it is shorter
        than the horizon. Scoring must not: a shorter window accumulates error
        over less time, so its residuals come out systematically smaller and
        would flatter the metrics. Display must, or the curve stops up to a
        full horizon - six hours here - short of the last measurement.
        """

        runs, warmup_samples, horizon_samples = plan

        model = self._model_from_parameters(x)
        inputs = self._inputs(model, df)
        a, b = self._state_space(model)

        measured = df["T_air"].to_numpy(dtype=float)
        dt_seconds = df["dt_seconds"].to_numpy(dtype=float)
        floor_heat = df["Q_floor_w"].to_numpy(dtype=float)

        predicted_parts: list[np.ndarray] = []
        measured_parts: list[np.ndarray] = []
        active_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []

        measurement_variance = self.SENSOR_RESOLUTION_K**2 / 12.0
        observation = zone_observation(model)

        for run_start, run_end in runs:
            # Every window starts from the filter's estimate of the WHOLE state
            # at that moment, so its measured and unmeasured parts stay
            # consistent with each other. For a single-state model that is the
            # measurement, lightly smoothed; for a two-state one it is the only
            # principled way to know where the mass node is.
            estimates = kalman_states(
                a,
                b,
                measured[run_start:run_end],
                inputs[run_start:run_end],
                dt_seconds[run_start:run_end],
                self.PROCESS_NOISE_W,
                measurement_variance,
                observation,
                self.DISTURBANCE_INPUTS,
            )

            window_start = run_start

            # Two samples is the shortest window _rollout can step through.
            minimum = 2 if include_partial else horizon_samples

            while window_start + minimum <= run_end:
                window_end = min(window_start + horizon_samples, run_end)

                simulated = _rollout(
                    a,
                    b,
                    initial_state=estimates[window_start - run_start],
                    inputs=inputs[window_start:window_end],
                    dt_seconds=dt_seconds[window_start:window_end],
                )

                if window_start - run_start >= warmup_samples:
                    # What the thermostat would have read, which is what the
                    # residual is measured against.
                    predicted_parts.append(simulated @ observation)
                    measured_parts.append(measured[window_start:window_end])
                    active_parts.append(floor_heat[window_start:window_end] != 0.0)
                    index_parts.append(np.arange(window_start, window_end))

                window_start = window_end

        if not predicted_parts:
            raise ValueError(
                "No usable rollout windows left after the "
                f"{self.MASS_WARMUP_HOURS:.0f} hour warm-up."
            )

        return (
            np.concatenate(predicted_parts),
            np.concatenate(measured_parts),
            np.concatenate(active_parts),
            np.concatenate(index_parts),
        )

    def _window_residuals(
        self,
        x: np.ndarray,
        df: pd.DataFrame,
        plan: tuple[list[tuple[int, int]], int, int],
    ) -> np.ndarray:
        predicted, measured, _, _ = self._simulate_windows(x, df, plan)

        return predicted - measured

    def _bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lower, initial, upper) for the parameter vector.

        The two capacity-related bounds are derived from configuration rather
        than asserted: the air node cannot hold less than the zone's own air,
        and the effective solar aperture cannot exceed the geometric glass area
        because it is that area times a g-value and an incidence factor, both at
        most one.
        """

        if self.volume_m3 <= 0.0:
            raise ValueError(
                "building.ceiling_height and building.rooms "
                "areas must be configured: the air node's heat capacity bounds "
                "are derived from the zone's air volume, not assumed."
            )

        air_capacity = RHO_AIR_KG_PER_M3 * self.volume_m3 * CP_AIR_J_PER_KG_K

        # least_squares needs a strictly positive bound width, so a zone with no
        # configured south glass gets a numerically-zero aperture rather than a
        # degenerate bound - solar gain is then simply absent from the model.
        max_aperture = max(sum(self.glazing_areas_m2), self.MIN_F_SCALE)

        lower = np.array(
            [
                self.MIN_UA_ENVELOPE_W_PER_K,
                self.MIN_UA_AIR_MASS_W_PER_K,
                self.MIN_AIR_CAPACITY_MULTIPLE * air_capacity,
                self.MIN_C_MASS_J_PER_K,
                0.0,
                0.0,
                0.0,
            ]
        )

        upper = np.array(
            [
                self.MAX_UA_ENVELOPE_W_PER_K,
                self.MAX_UA_AIR_MASS_W_PER_K,
                self.MAX_AIR_CAPACITY_MULTIPLE * air_capacity,
                self.MAX_C_MASS_J_PER_K,
                max_aperture,
                self.MAX_SENSOR_MASS_FRACTION,
                1.0,
            ]
        )

        initial = np.array(
            [
                self.INITIAL_UA_ENVELOPE_W_PER_K,
                self.INITIAL_UA_AIR_MASS_W_PER_K,
                self.INITIAL_AIR_CAPACITY_MULTIPLE * air_capacity,
                self.INITIAL_C_MASS_J_PER_K,
                self.INITIAL_APERTURE_FRACTION * max_aperture,
                self.INITIAL_SENSOR_MASS_FRACTION,
                self.INITIAL_INTERNAL_GAIN_FRACTION,
            ]
        )

        return lower, np.clip(initial, lower, upper), upper

    def _log_drive_collinearity(self, df: pd.DataFrame) -> None:
        """Warn when solar gain and envelope loss cannot be told apart.

        Both drives peak in the afternoon, and in summer they do so partly
        together: the sun that shines through the glass is also what warmed the
        outdoor air. Where the two are strongly correlated over the calibration
        window, any split of the afternoon warming between a_eff_m2 and
        ua_envelope_w_per_k fits about equally well and the fit will hand it to
        one of them, so this number says whether the reported split means
        anything.

        It is a necessary check, not a sufficient one. On this installation's
        August/September data the correlation is only about +0.3 while a_eff_m2
        still collapsed to zero, so a low value here does not by itself prove
        the two are separable - the daily indoor swing there is under 1 K at a
        0.1 K sensor resolution, which limits what any diagnostic can resolve.

        An identification diagnostic, not a validation result: it says what the
        data can determine, never whether the model is right.
        """

        solar_drive = (df["shutter_open_fraction"] * df["I_facade_w_per_m2"]).to_numpy(
            dtype=float
        )
        envelope_drive = (df["T_out"] - df["T_air"]).to_numpy(dtype=float)

        if np.std(solar_drive) == 0.0 or np.std(envelope_drive) == 0.0:
            return

        correlation = float(np.corrcoef(solar_drive, envelope_drive)[0, 1])

        logger.info(
            "Building thermal calibration: solar and envelope drives correlate "
            "%+.2f over this window - a_eff_m2 and ua_envelope_w_per_k are "
            "separable only where this is well below 1.",
            correlation,
        )

    def calibrate(self, df: pd.DataFrame) -> BuildingThermalModel:
        df = self.prepare(df)

        median_dt = float(df["dt_seconds"].median())

        split_index = int(len(df) * self.TRAIN_RATIO)

        if split_index <= 1 or split_index >= len(df):
            raise ValueError("Invalid train/test split.")

        train_df = df.iloc[:split_index].reset_index(drop=True)

        plan = self._rollout_plan(train_df, median_dt)
        lower, initial, upper = self._bounds()

        if sum(self.glazing_areas_m2) <= 0.0:
            logger.warning(
                "Building thermal calibration: building.south_glazing is not "
                "configured, so solar gain is bounded to zero - any real solar "
                "gain will be absorbed by the envelope and capacity parameters."
            )

        def residuals(x: np.ndarray) -> np.ndarray:
            return self._window_residuals(x, train_df, plan)

        # x_scale="jac": the parameters span conductances (10^2 W/K) and heat
        # capacities (10^7 J/K), so an unscaled trust region would step
        # meaninglessly in one or the other.
        ordinary = least_squares(
            residuals, initial, bounds=(lower, upper), x_scale="jac"
        )

        # Robust loss scaled by the data's own residual spread, so a window
        # containing an unmodelled disturbance (an opened window, a visitor,
        # a wood stove) is down-weighted instead of dragging the fit.
        deviation = np.abs(ordinary.fun - np.median(ordinary.fun))
        f_scale = max(self.MAD_TO_STD * float(np.median(deviation)), self.MIN_F_SCALE)

        result = least_squares(
            residuals,
            ordinary.x,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=f_scale,
            x_scale="jac",
        )

        std_errors = self._parameter_std_errors(result)
        self.parameter_std_errors = dict(
            zip(self.PARAMETER_NAMES, std_errors, strict=True)
        )

        self.model = self._model_from_parameters(result.x)

        predicted, _, active, _ = self._simulate_windows(result.x, train_df, plan)

        logger.info(
            "Building thermal calibration: %d training points, %d validation "
            "points, %d scored rollout samples (%.0f%% with the floor circuit "
            "active), f_scale=%.4g K",
            len(train_df),
            len(df) - len(train_df),
            len(predicted),
            100.0 * float(np.mean(active)) if active.size else 0.0,
            f_scale,
        )

        self._log_drive_collinearity(train_df)

        logger.info(
            "Building thermal calibration: %s",
            " ".join(
                f"{name}={value:.4g}±{error:.4g}"
                for name, value, error in zip(
                    self.PARAMETER_NAMES, result.x, std_errors, strict=True
                )
            ),
        )

        return self.model

    @staticmethod
    def _skill(
        measured: np.ndarray,
        predicted: np.ndarray,
        persistence: np.ndarray,
    ) -> float:
        """How much of persistence's error the model removes, 0 = no better."""

        from sklearn.metrics import mean_absolute_error

        baseline = float(mean_absolute_error(measured, persistence))

        if baseline <= 0.0:
            return float("nan")

        return 1.0 - float(mean_absolute_error(measured, predicted)) / baseline

    @staticmethod
    def _trend(error: np.ndarray, *drives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Least-squares slopes of error on the drives, fitted together with an
        intercept, and their standard errors from the scatter around the fit."""

        design = np.column_stack([np.ones(len(error)), *drives])
        coefficients = np.linalg.lstsq(design, error, rcond=None)[0]
        residual = error - design @ coefficients
        variance = float(residual @ residual) / max(len(error) - design.shape[1], 1)
        covariance = variance * np.linalg.pinv(design.T @ design)

        return coefficients[1:], np.sqrt(np.diag(covariance))[1:]

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        """Forward-simulation accuracy over independent rollout windows.

        These are prediction metrics, kept strictly apart from the fit
        diagnostics reported by calibrate(): a parameter's standard error says
        how well the data determined it, and is reported here too, but it is
        never evidence that the model is physically right.
        """

        # Imported here, not at the top: the state update loads this module
        # every few minutes for estimate() and forecast() alone, and sklearn
        # was a third of its import time.
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model = self.get_model()

        df = self.prepare(df)
        median_dt = float(df["dt_seconds"].median())
        plan = self._rollout_plan(df, median_dt)

        x = self._parameters(model)

        predicted, measured, active, positions = self._simulate_windows(x, df, plan)

        metrics = {
            "scored_samples": float(len(predicted)),
            "r2": float(r2_score(measured, predicted)),
            "mae": float(mean_absolute_error(measured, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(measured, predicted))),
            # A systematic offset is what matters for planning a setpoint, and
            # it is invisible in MAE alone.
            "bias_k": float(np.mean(predicted - measured)),
        }

        _, _, horizon_samples = plan

        # Persistence: hold each window's own first measurement for that whole
        # window. Reshaping is safe because validate() never asks for the
        # partial trailing window, so every scored window has the same length.
        windows = len(predicted) // horizon_samples
        anchors = measured.reshape(windows, horizon_samples)[:, :1]
        persistence = np.repeat(anchors, horizon_samples, axis=1).ravel()

        metrics["mae_persistence"] = float(mean_absolute_error(measured, persistence))
        metrics["skill_vs_persistence"] = self._skill(measured, predicted, persistence)

        # The overall skill is an average over both regimes, and one of them
        # normally dominates by count: on cooling-season data the floor circuit
        # ran in 1.3% of quarter hours, so a headline skill is almost entirely
        # the free-floating one. Planning acts on the other regime, so it gets
        # its own baseline and its own number rather than being averaged away.
        active_windows = int(active.reshape(windows, horizon_samples).any(axis=1).sum())
        metrics["active_windows"] = float(active_windows)

        for label, mask in (("free_float", ~active), ("active", active)):
            if not mask.any():
                continue

            metrics[f"samples_{label}"] = float(mask.sum())
            metrics[f"mae_{label}"] = float(
                mean_absolute_error(measured[mask], predicted[mask])
            )
            metrics[f"bias_{label}_k"] = float(
                np.mean(predicted[mask] - measured[mask])
            )
            metrics[f"mae_persistence_{label}"] = float(
                mean_absolute_error(measured[mask], persistence[mask])
            )
            metrics[f"skill_{label}"] = self._skill(
                measured[mask], predicted[mask], persistence[mask]
            )

        skill_active = metrics.get("skill_active")

        if skill_active is None or active_windows < self.MIN_ACTIVE_WINDOWS:
            logger.warning(
                "Building thermal validation: only %d rollout window(s) "
                "contain floor activity, fewer than the %d needed to judge it "
                "- the model is effectively untested on the regime planning "
                "acts in, whatever its overall skill says.",
                active_windows,
                self.MIN_ACTIVE_WINDOWS,
            )
        elif skill_active <= self.MIN_SKILL_VS_PERSISTENCE:
            logger.warning(
                "Building thermal validation: with the floor circuit running "
                "the model scores %.3f K against %.3f K for holding the last "
                "measurement - skill %+.2f. It cannot predict the response to "
                "heating or cooling, which is the only thing planning needs.",
                metrics["mae_active"],
                metrics["mae_persistence_active"],
                skill_active,
            )

        lower, _, upper = self._bounds()

        # A parameter sitting exactly on a bound, or with a standard error
        # larger than the estimate itself, was not determined by this data. Both
        # are reported rather than hidden: summer data in particular excites the
        # envelope conductance only weakly, because indoor and outdoor
        # temperatures stay close.
        pinned = 0

        for index, name in enumerate(self.PARAMETER_NAMES):
            value = float(x[index])

            if np.isclose(value, lower[index]) or np.isclose(value, upper[index]):
                pinned += 1
                logger.warning(
                    "Building thermal validation: %s is pinned at a bound "
                    "(%.4g) - not identified by this data.",
                    name,
                    value,
                )

        metrics["pinned_parameters"] = float(pinned)

        if metrics["skill_vs_persistence"] <= self.MIN_SKILL_VS_PERSISTENCE:
            logger.warning(
                "Building thermal validation: overall the model scores %.3f K "
                "against %.3f K for simply holding the last measurement over a "
                "%.0f hour window - skill %+.2f. Its dynamics are adding "
                "error, not information.",
                metrics["mae"],
                metrics["mae_persistence"],
                self.ROLLOUT_HORIZON_HOURS,
                metrics["skill_vs_persistence"],
            )

        total_glass_m2 = sum(self.glazing_areas_m2)

        if total_glass_m2 > 0.0:
            aperture_fraction = model.a_eff_m2 / total_glass_m2
            metrics["aperture_fraction"] = aperture_fraction
            implausible = aperture_fraction < self.MIN_PLAUSIBLE_APERTURE_FRACTION
            metrics["implausible_aperture"] = float(implausible)

            if implausible:
                logger.warning(
                    "Building thermal validation: a_eff_m2 = %.2f m2 is only "
                    "%.3f of the %.1f m2 of configured south glass, which "
                    "would need a g-value no real glazing has. The fit may be "
                    "statistically converged and still not describe this "
                    "window - most likely the data covers too little sunlit "
                    "time with the shutters open.",
                    model.a_eff_m2,
                    aperture_fraction,
                    total_glass_m2,
                )

        # A model can score well on average and still respond wrongly to the one
        # input that drives it. If the envelope conductance is off, the error a
        # window ends with grows with the indoor-outdoor difference that drove
        # it - in BOTH directions, since the sign of the drive flips with the
        # season and the time of day. A slope of zero is what a correctly
        # identified envelope looks like; anything else is structural, and
        # invisible to a mean absolute error.
        drive = (df["T_out"] - df["T_air"]).to_numpy(dtype=float)[positions]
        sunlit = (df["shutter_open_fraction"] * df["I_facade_w_per_m2"]).to_numpy(
            dtype=float
        )[positions]

        window_drive = drive.reshape(windows, horizon_samples).mean(axis=1)
        window_error = (predicted - measured).reshape(windows, horizon_samples)[:, -1]

        # Dark windows only. Solar gain and the indoor-outdoor difference both
        # peak in the afternoon - on this installation they correlate about
        # +0.4 - so a trend fitted over all windows measures the NET of two
        # errors and can read clean while both are large. Measured directly:
        # over all windows the single-node model slopes +0.0009 K/K, but after
        # dark it slopes -0.0218, its oversized solar term having cancelled its
        # own envelope error. After dark the envelope stands alone.
        dark = (
            sunlit.reshape(windows, horizon_samples).max(axis=1)
            < self.NEGLIGIBLE_SOLAR_GAIN_W
        )

        metrics["dark_windows"] = float(dark.sum())

        if (
            dark.sum() >= self.MIN_WINDOWS_FOR_A_TREND
            and np.std(window_drive[dark]) > 0
        ):
            window_drive = window_drive[dark]
            window_error = window_error[dark]
            (slope,), (standard_error,) = self._trend(window_error, window_drive)
            span = float(
                np.quantile(window_drive, 0.95) - np.quantile(window_drive, 0.05)
            )

            metrics["envelope_bias_slope_k_per_k"] = slope
            metrics["envelope_bias_span_k"] = abs(slope) * span

            # A slope is only evidence if it can be told from zero. Its standard
            # error comes from the scatter around the fitted line, which is the
            # honest yardstick here - a fixed threshold in kelvin would either
            # fire on noise or hide a real trend depending on how spread out the
            # conditions happened to be.
            metrics["envelope_bias_slope_std_error"] = standard_error

            significant = standard_error > 0.0 and abs(slope) > (
                self.SIGNIFICANT_SLOPE_STD_ERRORS * standard_error
            )

            if significant and metrics["envelope_bias_span_k"] > (
                self.SENSOR_RESOLUTION_K
            ):
                logger.warning(
                    "Building thermal validation: the window error slopes "
                    "%+.4f K per K of indoor-outdoor difference, worth %.2f K "
                    "across the range seen after dark - the envelope "
                    "response is too %s. A mean error cannot show this: the "
                    "model is right on average and wrong in how it reacts to "
                    "the drive.",
                    slope,
                    metrics["envelope_bias_span_k"],
                    "weak" if slope < 0.0 else "strong",
                )

            # Wind drives air through the envelope, and the heat that carries
            # scales with wind speed times the indoor-outdoor difference. The
            # model has no such term: its envelope conductance holds the mean
            # infiltration. Were that too little, the error would slope with
            # wind x drive beyond what the drive alone explains - so both are
            # fitted together, and only the wind's own slope is judged.
            #
            # A diagnostic, not a reason to add a term: on summer data the one
            # found was ~0.34 air changes per hour per m/s, several times what
            # a closed envelope of this quality leaks - windows opened on windy
            # days, not infiltration. Only data with the windows shut (the
            # heating season) can show an infiltration term.
            if "wind_m_per_s" in df.columns:
                wind_drive = (
                    (df["wind_m_per_s"].to_numpy(dtype=float)[positions] * drive)
                    .reshape(windows, horizon_samples)
                    .mean(axis=1)[dark]
                )
                known = np.isfinite(wind_drive)

                if (
                    known.sum() >= self.MIN_WINDOWS_FOR_A_TREND
                    and np.std(wind_drive[known]) > 0
                ):
                    (_, wind_slope), (_, wind_error) = self._trend(
                        window_error[known], window_drive[known], wind_drive[known]
                    )
                    wind_span = abs(wind_slope) * float(
                        np.quantile(wind_drive[known], 0.95)
                        - np.quantile(wind_drive[known], 0.05)
                    )

                    # K of window error per K of drive per m/s of wind.
                    metrics["wind_bias_slope_k_per_k_m_s"] = float(wind_slope)
                    metrics["wind_bias_slope_std_error"] = float(wind_error)
                    metrics["wind_bias_span_k"] = wind_span

                    if (
                        abs(wind_slope) > self.SIGNIFICANT_SLOPE_STD_ERRORS * wind_error
                        and wind_span > self.SENSOR_RESOLUTION_K
                    ):
                        logger.warning(
                            "Building thermal validation: beyond the envelope "
                            "slope, the window error slopes %+.4f K per K of "
                            "indoor-outdoor difference per m/s of wind, worth "
                            "%.2f K after dark - the zone %s heat with the "
                            "wind than the model. Infiltration only if the "
                            "windows were shut; otherwise opened windows.",
                            wind_slope,
                            wind_span,
                            "loses more" if wind_slope < 0.0 else "loses less",
                        )

        if self.parameter_std_errors is not None:
            weakly_identified = 0

            for index, name in enumerate(self.PARAMETER_NAMES):
                error = self.parameter_std_errors[name]
                metrics[f"std_error_{name}"] = error

                if not np.isfinite(error) or error >= abs(float(x[index])):
                    weakly_identified += 1
                    logger.warning(
                        "Building thermal validation: %s = %.4g has a standard "
                        "error of %.4g - the data does not determine it.",
                        name,
                        float(x[index]),
                        error,
                    )

            metrics["weakly_identified_parameters"] = float(weakly_identified)

        return metrics

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicted and measured zone temperature over a window, by time.

        Returns both, because the measured column is the zone average over
        config.building.rooms - not any single room's thermostat -
        and comparing the prediction against anything else would compare two
        different quantities.

        Exactly the view validate() scores - the room temperature is re-anchored
        to the measurement every ROLLOUT_HORIZON_HOURS while the unmeasured mass
        node carries through - so what the dashboard draws is the same quantity
        the reported metrics describe, not a more flattering variant of it.

        Every input is measured. This is deliberately a reconstruction of what
        the model says the house did, not a forecast: predicting forward would
        need future shutter positions and occupancy, which nothing reports and
        which are not the model's to invent.
        """

        model = self.get_model()

        prepared = self.prepare(df)
        median_dt = float(prepared["dt_seconds"].median())
        plan = self._rollout_plan(prepared, median_dt)

        predicted, measured, _, positions = self._simulate_windows(
            self._parameters(model), prepared, plan, include_partial=True
        )

        return pd.DataFrame(
            {"predicted": predicted, "measured": measured},
            index=prepared["time"].to_numpy()[positions],
        ).sort_index()

    def estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """The filter's running estimate of every state, indexed by time.

        Unlike simulate(), this never restarts: the filter corrects at every
        step, so the result is continuous by construction and has none of the
        jumps a sequence of fixed-horizon rollouts necessarily shows. That
        makes it the right thing to draw, and for the two-node model it is the
        only way to see the mass node at all - no sensor measures the screed.

        It is also what an MPC has to start from, so this is the same quantity
        planning will consume, not a display-only variant of it.
        """

        model = self.get_model()
        prepared = self.prepare(df)

        a, b = self._state_space(model)

        estimates = kalman_states(
            a,
            b,
            prepared["T_air"].to_numpy(dtype=float),
            self._inputs(model, prepared),
            prepared["dt_seconds"].to_numpy(dtype=float),
            self.PROCESS_NOISE_W,
            self.SENSOR_RESOLUTION_K**2 / 12.0,
            zone_observation(model),
            self.DISTURBANCE_INPUTS,
        )

        names = ["air", "mass"][: estimates.shape[1]]

        return pd.DataFrame(
            estimates[:, : len(names)],
            columns=names,
            index=prepared["time"].to_numpy(),
        )

    def forecast(
        self,
        df: pd.DataFrame,
        now: datetime,
        baseload_w: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Rolls the model forward past the last measurement, indexed by time.

        Free-running: no heat is delivered to the zone, so this answers what the
        house does if nothing is done to it. There is no decision in it, which
        is why it lives here and not in the optimizer - putting it there would
        suggest something was chosen.

        It starts from the filter's estimate of the whole state at the last
        measurement (see estimate()), so the unmeasured part of that state is
        inferred rather than assumed, and it is the same starting point an MPC
        would plan from.

        Two of the inputs are genuine forecasts - irradiance and outdoor
        temperature both come from Open-Meteo - and `baseload_w` takes the
        baseload forecaster's own curve when given. The rest is each sensor's
        last reading carried forward, chiefly the shutter position. Measured
        over 160 rolling 24 hour forecasts on this installation, perfect
        knowledge of all three inputs would improve the result by 0.055 K
        against a forecast error of 0.288 K: the model is what limits this, not
        the inputs, so a shutter or occupancy forecaster would be solving the
        wrong problem.
        """

        zone, last = self._trajectory(df, now, baseload_w)

        return zone.iloc[last:]

    def trajectory(
        self,
        df: pd.DataFrame,
        now: datetime,
        baseload_w: pd.Series | None = None,
    ) -> pd.DataFrame:
        """The zone over the whole frame, indexed by time: the filter's estimate
        of every state up to now, then the free-running forecast from there (see
        forecast()), with the measured zone temperature and the gains beside
        them. One filter pass serves both halves - what a plan starts from and
        is driven by, and the history it is drawn against.
        """

        zone, _ = self._trajectory(df, now, baseload_w)

        return zone

    def _trajectory(
        self,
        df: pd.DataFrame,
        now: datetime,
        baseload_w: pd.Series | None,
    ) -> tuple[pd.DataFrame, int]:

        model = self.get_model()
        prepared = self.prepare(df)

        # Where the measurements stop cannot be read off the frame: every
        # sensor column carries its last reading forward, so a future row looks
        # exactly like a measured one. The caller says where now is.
        known = (prepared["time"] <= now).to_numpy()

        if not known.any() or known.all():
            raise ValueError(
                "No future rows to forecast: the frame must reach past the last "
                "measurement, and must contain measurements to start from."
            )

        if baseload_w is not None:
            aligned = baseload_w.reindex(prepared["time"]).to_numpy(dtype=float)
            prepared["baseload_w"] = np.where(
                np.isnan(aligned), prepared["baseload_w"], aligned
            )

        a, b = self._state_space(model)
        inputs = self._inputs(model, prepared)
        dt_seconds = prepared["dt_seconds"].to_numpy(dtype=float)

        last = int(np.flatnonzero(known)[-1])

        estimates = kalman_states(
            a,
            b,
            prepared["T_air"].to_numpy(dtype=float)[: last + 1],
            inputs[: last + 1],
            dt_seconds[: last + 1],
            self.PROCESS_NOISE_W,
            self.SENSOR_RESOLUTION_K**2 / 12.0,
            zone_observation(model),
            self.DISTURBANCE_INPUTS,
        )

        # Nothing is delivered to the zone: this is what happens if the heat
        # pump is left out of it.
        inputs = inputs.copy()
        inputs[last:, 3] = 0.0

        simulated = _rollout(
            a,
            b,
            initial_state=estimates[-1],
            inputs=inputs[last:],
            dt_seconds=dt_seconds[last:],
        )

        # The rollout starts from the filter's last estimate, so the two join
        # without a seam.
        states = np.vstack([estimates[:last], simulated])
        names = ["air", "mass"][: states.shape[1]]

        zone = pd.DataFrame(
            states[:, : len(names)],
            columns=names,
            index=prepared["time"].to_numpy(),
        )

        # What a thermostat would read of it, which is the only column that
        # can be drawn against - or scored on - the measurement (see
        # zone_observation).
        zone["reading"] = states @ zone_observation(model)
        zone["measured"] = np.where(
            known, prepared["T_air"].to_numpy(dtype=float), np.nan
        )
        # The heat no decision changes, split by where it lands, for a plan to
        # start from: the same gains the rollout was driven by.
        zone["internal_gain_w"] = inputs[:, 1]
        zone["solar_gain_w"] = inputs[:, 2]

        return zone, last

    def dataset(self, config: Config) -> DatasetDefinition:
        self.room_areas_m2 = [room.area_m2 for room in config.building.rooms]
        self.states = config.heat_pump.states
        # Derived, not configured: the same areas already drive the weighting.
        self.volume_m3 = sum(self.room_areas_m2) * config.building.ceiling_height
        self.glazing_areas_m2 = [
            glazing.glass_m2 for glazing in config.building.south_glazing
        ]
        shaded = [
            glazing
            for glazing in config.building.south_glazing
            if glazing.cover is not None
        ]
        self.shutter_areas_m2 = [glazing.glass_m2 for glazing in shaded]
        self.room_temperature_columns = [
            f"room_temperature_{i}" for i in range(len(self.room_areas_m2))
        ]
        self.shutter_columns = [f"shutter_{i}" for i in range(len(shaded))]
        self.presence_columns = [f"presence_{i}" for i in range(len(config.presence))]

        # Open-Meteo snapshots are the base frame because they are the only
        # source of irradiance. The sensor uses the minutely_15 endpoint, so its
        # horizon already lands on the 15 minute model grid; taking one snapshot
        # per hour keeps the row count bounded while still giving every valid
        # time a forecast at most an hour old.
        #
        # target_shift corrects a real misalignment: Open-Meteo documents the
        # radiation fields as the mean over the PRECEDING 15 minutes, so the
        # value stamped at t describes [t-15min, t], while zero-order hold needs
        # the mean over [t, t+15min] to drive the step starting at t. Shifting
        # those columns one row earlier lines them up. Temperature is
        # instantaneous in the same API and must NOT be shifted with them.
        builder = DatasetBuilder().attribute_timeseries(
            "weather",
            config.forecast.open_meteo,
            attributes=[
                "direct_radiation",
                "diffuse_radiation",
                "direct_normal_irradiance",
                "temperature",
                # Instantaneous like temperature, so not shifted either. Read
                # by validate()'s wind diagnostic only - the model has no
                # infiltration term (see there).
                "wind_speed",
            ],
            interval="1h",
            aggregation="last",
            target_interval="15min",
            target_shift=[
                "direct_radiation",
                "diffuse_radiation",
                "direct_normal_irradiance",
            ],
        )

        # Numeric series are averaged onto a 5 minute grid first and only then
        # onto the 15 minute model grid: with fill="previous" that makes each
        # model step a time-weighted average of a sensor that only reports on
        # change, rather than a mean over however many raw points happened to
        # land in the step.
        #
        # Temperatures are continuous and only reported on change, so the
        # previous reading is the current one; flow is rate-like and physically
        # drops to zero when the pump stops, so a reporting gap must stay a gap
        # for prepare() to resolve against the compressor state.
        numeric: list[tuple[str, SensorReference, Aggregation, FillMethod]] = [
            ("T_air", config.building.thermostat.temperature, "mean", "previous"),
            *[
                (name, room.temperature, "mean", "previous")
                for name, room in zip(
                    self.room_temperature_columns,
                    config.building.rooms,
                    strict=True,
                )
            ],
            ("T_supply", config.heat_pump.supply_temperature, "mean", "previous"),
            ("T_return", config.heat_pump.return_temperature, "mean", "previous"),
            ("flow_lpm", config.heat_pump.flow, "mean", "none"),
            ("baseload_w", config.baseload, "mean", "previous"),
        ]

        if config.heat_pump.outdoor_temperature is not None:
            numeric.append(
                ("T_out", config.heat_pump.outdoor_temperature, "mean", "previous")
            )

        # Cover position is an event-driven state - it only changes when a
        # shutter actually moves - so it is carried forward, not averaged.
        shutters: list[tuple[str, SensorReference, Aggregation, FillMethod]] = [
            (name, glazing.cover, "last", "previous")
            for name, glazing in zip(self.shutter_columns, shaded, strict=True)
            if glazing.cover is not None
        ]
        numeric += shutters

        # The compressor state and the device trackers arrive as strings, which
        # cannot be resampled by averaging: they are taken directly on the model
        # grid, keeping the value in force at the step's end.
        textual: list[tuple[str, SensorReference]] = [
            ("state", config.heat_pump.state),
            *zip(self.presence_columns, config.presence, strict=True),
        ]

        for name, sensor, aggregation, fill in numeric:
            builder = builder.timeseries(
                name,
                sensor,
                interval="5m",
                aggregation=aggregation,
                fill=fill,
                target_interval="15min",
            )

        for name, sensor in textual:
            builder = builder.timeseries(
                name,
                sensor,
                interval="15m",
                aggregation="last",
                fill="previous",
            )

        for name in [entry[0] for entry in numeric + textual]:
            builder = builder.join(
                left="weather",
                right=name,
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )

        return builder.build()
