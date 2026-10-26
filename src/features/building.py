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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pvlib import irradiance, solarposition
from scipy.optimize import least_squares

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.dynamics import ScheduledSystem, discretize_zoh, kalman_states
from domain.models import BuildingThermalModel
from domain.physics import (
    CP_AIR_J_PER_KG_K,
    Q_PERSON_SENSIBLE_W,
    RHO_AIR_KG_PER_M3,
    ZONE_SLAB_STATES,
    envelope_conductance_w_per_k,
    extension_shaded_fraction,
    floor_heat_moved_w,
    floor_heat_w,
    night_share,
    floor_living_fraction,
    floor_loops_conductance_w_per_k,
    internal_gain_w,
    solar_gain_w,
    zone_floor,
    zone_observation,
    zone_observations,
    zone_slab,
    zone_state_space,
)
from domain.sensors import Aggregation, FillMethod, SensorReference
from domain.time import LOCAL_TIMEZONE, to_local_time
from features.dataset import DatasetBuilder
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)

# The modelled glazing sits in a vertical facade (pvlib's tilt from horizontal);
# which way it faces is configured (BuildingConfig.facade_azimuth).
VERTICAL_FACADE_TILT_DEG = 90.0


def facade_irradiance_w_per_m2(
    interval_midpoints: pd.Series,
    direct_horizontal: np.ndarray,
    diffuse_horizontal: np.ndarray,
    latitude: float,
    longitude: float,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plane-of-array irradiance on the vertical facade facing azimuth_deg
    (degrees clockwise from north), in W/m2: the total and its direct part,
    with the sun's horizontal angle from the facade's normal (degrees, positive
    towards the west) - what side walls shade the direct part by (see
    physics.extension_shaded_fraction).

    The Open-Meteo `global_tilted_irradiance` attribute already in the config is
    computed for the PV array's own tilt and azimuth, so it does not describe
    this facade; the transposition is redone here from the horizontal direct
    and diffuse components Open-Meteo reports independently of any surface.

    The beam's normal intensity follows from its horizontal share by geometry,
    DNI = B_horizontal / cos(zenith) (pvlib's irradiance.dni, which zeroes it
    with the sun near the horizon): the same as Open-Meteo's own DNI to within
    a watt per square metre on this installation (2106 daytime quarter hours,
    ratio 0.999). Derived rather than read, because that attribute was first
    stored three days after the others, and those days' missing DNI read as no
    sun at all on the facade - three hot days the fit and the zone's rollouts
    then saw without sun.

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

    zenith = position["apparent_zenith"].to_numpy()
    # Global horizontal irradiance is the direct and diffuse sum by definition.
    global_horizontal = np.nan_to_num(direct_horizontal) + np.nan_to_num(
        diffuse_horizontal
    )
    total = irradiance.get_total_irradiance(
        surface_tilt=VERTICAL_FACADE_TILT_DEG,
        surface_azimuth=azimuth_deg,
        solar_zenith=zenith,
        solar_azimuth=position["azimuth"].to_numpy(),
        dni=irradiance.dni(global_horizontal, diffuse_horizontal, zenith),
        ghi=global_horizontal,
        dhi=diffuse_horizontal,
        model="isotropic",
    )

    # Wrapped into -180..180, so east of the normal is negative on any facade.
    sun_from_normal_deg = (
        position["azimuth"].to_numpy() - azimuth_deg + 180.0
    ) % 360.0 - 180.0

    return (
        np.nan_to_num(np.asarray(total["poa_global"], dtype=float), nan=0.0),
        np.nan_to_num(np.asarray(total["poa_direct"], dtype=float), nan=0.0),
        sun_from_normal_deg,
    )


def _asleep_in_rest(model: BuildingThermalModel) -> float:
    """The share of the occupants' heat in the rest while they sleep: by floor
    area for a model saved without one."""

    if model.asleep_in_rest_fraction is None:
        return 1.0 - model.living_area_fraction

    return model.asleep_in_rest_fraction


def _rollout(
    a: np.ndarray,
    b: np.ndarray,
    initial_state: np.ndarray,
    inputs: np.ndarray,
    dt_seconds: np.ndarray,
    schedule: ScheduledSystem | None = None,
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

    if schedule is not None:
        scheduled = schedule.steps(np.asarray(dt_seconds, float)[1:])

    for i in range(1, n):
        dt = float(dt_seconds[i])
        key = round(dt, 3)

        if schedule is not None:
            a_d, b_d = scheduled[0][i - 1], scheduled[1][i - 1]
        else:
            if key not in cache:
                cache[key] = discretize_zoh(a, b, dt)

            a_d, b_d = cache[key]

        simulated[i] = a_d @ simulated[i - 1] + b_d @ inputs[i - 1]

    return simulated


def _window(
    schedule: ScheduledSystem | None, start: int, end: int
) -> ScheduledSystem | None:
    """A schedule's steps start to end, or none for a constant system."""

    return None if schedule is None else schedule.window(start, end)


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

    # How far back forecast() takes the household's weekly habit from - who
    # is home, and the shutters that follow them (see forecast()). Three weeks
    # gives each weekday three days to take a median over, so one odd week -
    # a holiday, a day off - does not set it. Not a sharp choice for the
    # shutters (13 Sep - 16 Oct 2026: the thermostat room's sun 119, 104, 107
    # and 109 W off over six hours from two, three, four and six weeks); for
    # who is home two weeks is a mean of two days, which a holiday carried for
    # a fortnight after it (21 Aug - 16 Oct: 0.39 people wrong over a day
    # against 0.27 from three weeks, 0.27 from four).
    HABIT_HISTORY_DAYS = 21

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

    # From the floor slab's surface to the room, per m2 of floor: EN 1264's
    # surface coefficient (about 7 W/m2K cooling, 10.8 heating) in series with
    # the screed and covering above the pipes, which take it down to some 2.5.
    MIN_UA_AIR_MASS_W_PER_M2_K = 2.5
    MAX_UA_AIR_MASS_W_PER_M2_K = 11.0
    INITIAL_UA_AIR_MASS_W_PER_M2_K = 6.0

    # The room node per m2 of floor: at least its air (rho*V*cp, from the
    # configured volume); at most air, furnishings (some 10 kg/m2 of wood, 25
    # kJ/m2K) and internal walls - about a square metre of 10 cm calcium silicate
    # brick per square metre of floor, 160 kJ/m2K - with room to spare. The
    # walls belong here, not with the slab: they follow the room air closely
    # (a three-node fit coupled them to it at its 2000 W/K ceiling), while the
    # slab exchanges with the room through its surface only.
    MAX_AIR_CAPACITY_J_PER_M2_K = 250.0e3
    INITIAL_AIR_CAPACITY_J_PER_M2_K = 100.0e3

    # The floor slab per m2: at least 4 cm of screed (2000 kg/m3, 850 J/kgK);
    # at most 8 cm of screed on 26 cm of concrete (2200 and 2400 kg/m3,
    # 1000 J/kgK) - a new build's ground floor, insulated below.
    MIN_C_MASS_J_PER_M2_K = 68.0e3
    MAX_C_MASS_J_PER_M2_K = 800.0e3
    INITIAL_C_MASS_J_PER_M2_K = 400.0e3

    # Fraction of the configured south glass area used as the starting guess
    # for the effective aperture: a typical double-glazing g-value times a
    # yearly-average incidence/soiling factor lands near half the geometric
    # area. The bounds themselves are [0, MAX_APERTURE_FRACTION] of it.
    INITIAL_APERTURE_FRACTION = 0.5

    # The most of its glass area an aperture can be: clear double glazing's
    # g-value of about 0.75 times a frame factor of 0.8 (HR++ and triple
    # glazing pass less). Bounded at the glass area itself, the rest's aperture
    # sat at 0.6-0.65 of it (Jul-Oct 2026), standing in for heat that reaches
    # the bedrooms some other way. It now sits on this bound, which validation
    # reports as pinned: the missing heat shows instead of being absorbed, and
    # October's prediction changed by under 0.01 K.
    #
    # The neighbours' extensions shade the thermostat's room's glazing (see
    # extension_east_depth_ratio). Fitted before the glass areas were measured
    # and the sun was corrected by the PV array, the west depth went from 0.8
    # to its bound with three more days of data; since, it fits at 1.6, 2.1 and
    # 2.8 of the glazing's width on three windows (Jul-Oct 2026, each within
    # the others' standard error) and east at 0. Shutter moves agree: closed
    # with the sun over 30 degrees west, the room did not cool. It makes the
    # afternoon right (+0.31 to -0.01 K at 17:00 on sunny October days) and
    # midday colder, the room's quick warming in the sun that it no longer
    # hides (0.214 to 0.221 K over the week): another cause, not this one.
    MAX_APERTURE_FRACTION = 0.6

    # A wall thermostat exchanges longwave radiation with the surfaces around
    # it, so it reads an operative temperature between the room and the floor
    # rather than the room alone (see BuildingThermalModel.sensor_mass_fraction).
    # 0.5 is the textbook average in still air, and the ceiling here: the sensor
    # sits in moving room air, so it cannot follow a surface more closely than
    # evenly. Not yet settled by the data: 0 on either half of July-October,
    # the 0.5 ceiling on August-September alone, 0.28 on July-September. Held
    # at 0 instead (Oct 2026), the room capacity went to its bound to make up
    # for it, so it stays fitted.
    MAX_SENSOR_MASS_FRACTION = 0.5
    INITIAL_SENSOR_MASS_FRACTION = 0.25

    # An unmodelled heat flow can enter any node: a room through ventilation,
    # a stove or a visitor - one room's window, not the other's - the slabs
    # through a calorimeter that reads somewhat off. Each room's sun and the
    # floor's heat are the input channels that land there, so the filter
    # carries a disturbance per room and one for the slabs instead of claiming
    # either is driven exactly as measured.
    DISTURBANCE_INPUTS = (2, 3, 4)

    # Home Assistant's hvac mode of a room thermostat that has closed its
    # floor loops (see Room.climate).
    CLIMATE_OFF_STATE = "off"

    # Between the two rooms (W/K): at least what a closed door and an
    # insulated floor between them still pass; at most an open stairwell and
    # an uninsulated concrete ceiling, some 2.5 W/m2K over its whole area, with
    # room to spare.
    MIN_UA_ROOMS_W_PER_K = 1.0
    MAX_UA_ROOMS_W_PER_K = 1000.0
    INITIAL_UA_ROOMS_W_PER_K = 100.0

    # The rest's slab to the living room below it, per m2 of the living room's
    # ceiling: at most a bare concrete one, the surface resistance for heat
    # flowing down (EN ISO 6946: 0.17 m2K/W) the whole of it - some 6 W/m2K;
    # zero is no floor of the rest's above it. Fitted at 2.8-4.9 W/m2K on
    # three thirds of June-October 2026, each within 0.4: the heat between
    # the floors goes through it, the rooms' own coupling stays at its bound.
    # With it the living room was predicted to 0.120 K against 0.123 K on the
    # thirds left out, better on each.
    MAX_UA_CEILING_W_PER_M2_K = 6.0
    INITIAL_UA_CEILING_W_PER_M2_K = 3.0

    # The windows opened overnight (see BuildingThermalModel.ua_window_w_per_k):
    # at most 300 W/K, some 900 m3/h - several wide open, the house aired
    # through at two to three air changes an hour. Half the nights open on a
    # day between 10 degC, still heating, and 25 degC, a hot summer's day; the
    # occupants' choice spread over at least 0.5 K, a sharper switch than a
    # day's mean resolves from one night to the next, and at most 5 K, past
    # which it is the season-long ramp the data rejects. Started where the
    # filter's missing loss puts them. Fitted on Jun-Oct 2026 with two weeks
    # held out (four folds): half the nights at 17.9-18.9 degC (+-0.2),
    # spread 0.9-1.2 K, the windows shut 71-97 W/K. Held out, the living
    # room 24 h ahead 0.205 -> 0.153 K (Oct), 0.205 -> 0.167 (Jul-Aug),
    # 0.258 -> 0.188 (Jun), 0.144 -> 0.153 (Sep); the rest better in Oct and
    # Jul-Aug, worse in Jun (too cold, 0.244 -> 0.330) and Sep. A share of
    # the windows of their own in the living room fitted 0.56-0.65, no gain;
    # each room its own windows (conductance, half day, hours) within 0.008 K
    # held out, both rooms half open at the same day (17.8-19.5 degC), the
    # rest's opened earlier (20-22 h) - no gain for four parameters more.
    MAX_UA_WINDOW_W_PER_K = 300.0
    INITIAL_UA_WINDOW_W_PER_K = 50.0
    MIN_WINDOW_HALF_DAY_C = 10.0
    MAX_WINDOW_HALF_DAY_C = 25.0
    INITIAL_WINDOW_HALF_DAY_C = 17.0
    MIN_WINDOW_SPREAD_K = 0.5
    MAX_WINDOW_SPREAD_K = 5.0
    INITIAL_WINDOW_SPREAD_K = 1.5
    # The household's own hours of opening and closing them: opened in the
    # evening or night, from 16 h to 4 h, and closed in the morning, from 3 to
    # 12 h - the windows of a night. Their times spread from night to night
    # over at least a quarter of an hour, the data's own step, and at most
    # 3 h. Started at the hours the filter's missing loss shows (23 and 7).
    # On all of Jun-Oct 2026 opened at 21:50 (+-15 min) and closed at 05:35
    # (+-5 min), 249 +-19 W/K open; in three folds of four within the hour
    # of it, in the fourth (Jul-Aug held out) a shorter night at the bound
    # of 300 W/K - the same heat out. Held out as well as hours fixed at 23
    # to 7 h, within 0.01 K.
    MIN_WINDOW_OPEN_HOUR = 16.0
    MAX_WINDOW_OPEN_HOUR = 28.0
    INITIAL_WINDOW_OPEN_HOUR = 23.0
    MIN_WINDOW_CLOSE_HOUR = 3.0
    MAX_WINDOW_CLOSE_HOUR = 12.0
    INITIAL_WINDOW_CLOSE_HOUR = 7.0
    MIN_WINDOW_HOUR_SPREAD_H = 0.25
    MAX_WINDOW_HOUR_SPREAD_H = 3.0
    INITIAL_WINDOW_HOUR_SPREAD_H = 0.5
    # The day the windows follow: the mean outdoor temperature over the last
    # 24 hours, known ahead from the forecast as well.
    WINDOW_DAY = "24h"

    # The household's own hours of sleep, in the bedrooms in the rest of the
    # house (see BuildingThermalModel.asleep_in_rest_fraction); by day their
    # heat lands by floor area. Read off the household's electricity, not the
    # rooms' temperatures (see _measured_sleep): fitted to those the hours ran
    # to their bounds (+-15-23 h) and the share asleep upstairs to zero - an
    # evening's heat in the living room, not sleep. To bed from 19 h to 4 h,
    # up from 4 to 11 h - a night's sleep; spread from night to night over at
    # least a quarter of an hour, the data's own step, and at most 3 h. On
    # Jun-Oct 2026 to bed at 23:15-23:30 and up at 06:30-07:25 month by month
    # (August, the holidays, 00:30), 130 W asleep against 190 W awake. With
    # hours fixed at 23 to 7 h, fitted on two folds (to 15 Oct / from 15 Sep
    # 2026) the share asleep in the rest came out 0.89 on both (std error
    # 0.05) and the same, 0.87-0.90, from 22 or 0 h and to 7 or 8 h. Held out,
    # the living room was predicted to 0.084/0.106 K against 0.091/0.114 (6 h)
    # and 0.161/0.201 against 0.179/0.212 (24 h), the rest to 0.096/0.143
    # against 0.089/0.137: what it moves into the bedrooms the rest's other
    # parameters take back. A share by day as well fitted 0.71 and 0.92, not
    # identified.
    MIN_ASLEEP_FROM_HOUR = 19.0
    MAX_ASLEEP_FROM_HOUR = 28.0
    INITIAL_ASLEEP_FROM_HOUR = 23.0
    MIN_ASLEEP_UNTIL_HOUR = 4.0
    MAX_ASLEEP_UNTIL_HOUR = 11.0
    INITIAL_ASLEEP_UNTIL_HOUR = 7.0
    MIN_ASLEEP_SPREAD_H = 0.25
    MAX_ASLEEP_SPREAD_H = 3.0
    INITIAL_ASLEEP_SPREAD_H = 0.5

    # The screed (see _fit_screed), per m2 of floor: 1 to 15 cm of it (some
    # 2000 kg/m3 at 850 J/kgK), and at most half the slabs; the pipes' UA to
    # it and from it down to the concrete, anything a floor's pipes and
    # layers conduct. A loose physical range, not an expectation. With it the
    # floor's heat held out of a run was predicted to 207 W against 424 W for
    # one conductance to the slabs (33 runs, Jun-Oct 2026), which read too
    # little at a run's start and too much later; and the building with the
    # screed at the floor's surface predicted the living room as well
    # (0.120 K) and its floor runs better (0.117 against 0.136 K) on thirds
    # left out.
    MIN_SCREED_J_PER_M2_K = 20.0e3
    MAX_SCREED_J_PER_M2_K = 300.0e3
    INITIAL_SCREED_J_PER_M2_K = 100.0e3
    MIN_SCREED_W_PER_M2_K = 1.0
    MAX_SCREED_W_PER_M2_K = 100.0
    INITIAL_SCREED_WATER_W_PER_M2_K = 10.0
    INITIAL_SCREED_DOWN_W_PER_M2_K = 20.0
    # Three parameters, each read off a different part of a run - its start,
    # how fast that falls, where it settles: a few runs before they separate.
    MIN_SCREED_RUNS = 5

    # Share of the baseload dissipated inside the zone. The zone is the whole
    # dwelling and the baseload every appliance but the heat pump, so nearly
    # all of it: only hot water down the drain and appliances outside leave.
    # Yet it fits at 0.4 (Jul-Oct 2026), and bounded at 0.8 it sat on that
    # bound while the envelope, the slab and the sensor fraction all moved:
    # something the baseload goes with - plausibly the cooker hood and the
    # bathroom's ventilation boost - carries heat out that the model lacks.
    # Left at [0, 1] until that is modelled, rather than hide it elsewhere.
    INITIAL_INTERNAL_GAIN_FRACTION = 0.5

    def __init__(self, latitude: float, longitude: float) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        # Overwritten by dataset() from ClimateConfig once available; the
        # defaults only keep prepare()/calibrate() callable on their own.
        self.volume_m3: float = 0.0
        self.glazing_areas_m2: list[float] = []
        self.facade_azimuth_deg = 180.0
        self.shutter_areas_m2: list[float] = []
        self.shutter_closed_positions: list[float] = []
        self.shutter_sensors: list[SensorReference] = []
        self.presence_sensors: list[SensorReference] = []
        self.room_temperature_columns: list[str] = []
        self.room_areas_m2: list[float] = []
        # Each room's floor thermostat mode (see Room.climate), aligned with
        # room_areas_m2; None for a room whose loops are always open.
        self.room_climate_columns: list[str | None] = []
        self.shutter_columns: list[str] = []
        # Which glazing, and which shutter, is in the thermostat's room
        # (SouthGlazing.thermostat_room), aligned with the lists above.
        self.glazing_in_thermostat_room: list[bool] = []
        self.shutter_in_thermostat_room: list[bool] = []
        # The room the thermostat is in, as a room_temperature_columns index.
        self.thermostat_room_index: int = 0
        self.presence_columns: list[str] = []
        self.states = HeatPumpStates()
        self.parameter_std_errors: dict[str, float] | None = None
        # See BuildingThermalModel.pv_power_per_irradiance_m2: set by
        # calibrate(), taken from the model by load().
        self.pv_power_per_irradiance_m2 = 0.0
        # The extensions configured beside the thermostat's room's glazing
        # (see SouthGlazing.extensions): side -> depth over its width. Fixed,
        # not fitted.
        self.extensions: dict[str, float] = {}
        # The floor's screed, (capacity J/K, conductance to its slab W/K), read
        # off the floor's runs before each fit (see _fit_screed) and held
        # through it; none until a model to read it against has been saved.
        self.screed: tuple[float, float] = (0.0, 0.0)
        # The household's hours of sleep (to bed, up, spread - see
        # _measured_sleep), read off its electricity before each fit; until
        # then a model saved before them.
        self.sleep: tuple[float, float, float] = (
            BuildingThermalModel.asleep_from_hour,
            BuildingThermalModel.asleep_until_hour,
            BuildingThermalModel.asleep_spread_h,
        )

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

    # A shut external shutter in front of glazing passes roughly 0.05-0.15 of
    # the glazing's own solar gain - the darker the slats, the more they absorb
    # and pass on. Twice the top of that range bounds it. Real data, Jul-Oct
    # 2026 with cover.woonkamer down at 18-20: 0.17, and 0.12-0.20 over 8
    # cross-validated splits. Without the term the rest's aperture sat at all
    # its glass, standing in for the heat its shut shutters pass, and read as
    # 18% open the living room's shutter did the same.
    MAX_CLOSED_SHUTTER_GAIN_FRACTION = 0.3
    INITIAL_CLOSED_SHUTTER_GAIN_FRACTION = 0.1

    # An extension's depth over the width of the glazing beside it. A Dutch
    # extension built without a permit reaches 4 m deep (Bbl), a ground-floor
    # window or sliding door is some 2-5 m wide: a ratio up to 2, and the
    # glazing set back in its reveal deepens it a little. Twice that bounds it.
    # Started at 0.5, the wall's shadow across the glazing at 63 degrees off
    # the facade; zero, no extension there, is a result the fit can reach.
    MAX_EXTENSION_DEPTH_RATIO = 4.0
    INITIAL_EXTENSION_DEPTH_RATIO = 0.5

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

    # The PV array measures the sun the weather forecast only predicts. Above
    # this irradiance on its plane its output is proportional to the light,
    # clear of its inverter's start-up and low-light losses: the hours its
    # output per W/m2 is taken from.
    PV_CALIBRATION_MIN_IRRADIANCE_W_PER_M2 = 200.0
    # Below this the forecast's plane irradiance is twilight or deep overcast,
    # and measured over forecast is a ratio of two small numbers that would
    # multiply a few watts of error many times: the forecast stands there.
    PV_CORRECTION_MIN_IRRADIANCE_W_PER_M2 = 50.0

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
        "a_eff_rest_m2",
        "ua_rooms_w_per_k",
        "living_envelope_fraction",
        "closed_shutter_gain_fraction",
        "extension_east_depth_ratio",
        "extension_west_depth_ratio",
        "ua_ceiling_w_per_k",
        "asleep_in_rest_fraction",
        "ua_window_w_per_k",
        "window_half_day_c",
        "window_spread_k",
        "window_open_hour",
        "window_close_hour",
        "window_hour_spread_h",
    )

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
                model.a_eff_rest_m2,
                model.ua_rooms_w_per_k,
                model.living_envelope_fraction,
                model.closed_shutter_gain_fraction,
                model.extension_east_depth_ratio,
                model.extension_west_depth_ratio,
                model.ua_ceiling_w_per_k,
                _asleep_in_rest(model),
                model.ua_window_w_per_k,
                model.window_half_day_c,
                model.window_spread_k,
                model.window_open_hour,
                model.window_close_hour,
                model.window_hour_spread_h,
            ]
        )

    @property
    def living_area_fraction(self) -> float:
        """The thermostat's room's share of the zone's floor area."""

        total = sum(self.room_areas_m2)

        if total <= 0.0:
            return 1.0

        return self.room_areas_m2[self.thermostat_room_index] / total

    def _model_from_parameters(self, x: np.ndarray) -> BuildingThermalModel:
        return BuildingThermalModel(
            ua_envelope_w_per_k=float(x[0]),
            ua_air_mass_w_per_k=float(x[1]),
            c_air_j_per_k=float(x[2]),
            c_mass_j_per_k=float(x[3]),
            a_eff_m2=float(x[4]),
            sensor_mass_fraction=float(x[5]),
            internal_gain_fraction=float(x[6]),
            a_eff_rest_m2=float(x[7]),
            ua_rooms_w_per_k=float(x[8]),
            living_envelope_fraction=float(x[9]),
            living_area_fraction=self.living_area_fraction,
            closed_shutter_gain_fraction=float(x[10]),
            pv_power_per_irradiance_m2=self.pv_power_per_irradiance_m2,
            extension_east_depth_ratio=float(x[11]),
            extension_west_depth_ratio=float(x[12]),
            ua_ceiling_w_per_k=float(x[13]),
            asleep_in_rest_fraction=float(x[14]),
            ua_window_w_per_k=float(x[15]),
            window_half_day_c=float(x[16]),
            window_spread_k=float(x[17]),
            window_open_hour=float(x[18]),
            window_close_hour=float(x[19]),
            window_hour_spread_h=float(x[20]),
            asleep_from_hour=self.sleep[0],
            asleep_until_hour=self.sleep[1],
            asleep_spread_h=self.sleep[2],
            screed_capacity_j_per_k=self.screed[0],
            screed_to_slab_w_per_k=self.screed[1],
        )

    def load(self, path) -> None:
        super().load(path)

        # A model saved before the zone had two rooms, or before shut shutters
        # passed heat, lacks parameters: as good as none until recalibrated.
        if self.model is not None and "closed_shutter_gain_fraction" not in vars(
            self.model
        ):
            logger.warning("Building model is out of date - recalibrate it")
            self.model = None

        if self.model is not None:
            self.pv_power_per_irradiance_m2 = self.model.pv_power_per_irradiance_m2
            self.screed = (
                self.model.screed_capacity_j_per_k,
                self.model.screed_to_slab_w_per_k,
            )
            self.sleep = (
                self.model.asleep_from_hour,
                self.model.asleep_until_hour,
                self.model.asleep_spread_h,
            )

    def _shutter_open_fraction(
        self, df: pd.DataFrame, thermostat_room: bool | None = None
    ) -> pd.Series:
        """Area-weighted unshaded fraction of the zone's south glazing - of
        the thermostat's room's alone, or the rest's, with `thermostat_room`.

        sum(area_i * open_i) / sum(area_i): shading is an area effect, so a
        small bedroom window may not carry the same weight as a large living
        room screen. Confirmed necessary on this installation's data - the four
        south shutters move almost independently, and an unweighted mean differs
        from this by a median of 8.8 percentage points.

        Glazing without a cover contributes its full area as permanently
        unshaded, which is what keeps this a fraction of the TOTAL south glass
        that a_eff_m2 is bounded by.
        """

        def chosen(flags: list[bool], count: int) -> list[bool]:
            if thermostat_room is None:
                return [True] * count

            return [f == thermostat_room for f in flags or [True] * count]

        glazing = chosen(self.glazing_in_thermostat_room, len(self.glazing_areas_m2))
        shutters = chosen(self.shutter_in_thermostat_room, len(self.shutter_columns))
        total_area = sum(
            a for a, c in zip(self.glazing_areas_m2, glazing, strict=True) if c
        )

        if total_area <= 0.0:
            return pd.Series(1.0, index=df.index)

        unshaded_area = total_area - sum(
            a for a, c in zip(self.shutter_areas_m2, shutters, strict=True) if c
        )
        weighted = pd.Series(unshaded_area, index=df.index)

        for column, area, closed, included in zip(
            self.shutter_columns,
            self.shutter_areas_m2,
            self.shutter_closed_positions or [0.0] * len(self.shutter_columns),
            shutters,
            strict=True,
        ):
            if not included:
                continue

            position = pd.to_numeric(df[column], errors="coerce")
            # A missing position reading must not silently mean "shut": an
            # absent cover reading says nothing about the glass, and assuming
            # full shading would attribute real solar gain to the envelope.
            open_fraction = (
                (position - closed) / (self.FULLY_OPEN_POSITION - closed)
            ).fillna(1.0)
            weighted = weighted + area * open_fraction.clip(0.0, 1.0)

        return (weighted / total_area).clip(0.0, 1.0)

    def _measured_sky(self, df: pd.DataFrame) -> np.ndarray:
        """The sky the PV array measured over the one the forecast predicted:
        its output over what it gives at the forecast's irradiance on its
        plane, where it was measured. The array and the facade share the sky
        (and, on this terraced house, their bearing), so the facade's forecast
        irradiance is scaled by it: on 4 Oct 2026 the morning sun was 2-4
        times the forecast's, and the zone was fitted to forecast sun.

        One where there is no measurement - ahead of now without a PV forecast
        (see forecast()), a reporting gap, no plane irradiance before
        Open-Meteo reported it - or too little forecast light to divide by (see
        PV_CORRECTION_MIN_IRRADIANCE_W_PER_M2). Carrying
        the last measured sky on into the forecast, fading to it over 0.5-2
        hours, did worse at every fade (105 forecasts, Sep-Oct 2026: 0.128 K
        over the first two hours against 0.131-0.138): the hourly forecast has
        already moved on with the clouds the measurement saw.
        """

        plane = df["global_tilted_irradiance"].to_numpy(dtype=float)
        measured = df["pv_w"].to_numpy(dtype=float)
        valid = (
            (self.pv_power_per_irradiance_m2 > 0.0)
            & (plane >= self.PV_CORRECTION_MIN_IRRADIANCE_W_PER_M2)
            & np.isfinite(measured)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = measured / (self.pv_power_per_irradiance_m2 * plane)

        return np.where(valid, ratio, 1.0)

    def _pv_power_per_irradiance(self, df: pd.DataFrame) -> float:
        """The array's output per W/m2 on its plane over the frame's sunny
        hours (see PV_CALIBRATION_MIN_IRRADIANCE_W_PER_M2): a median, since
        the forecast those hours are measured against is itself off at times.
        Zero without any."""

        plane = df["global_tilted_irradiance"].to_numpy(dtype=float)
        measured = df["pv_w"].to_numpy(dtype=float)
        sunny = (plane >= self.PV_CALIBRATION_MIN_IRRADIANCE_W_PER_M2) & np.isfinite(
            measured
        )

        if not sunny.any():
            return 0.0

        return float(np.median(measured[sunny] / plane[sunny]))

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

            def mean(columns: list[str]) -> pd.Series:
                available = rooms[columns].notna() * weights[columns]
                return (rooms[columns] * weights[columns]).sum(axis=1) / available.sum(
                    axis=1
                )

            # The model's two rooms (see BuildingThermalModel): the thermostat's,
            # and the rest averaged the same way.
            living = self.room_temperature_columns[self.thermostat_room_index]
            rest = [c for c in self.room_temperature_columns if c != living]
            df["T_air"] = mean(self.room_temperature_columns)
            df["T_living"] = rooms[living]
            df["T_rest"] = mean(rest)

        # A step's state is the one at its end and its readings their mean, as
        # they were before the samples were loaded (see dataset).
        df["state"] = self._samples(df, "state").ffill(axis=1).iloc[:, -1]

        for name in ("T_supply", "T_return", "flow_lpm"):
            df[name] = (
                self._samples(df, name).apply(pd.to_numeric, errors="coerce").mean(1)
            )

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
        # The rest of the house may go unmeasured for a while - a sensor that
        # has dropped out - which the filter and the fit simply skip; the
        # thermostat's room is what the zone is about.
        df = df.dropna(subset=["T_living", "T_out"]).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No overlapping room temperature, outdoor temperature and "
                "weather data in the requested range."
            )

        numeric_columns = [
            "T_air",
            "T_living",
            "T_rest",
            "T_out",
            "flow_lpm",
            "T_supply",
            "T_return",
            "baseload_w",
            "direct_radiation",
            "diffuse_radiation",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # The day the windows follow (see WINDOW_DAY): over the frame's rows
        # before each, so a forecast's day takes in the forecast.
        df["T_out_day"] = (
            df.set_index("time")["T_out"]
            .rolling(self.WINDOW_DAY, min_periods=1)
            .mean()
            .to_numpy()
        )
        local = pd.DatetimeIndex(df["time"]).tz_convert(LOCAL_TIMEZONE)
        df["hour"] = local.hour + local.minute / 60.0

        # Optional: Open-Meteo's plane irradiance only from August 2026 on, and
        # a frame need not carry the array at all (see _measured_sky).
        for column in ("global_tilted_irradiance", "pv_w"):
            df[column] = pd.to_numeric(df.get(column, np.nan), errors="coerce")

        # Per sample (see dataset). The flow sensor stops reporting while the
        # pump stands, so its last reading holds into the stop: the state
        # settles that it is none.
        state = self._samples(df, "state").to_numpy()
        supply, ret, flow = (
            self._samples(df, name)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            for name in ("T_supply", "T_return", "flow_lpm")
        )
        flow = np.nan_to_num(np.where(state != self.states.off, flow, 0.0))

        # Heat only reaches the floor circuit while the heat pump is actually
        # serving the space; during SWW the same flow goes to the tank instead.
        measurable = (
            np.isin(state, (self.states.heating, self.states.cooling))
            & (flow > self.MIN_FLOW_LPM)
            & ~np.isnan(supply)
            & ~np.isnan(ret)
        )

        heat = np.where(measurable, floor_heat_w(flow, supply, ret), 0.0)
        df["flow_lpm"] = flow.mean(axis=1)
        df["Q_floor_w"] = heat.mean(axis=1)

        # Each floor's open loop area per sample, as a share of the zone's
        # floor area (see Room.climate): a room without a thermostat mode, or
        # whose mode has no reading, is open. The heat lands where the water
        # runs, so the floor's heat is split per sample too.
        share = self.living_area_fraction
        open_floor = np.zeros((2,) + heat.shape)
        total_m2 = sum(self.room_areas_m2)

        for i, column in enumerate(
            self.room_climate_columns or [None] * len(self.room_areas_m2)
        ):
            is_open = (
                1.0
                if column is None
                else self._samples(df, column).to_numpy() != self.CLIMATE_OFF_STATE
            )
            floor = 0 if i == self.thermostat_room_index else 1
            open_floor[floor] += self.room_areas_m2[i] / total_m2 * is_open

        if not self.room_areas_m2:
            open_floor[0], open_floor[1] = share, 1.0 - share

        df["Q_floor_moved_w"] = floor_heat_moved_w(
            heat, floor_living_fraction(*open_floor, share), share
        ).mean(axis=1)

        # Each floor's open share over the step's samples that ran the floor,
        # and the flow through them: what the water was delivered against
        # (see estimate()) and how much of the circuit took it (see
        # features.floor). NaN where none did.
        running = measurable.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            df["floor_open_living"], df["floor_open_rest"], df["floor_flow_lpm"] = (
                np.where(
                    running > 0,
                    (np.vstack([open_floor, flow[None]]) * measurable).sum(axis=2)
                    / running,
                    np.nan,
                )
            )

        # dt_seconds[i] is the step ENDING at row i, matching _rollout, which
        # advances from i-1 to i over it.
        df["dt_seconds"] = df["time"].diff().dt.total_seconds()
        df.loc[df.index[0], "dt_seconds"] = df["dt_seconds"].median()

        # The step DRIVEN by row i is the one starting there, so its midpoint
        # uses the forward gap, not the backward one.
        forward_dt = df["dt_seconds"].shift(-1)
        forward_dt = forward_dt.fillna(df["dt_seconds"].median())

        (
            df["I_facade_w_per_m2"],
            df["I_facade_direct_w_per_m2"],
            df["sun_from_normal_deg"],
        ) = facade_irradiance_w_per_m2(
            df["time"] + pd.to_timedelta(forward_dt / 2.0, unit="s"),
            direct_horizontal=df["direct_radiation"].to_numpy(dtype=float),
            diffuse_horizontal=df["diffuse_radiation"].to_numpy(dtype=float),
            latitude=self.latitude,
            longitude=self.longitude,
            azimuth_deg=self.facade_azimuth_deg,
        )

        sky = self._measured_sky(df)
        df["I_facade_w_per_m2"] *= sky
        df["I_facade_direct_w_per_m2"] *= sky

        df["shutter_open_fraction"] = self._shutter_open_fraction(df)
        df["open_living"] = self._shutter_open_fraction(df, thermostat_room=True)
        df["open_rest"] = self._shutter_open_fraction(df, thermostat_room=False)
        df["occupants"] = self._occupants(df)

        # A tracker that did not report - before it was recorded, say - says
        # nothing about who was home, not that nobody was: there, who is
        # usually home on that weekday at that time of day where they did.
        if self.presence_columns:
            reported = df[self.presence_columns].notna().all(axis=1)

            if reported.any():
                local = pd.DatetimeIndex(df["time"]).tz_convert(LOCAL_TIMEZONE)
                usual = (
                    df["occupants"]
                    .where(reported)
                    .groupby([local.weekday, local.hour * 60 + local.minute])
                    .transform("median")
                    .fillna(df.loc[reported, "occupants"].median())
                )
                df["occupants"] = df["occupants"].where(reported, usual)

        df["baseload_w"] = df["baseload_w"].fillna(0.0)

        if "wind_speed" in df.columns:
            df["wind_m_per_s"] = (
                pd.to_numeric(df["wind_speed"], errors="coerce") / self.KMH_PER_M_PER_S
            )

        return df

    @staticmethod
    def _samples(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """A step's samples of `name` in time order (see dataset), or the step
        as its own one sample in a frame that has none."""

        columns = [
            column
            for column in df.columns
            if column.startswith(f"{name}_") and column[len(name) + 1 :].isdigit()
        ]

        return df[columns or [name]]

    def _inputs(self, model, df: pd.DataFrame) -> np.ndarray:
        """u = [T_outdoor, Q_internal, Q_living, Q_rest, Q_floor,
        Q_floor_moved] per sample (see physics.zone_state_space).

        Rebuilt per fit iteration because three of the six depend on
        identified parameters (the effective apertures, the in-zone baseload
        fraction and where the occupants sleep). Takes the model rather than
        the raw parameter vector.
        """

        irradiance = df["I_facade_w_per_m2"].to_numpy(dtype=float)
        # The direct sun the extensions' side walls take off the thermostat's
        # room's glazing; the rest's, upstairs, looks over them.
        shaded = df["I_facade_direct_w_per_m2"].to_numpy(
            dtype=float
        ) * extension_shaded_fraction(
            df["sun_from_normal_deg"].to_numpy(dtype=float),
            model.extension_east_depth_ratio,
            model.extension_west_depth_ratio,
        )
        closed_gain = model.closed_shutter_gain_fraction
        q_solar = [
            solar_gain_w(
                a_eff_m2=aperture,
                # The open glass passes its full gain, the shut part a share.
                shutter_open_fraction=closed_gain
                + (1.0 - closed_gain) * df[column].to_numpy(dtype=float),
                facade_irradiance=facade,
            )
            for aperture, column, facade in (
                (model.a_eff_m2, "open_living", irradiance - shaded),
                (model.a_eff_rest_m2, "open_rest", irradiance),
            )
        ]

        occupants = df["occupants"].to_numpy(dtype=float)
        q_internal = internal_gain_w(
            baseload_w=df["baseload_w"].to_numpy(dtype=float),
            internal_gain_fraction=model.internal_gain_fraction,
            occupants=occupants,
        )

        # Q_internal lands by floor area; asleep, the occupants are in the
        # bedrooms instead, so what the thermostat's room has of them beyond
        # its asleep share moves to the rest - the dwelling's total unchanged.
        asleep = night_share(
            df["hour"].to_numpy(dtype=float),
            model.asleep_from_hour,
            model.asleep_until_hour,
            model.asleep_spread_h,
        )
        moved = (
            asleep
            * (model.living_area_fraction - (1.0 - _asleep_in_rest(model)))
            * occupants
            * Q_PERSON_SENSIBLE_W
        )

        return np.column_stack(
            [
                df["T_out"].to_numpy(dtype=float),
                q_internal,
                q_solar[0] - moved,
                q_solar[1] + moved,
                df["Q_floor_w"].to_numpy(dtype=float),
                df["Q_floor_moved_w"].to_numpy(dtype=float),
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward-simulate every scored window; returns (predicted, measured,
        active, row indices), predicted and measured with a column per
        thermometer (see physics.zone_observations): the thermostat's room
        first, the rest of the house second.

        `active` marks the samples where the floor circuit was actually serving
        the space, so calibration diagnostics and validation can report the
        free-floating and actively conditioned regimes apart instead of letting
        a good result in one hide a poor one in the other.

        A run's trailing window shorter than the horizon is left out: it
        accumulates error over less time, so its residuals would come out
        systematically smaller and flatter the metrics.
        """

        runs, warmup_samples, horizon_samples = plan

        model = self._model_from_parameters(x)
        inputs = self._inputs(model, df)
        a, b = zone_state_space(model)
        schedule = self._schedule(model, df)

        measured = df[["T_living", "T_rest"]].to_numpy(dtype=float)
        dt_seconds = df["dt_seconds"].to_numpy(dtype=float)
        floor_heat = df["Q_floor_w"].to_numpy(dtype=float)

        predicted_parts: list[np.ndarray] = []
        measured_parts: list[np.ndarray] = []
        active_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []

        measurement_variance = self.SENSOR_RESOLUTION_K**2 / 12.0
        observation = zone_observations(model)

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
                _window(schedule, run_start, run_end),
            )

            window_start = run_start

            while window_start + horizon_samples <= run_end:
                window_end = window_start + horizon_samples

                simulated = _rollout(
                    a,
                    b,
                    initial_state=estimates[window_start - run_start],
                    inputs=inputs[window_start:window_end],
                    dt_seconds=dt_seconds[window_start:window_end],
                    schedule=_window(schedule, window_start, window_end),
                )

                if window_start - run_start >= warmup_samples:
                    # What the thermostat would have read, which is what the
                    # residual is measured against.
                    predicted_parts.append(simulated @ observation.T)
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

        # Both thermometers, each in its own right: the rest of the house is
        # what pins down how the rooms share the dwelling's heat. A reading
        # that is missing has nothing to say.
        return np.nan_to_num((predicted - measured).ravel())

    def _bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lower, initial, upper) for the parameter vector.

        The capacity and coupling bounds are derived from configuration rather
        than asserted: per m2 of the configured floor, and the room node never
        less than the zone's own air. The effective solar aperture is the
        glass area times a g-value, a frame factor and an incidence factor, so
        at most MAX_APERTURE_FRACTION of that area.
        """

        if self.volume_m3 <= 0.0:
            raise ValueError(
                "building.ceiling_height and building.rooms "
                "areas must be configured: the air node's heat capacity bounds "
                "are derived from the zone's air volume, not assumed."
            )

        air_capacity = RHO_AIR_KG_PER_M3 * self.volume_m3 * CP_AIR_J_PER_KG_K
        floor_m2 = sum(self.room_areas_m2)
        ceiling_m2 = self.room_areas_m2[self.thermostat_room_index]

        # least_squares needs a strictly positive bound width, so a zone with no
        # configured south glass gets a numerically-zero aperture rather than a
        # degenerate bound - solar gain is then simply absent from the model.
        flags = self.glazing_in_thermostat_room or [True] * len(self.glazing_areas_m2)
        living_glass = sum(
            a for a, f in zip(self.glazing_areas_m2, flags, strict=True) if f
        )
        rest_glass = sum(
            a for a, f in zip(self.glazing_areas_m2, flags, strict=True) if not f
        )
        max_aperture = max(self.MAX_APERTURE_FRACTION * living_glass, self.MIN_F_SCALE)
        # No glazing in the thermostat's room, nothing for an extension to shade.
        max_extension = (
            self.MAX_EXTENSION_DEPTH_RATIO if living_glass > 0.0 else self.MIN_F_SCALE
        )
        max_rest_aperture = max(
            self.MAX_APERTURE_FRACTION * rest_glass, self.MIN_F_SCALE
        )

        lower = np.array(
            [
                self.MIN_UA_ENVELOPE_W_PER_K,
                self.MIN_UA_AIR_MASS_W_PER_M2_K * floor_m2,
                air_capacity,
                # At least the screed and the least of a floor under it.
                self.MIN_C_MASS_J_PER_M2_K * floor_m2 + self.screed[0],
                0.0,
                0.0,
                0.0,
                0.0,
                self.MIN_UA_ROOMS_W_PER_K,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                self.MIN_WINDOW_HALF_DAY_C,
                self.MIN_WINDOW_SPREAD_K,
                self.MIN_WINDOW_OPEN_HOUR,
                self.MIN_WINDOW_CLOSE_HOUR,
                self.MIN_WINDOW_HOUR_SPREAD_H,
            ]
        )

        upper = np.array(
            [
                self.MAX_UA_ENVELOPE_W_PER_K,
                self.MAX_UA_AIR_MASS_W_PER_M2_K * floor_m2,
                max(self.MAX_AIR_CAPACITY_J_PER_M2_K * floor_m2, air_capacity),
                self.MAX_C_MASS_J_PER_M2_K * floor_m2,
                max_aperture,
                self.MAX_SENSOR_MASS_FRACTION,
                1.0,
                max_rest_aperture,
                self.MAX_UA_ROOMS_W_PER_K,
                1.0,
                self.MAX_CLOSED_SHUTTER_GAIN_FRACTION,
                max_extension,
                max_extension,
                max(self.MAX_UA_CEILING_W_PER_M2_K * ceiling_m2, self.MIN_F_SCALE),
                1.0,
                self.MAX_UA_WINDOW_W_PER_K,
                self.MAX_WINDOW_HALF_DAY_C,
                self.MAX_WINDOW_SPREAD_K,
                self.MAX_WINDOW_OPEN_HOUR,
                self.MAX_WINDOW_CLOSE_HOUR,
                self.MAX_WINDOW_HOUR_SPREAD_H,
            ]
        )

        initial = np.array(
            [
                self.INITIAL_UA_ENVELOPE_W_PER_K,
                self.INITIAL_UA_AIR_MASS_W_PER_M2_K * floor_m2,
                self.INITIAL_AIR_CAPACITY_J_PER_M2_K * floor_m2,
                self.INITIAL_C_MASS_J_PER_M2_K * floor_m2,
                self.INITIAL_APERTURE_FRACTION * max_aperture,
                self.INITIAL_SENSOR_MASS_FRACTION,
                self.INITIAL_INTERNAL_GAIN_FRACTION,
                self.INITIAL_APERTURE_FRACTION * max_rest_aperture,
                self.INITIAL_UA_ROOMS_W_PER_K,
                # The envelope by floor area to start: the share it is the room's.
                self.living_area_fraction,
                self.INITIAL_CLOSED_SHUTTER_GAIN_FRACTION,
                self.INITIAL_EXTENSION_DEPTH_RATIO,
                self.INITIAL_EXTENSION_DEPTH_RATIO,
                self.INITIAL_UA_CEILING_W_PER_M2_K * ceiling_m2,
                # Asleep by floor area to start, as by day.
                1.0 - self.living_area_fraction,
                self.INITIAL_UA_WINDOW_W_PER_K,
                self.INITIAL_WINDOW_HALF_DAY_C,
                self.INITIAL_WINDOW_SPREAD_K,
                self.INITIAL_WINDOW_OPEN_HOUR,
                self.INITIAL_WINDOW_CLOSE_HOUR,
                self.INITIAL_WINDOW_HOUR_SPREAD_H,
            ]
        )

        # Configured extensions are held (see self.extensions).
        for index, side in ((11, "east"), (12, "west")):
            if side in self.extensions:
                lower[index] = upper[index] = self.extensions[side]

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
        # The array's output per W/m2 from this data, uncorrected, and then the
        # facade's sun corrected by it (see _measured_sky).
        self.pv_power_per_irradiance_m2 = 0.0
        df = self.prepare(df)
        self.pv_power_per_irradiance_m2 = self._pv_power_per_irradiance(df)
        sky = self._measured_sky(df)
        df["I_facade_w_per_m2"] *= sky
        df["I_facade_direct_w_per_m2"] *= sky
        self.screed = self._fit_screed(df)
        self.sleep = self._measured_sleep(df)

        median_dt = float(df["dt_seconds"].median())

        split_index = int(len(df) * self.TRAIN_RATIO)

        if split_index <= 1 or split_index >= len(df):
            raise ValueError("Invalid train/test split.")

        train_df = df.iloc[:split_index].reset_index(drop=True)

        plan = self._rollout_plan(train_df, median_dt)
        lower, initial, upper = self._bounds()

        # Only what is not held by the configuration (see _bounds).
        free = lower < upper

        def full(x_free: np.ndarray) -> np.ndarray:
            x = initial.copy()
            x[free] = x_free

            return x

        if sum(self.glazing_areas_m2) <= 0.0:
            logger.warning(
                "Building thermal calibration: building.south_glazing is not "
                "configured, so solar gain is bounded to zero - any real solar "
                "gain will be absorbed by the envelope and capacity parameters."
            )

        def residuals(x_free: np.ndarray) -> np.ndarray:
            return self._window_residuals(full(x_free), train_df, plan)

        # x_scale="jac": the parameters span conductances (10^2 W/K) and heat
        # capacities (10^7 J/K), so an unscaled trust region would step
        # meaninglessly in one or the other.
        ordinary = least_squares(
            residuals, initial[free], bounds=(lower[free], upper[free]), x_scale="jac"
        )

        # Robust loss scaled by the data's own residual spread, so a window
        # containing an unmodelled disturbance (an opened window, a visitor,
        # a wood stove) is down-weighted instead of dragging the fit.
        deviation = np.abs(ordinary.fun - np.median(ordinary.fun))
        f_scale = max(self.MAD_TO_STD * float(np.median(deviation)), self.MIN_F_SCALE)

        result = least_squares(
            residuals,
            ordinary.x,
            bounds=(lower[free], upper[free]),
            loss="soft_l1",
            f_scale=f_scale,
            x_scale="jac",
        )

        x = full(result.x)
        # A configured parameter carries no error from this fit.
        std_errors = np.zeros_like(x)
        std_errors[free] = self._parameter_std_errors(result)
        self.parameter_std_errors = dict(
            zip(self.PARAMETER_NAMES, std_errors, strict=True)
        )

        self.model = self._model_from_parameters(x)

        predicted, _, active, _ = self._simulate_windows(x, train_df, plan)

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
                    self.PARAMETER_NAMES, x, std_errors, strict=True
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

        predicted_both, measured_both, active, positions = self._simulate_windows(
            x, df, plan
        )
        # Headline: the thermostat's room, where comfort is set and judged.
        predicted, measured = predicted_both[:, 0], measured_both[:, 0]

        metrics = {
            "mae_rest": float(
                np.nanmean(np.abs(measured_both[:, 1] - predicted_both[:, 1]))
            ),
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

            # Held by the configuration, not by the data (see _bounds).
            if lower[index] >= upper[index]:
                continue

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
            aperture_fraction = (model.a_eff_m2 + model.a_eff_rest_m2) / total_glass_m2
            metrics["aperture_fraction"] = aperture_fraction
            implausible = aperture_fraction < self.MIN_PLAUSIBLE_APERTURE_FRACTION
            metrics["implausible_aperture"] = float(implausible)

            if implausible:
                logger.warning(
                    "Building thermal validation: the apertures (%.2f m2) are "
                    "only %.3f of the %.1f m2 of configured south glass, which "
                    "would need a g-value no real glazing has. The fit may be "
                    "statistically converged and still not describe this "
                    "window - most likely the data covers too little sunlit "
                    "time with the shutters open.",
                    model.a_eff_m2 + model.a_eff_rest_m2,
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
        # The envelope is the whole dwelling's, so these trends are read off the
        # zone's area-weighted average - both rooms, the rest's own error where
        # it was measured.
        drive = (df["T_out"] - df["T_air"]).to_numpy(dtype=float)[positions]
        share = model.living_area_fraction
        room_error = predicted_both - measured_both
        zone_error = share * room_error[:, 0] + (1.0 - share) * np.where(
            np.isnan(room_error[:, 1]), room_error[:, 0], room_error[:, 1]
        )
        sunlit = (df["shutter_open_fraction"] * df["I_facade_w_per_m2"]).to_numpy(
            dtype=float
        )[positions]

        window_drive = drive.reshape(windows, horizon_samples).mean(axis=1)
        window_error = zone_error.reshape(windows, horizon_samples)[:, -1]

        # Dark windows only. Solar gain and the indoor-outdoor difference both
        # peak in the afternoon - on this installation they correlate about
        # +0.4 - so a trend fitted over all windows measures the NET of two
        # errors and can read clean while both are large. Measured directly:
        # over all windows the single-node model this replaced slopes +0.0009
        # K/K, but after
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

                if lower[index] >= upper[index]:
                    continue

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

    def estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """The filter's running estimate of every state, indexed by time.

        Unlike the fixed-horizon rollouts validate() scores, this never
        restarts: the filter corrects at every step, so the result is
        continuous by construction and has none of their jumps. That
        makes it the right thing to draw, and for the two-node model it is the
        only way to see the mass node at all - no sensor measures the screed.

        It is also what an MPC has to start from, so this is the same quantity
        planning will consume, not a display-only variant of it.
        """

        model = self.get_model()
        prepared = self.prepare(df)

        return self._states_frame(model, self._filtered(model, prepared), prepared)

    def _filtered(
        self, model: BuildingThermalModel, prepared: pd.DataFrame
    ) -> np.ndarray:
        """The filter's estimate of every state over a prepared frame."""

        a, b = zone_state_space(model)

        return kalman_states(
            a,
            b,
            prepared[["T_living", "T_rest"]].to_numpy(dtype=float),
            self._inputs(model, prepared),
            prepared["dt_seconds"].to_numpy(dtype=float),
            self.PROCESS_NOISE_W,
            self.SENSOR_RESOLUTION_K**2 / 12.0,
            zone_observations(model),
            self.DISTURBANCE_INPUTS,
            self._schedule(model, prepared),
        )

    def _schedule(
        self, model: BuildingThermalModel, prepared: pd.DataFrame
    ) -> ScheduledSystem | None:
        """The zone's system step by step where its envelope opens overnight
        (see physics.envelope_conductance_w_per_k); None for a constant
        envelope, exactly as before it."""

        if not model.ua_window_w_per_k:
            return None

        return ScheduledSystem(
            lambda ua: zone_state_space(model, ua), self._envelope(model, prepared)
        )

    @staticmethod
    def _envelope(model: BuildingThermalModel, prepared: pd.DataFrame) -> np.ndarray:
        """The envelope's conductance (W/K) at each row of a prepared frame."""

        return envelope_conductance_w_per_k(
            model,
            prepared["T_out_day"].to_numpy(dtype=float),
            prepared["hour"].to_numpy(dtype=float),
        )

    def room_heat_habit(self, df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """What each room gained or lost beyond the model - windows opened, a
        shower's warmth - per local quarter hour of the day (W; index the
        minute of the day, a column a room): the median over the frame's
        HABIT_HISTORY_DAYS up to now, the floor's runs left out - its own
        error is not a habit, and a plan decides its runs (see forecast())."""

        model = self.get_model()
        prepared = self.prepare(df)
        prepared = prepared[prepared["time"] <= now].reset_index(drop=True)

        if prepared.empty:
            raise ValueError("No measurements to take the rooms' habit from.")

        unexplained = self._unexplained_heat_w(
            model,
            self._filtered(model, prepared),
            self._inputs(model, prepared),
            prepared["dt_seconds"].to_numpy(dtype=float),
            self._schedule(model, prepared),
        )
        usable = (
            (prepared["time"] >= now - timedelta(days=self.HABIT_HISTORY_DAYS))
            & (prepared["Q_floor_w"] == 0.0)
        ).to_numpy()
        # The filter's first step has no prediction to differ from.
        usable[0] = False
        local = prepared["time"].map(to_local_time)
        minute = local.map(lambda t: t.hour * 60 + t.minute).to_numpy()

        return (
            pd.DataFrame(unexplained[usable], columns=["living", "rest"])
            .groupby(minute[usable])
            .median()
            .reindex(range(0, 24 * 60, 15), fill_value=0.0)
        )

    def _unexplained_heat_w(
        self,
        model: BuildingThermalModel,
        estimates: np.ndarray,
        inputs: np.ndarray,
        dt_seconds: np.ndarray,
        schedule: ScheduledSystem | None = None,
    ) -> np.ndarray:
        """The heat into each room (W, a column each) the filter added over
        each step beyond the model's own prediction: its correction of the
        whole state read back as the heat flows it lets in (see
        DISTURBANCE_INPUTS) - a room's loss also cools the floor under it."""

        a, b = zone_state_space(model)
        heat = np.zeros((len(estimates), len(self.DISTURBANCE_INPUTS)))
        cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for i in range(1, len(estimates)):
            dt = float(dt_seconds[i])
            key = round(dt, 3)

            if schedule is not None:
                a_d, b_d = schedule.discrete(i - 1, dt)
                back = np.linalg.pinv(b_d[:, self.DISTURBANCE_INPUTS])
            else:
                if key not in cache:
                    a_d, b_d = discretize_zoh(a, b, dt)
                    cache[key] = (
                        a_d,
                        b_d,
                        np.linalg.pinv(b_d[:, self.DISTURBANCE_INPUTS]),
                    )

                a_d, b_d, back = cache[key]
            heat[i] = back @ (
                estimates[i] - a_d @ estimates[i - 1] - b_d @ inputs[i - 1]
            )

        # The rooms' own channels; the floor's is the floor's error.
        return heat[:, :2]

    def _measured_sleep(self, df: pd.DataFrame) -> tuple[float, float, float]:
        """The household's hours of sleep (to bed, up, spread - see
        physics.night_share) from its electricity: the median baseload at each
        local time of day, one level awake and a lower one asleep, stepping
        between them at those hours. Robustly - an evening's cooking, a
        morning's kettle stand above the awake level without being sleep."""

        profile = df.groupby("hour")["baseload_w"].median().dropna()
        hour = profile.index.to_numpy(dtype=float)
        baseload = profile.to_numpy(dtype=float)

        def residuals(x: np.ndarray) -> np.ndarray:
            asleep_w, awake_w, to_bed, up, spread = x
            asleep = night_share(hour, to_bed, up, spread)

            return asleep_w + (awake_w - asleep_w) * (1.0 - asleep) - baseload

        median = float(np.median(baseload))
        result = least_squares(
            residuals,
            [
                float(baseload.min()),
                median,
                self.INITIAL_ASLEEP_FROM_HOUR,
                self.INITIAL_ASLEEP_UNTIL_HOUR,
                self.INITIAL_ASLEEP_SPREAD_H,
            ],
            bounds=(
                [
                    0.0,
                    0.0,
                    self.MIN_ASLEEP_FROM_HOUR,
                    self.MIN_ASLEEP_UNTIL_HOUR,
                    self.MIN_ASLEEP_SPREAD_H,
                ],
                [
                    np.inf,
                    np.inf,
                    self.MAX_ASLEEP_FROM_HOUR,
                    self.MAX_ASLEEP_UNTIL_HOUR,
                    self.MAX_ASLEEP_SPREAD_H,
                ],
            ),
            loss="soft_l1",
            f_scale=max(float(np.median(np.abs(baseload - median))), self.MIN_F_SCALE),
        )
        to_bed, up, spread = (float(value) for value in result.x[2:])
        logger.info(
            "Building thermal calibration: asleep from %.2f h to %.2f h "
            "(spread %.2f h), %.0f W against %.0f W awake",
            to_bed,
            up,
            spread,
            result.x[0],
            result.x[1],
        )

        return to_bed, up, spread

    def _fit_screed(self, df: pd.DataFrame) -> tuple[float, float]:
        """(screed J/K, its conductance down to the slabs W/K) from the
        floor's runs in a prepared frame, either mode: what the rooms alone
        hardly show - a layer that passes the heat on within the hour - the
        water sees in full, as a run's heat falling as it warms (see
        physics.zone_state_space). (0, 0) without a saved model to estimate
        the slabs by, or too few runs.

        Each run followed from its start with the screed where the slab below
        and the room above hold it while nothing heats it - in between, by
        their conductances (real data, Oct 2026: started at the slab, runs
        from a night off read 800-1000 W short at first) - its heat from the
        water, Q = G (supply - screed), shared with the slabs below and the
        rooms above through the floor's surface, at the saved model's
        surface conductance: the slabs as that model estimates them, the
        rooms as measured, both floors taken as one by the water under each.
        G is the pipes' UA at each step's open loops and flow (see
        physics.floor_loops_conductance_w_per_k), so a run through fewer
        loops heats less of the screed, from water that cools less along
        them. Scored on the heat where the compressor had settled. The
        pipes' UA is the floor circuit's to identify (see features.floor),
        against the screed this fixes. Bounded per m2 of floor (see
        MIN_SCREED_J_PER_M2_K) and below the slabs' whole capacity, which
        holds it.
        """

        # Here, not at the top: see validate().
        from features.cop import HeatPumpCOPIdentifier

        saved = self.model

        if saved is None:
            return 0.0, 0.0

        # Both floors as one, weighted by the water under each: the heated
        # part of the floor, which takes the open share of the circuit's
        # conductances and capacity alike - so only its heat scales with it.
        states = self._filtered(saved, df)
        living = floor_living_fraction(
            df["floor_open_living"], df["floor_open_rest"], saved.living_area_fraction
        )
        slab = (
            living * states[:, ZONE_SLAB_STATES[0]]
            + (1.0 - living) * states[:, ZONE_SLAB_STATES[1]]
        )
        room = living * df["T_living"] + (1.0 - living) * df["T_rest"]
        running = df["state"].isin((self.states.heating, self.states.cooling))
        begins = running & (
            ~running.shift(fill_value=False) | (df["state"] != df["state"].shift())
        )
        frame = pd.DataFrame(
            {
                "time": df["time"],
                "run": begins.cumsum(),
                "supply": df["T_supply"],
                "heat": df["Q_floor_w"],
                "slab": slab,
                "room": room,
                # Every loop, at the step's own flow, in a step the water did
                # not run (see prepare).
                "open": (df["floor_open_living"] + df["floor_open_rest"]).fillna(1.0),
                "flow": df["floor_flow_lpm"].fillna(df["flow_lpm"]),
            }
        )[running.to_numpy()].dropna()
        start = frame["time"].groupby(frame["run"]).transform("min")
        frame["settled"] = frame["time"] - start >= HeatPumpCOPIdentifier.STARTUP
        runs = [run for _, run in frame.groupby("run") if len(run) >= 3]

        if len(runs) < self.MIN_SCREED_RUNS:
            return 0.0, 0.0

        floor_m2 = sum(self.room_areas_m2)
        dt_s = float(df["dt_seconds"].median())
        surface = saved.ua_air_mass_w_per_k

        def residuals(log_x: np.ndarray) -> np.ndarray:
            pipes, capacity, down = np.exp(log_x)
            errors = []

            for run in runs:
                supply, slab_c, room_c, open_, flow = (
                    run[name].to_numpy(dtype=float)
                    for name in ("supply", "slab", "room", "open", "flow")
                )
                # The water to the heated part of the screed, per the open
                # share it is of the floor: the same balance as every loop's.
                loops = floor_loops_conductance_w_per_k(pipes, open_, flow)
                water = np.divide(
                    loops, open_, out=np.zeros_like(loops), where=open_ > 0.0
                )
                total = water + down + surface
                decay = np.exp(-total / capacity * dt_s)
                mean = (1.0 - decay) / (total / capacity * dt_s)
                screed_c = (down * slab_c[0] + surface * room_c[0]) / (down + surface)
                heat = []

                # Each step exactly: the screed relaxes to where the water's
                # heat and its losses balance.
                for k in range(len(run)):
                    balance = (
                        water[k] * supply[k] + down * slab_c[k] + surface * room_c[k]
                    ) / total[k]
                    heat.append(
                        loops[k]
                        * (supply[k] - balance - (screed_c - balance) * mean[k])
                    )
                    screed_c = balance + (screed_c - balance) * decay[k]

                errors.append(
                    (np.array(heat) - run["heat"].to_numpy(dtype=float))[
                        run["settled"].to_numpy(dtype=bool)
                    ]
                )

            # In kW, so the residuals sit near 1.
            return np.concatenate(errors) / 1000.0

        lower = np.log(
            [
                self.MIN_SCREED_W_PER_M2_K * floor_m2,
                self.MIN_SCREED_J_PER_M2_K * floor_m2,
                self.MIN_SCREED_W_PER_M2_K * floor_m2,
            ]
        )
        upper = np.log(
            [
                self.MAX_SCREED_W_PER_M2_K * floor_m2,
                min(
                    self.MAX_SCREED_J_PER_M2_K * floor_m2,
                    0.5 * saved.c_mass_j_per_k,
                ),
                self.MAX_SCREED_W_PER_M2_K * floor_m2,
            ]
        )
        initial = np.log(
            [
                self.INITIAL_SCREED_WATER_W_PER_M2_K * floor_m2,
                self.INITIAL_SCREED_J_PER_M2_K * floor_m2,
                self.INITIAL_SCREED_DOWN_W_PER_M2_K * floor_m2,
            ]
        )
        fit = least_squares(
            residuals, np.clip(initial, lower, upper), bounds=(lower, upper)
        )
        pipes, capacity, down = map(float, np.exp(fit.x))
        logger.info(
            "Building thermal calibration: screed %.1f MJ/K (%.0f kJ/m2K) on to "
            "the slabs at %.0f W/K, the pipes to it at UA %.0f W/K, over %d runs",
            capacity / 1e6,
            capacity / floor_m2 / 1e3,
            down,
            pipes,
            len(runs),
        )

        return capacity, down

    STATE_NAMES = ("living", "rest", "slab_living", "slab_rest")
    SCREED_NAMES = ("screed_living", "screed_rest")

    @classmethod
    def state_names(cls, model: BuildingThermalModel) -> tuple[str, ...]:
        """The zone's states by name (see physics.zone_state_space)."""

        if model.screed_capacity_j_per_k:
            return cls.STATE_NAMES + cls.SCREED_NAMES

        return cls.STATE_NAMES

    @classmethod
    def _states_frame(
        cls, model: BuildingThermalModel, states: np.ndarray, prepared: pd.DataFrame
    ) -> pd.DataFrame:
        """The states by name, indexed by the prepared frame's time, with
        `mass` the slabs' mean (see physics.zone_slab) and `floor` what the
        floor circuit delivers against (see physics.zone_floor): through the
        loops that were open, or every loop where it did not run."""

        frame = pd.DataFrame(
            states,
            columns=list(cls.state_names(model)),
            index=prepared["time"].to_numpy(),
        )
        frame["mass"] = states @ zone_slab(model)
        living = floor_living_fraction(
            prepared["floor_open_living"].to_numpy(dtype=float),
            prepared["floor_open_rest"].to_numpy(dtype=float),
            model.living_area_fraction,
        )
        frame["floor"] = living * (states @ zone_floor(model, 1.0)) + (1.0 - living) * (
            states @ zone_floor(model, 0.0)
        )

        return frame

    def forecast(
        self,
        df: pd.DataFrame,
        now: datetime,
        baseload_w: pd.Series | None = None,
        pv_w: pd.Series | None = None,
        habits: pd.DataFrame | None = None,
        room_heat_w: pd.DataFrame | None = None,
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
        baseload forecaster's own curve when given. `pv_w`, the PV forecast,
        sets the sky ahead as the measured output does behind (see
        _measured_sky): the array is the facade's sun meter, and Solcast
        forecasts its sun better than Open-Meteo, which runs low (20 Aug - 16
        Oct 2026, 0-6 h ahead: PV 148-164 W out against 176-216 W, Open-Meteo
        65-89 W short). Over 156 forecasts (2 Sep - 16 Oct, issued 7-13 h) that
        put the zone 0.153 K out over the first six hours against 0.166 K, with
        the real shutters - the measured sun did 0.147 K - and halved its bias
        (-0.014 against -0.050 K).

        The shutters stand as they did on the same weekday at that time of day
        over `habits`, their past weeks (see HABIT_HISTORY_DAYS; the frame's
        own past without), and one that is open holds where it stands
        for the rest of the day. They follow who is home in the day, which
        follows the household's week, not the sky: the thermostat room's
        shutter opened by 8-10 h on days someone was in and stayed shut until
        18 h on days no one was (Aug-Oct 2026: its day's mean position
        correlated 0.75 with who was in, 0.13 with the sun). Over 152
        forecasts (12 Sep - 16 Oct, issued 7-22 h, 30 h ahead) that put the
        zone 0.173 K out over the first six hours and 0.229 K over a day,
        against 0.185 and 0.246 K by the time of day over the frame's last two
        days, 0.183 and 0.246 K holding today's position, and 0.158 and 0.219
        K with the real shutters; issued at 7 h, while they are still shut,
        the thermostat room's sun came out 82 W off against 126 W holding them
        shut. A gradient-boosted model of the position (weekday, time, lead,
        position and who is in at issue, and the sun as measured) did worse
        on the same 34 days, walk-forward: 116-135 W over six hours trained on
        the last 28-63 days, 154 W on all since June, against 96 W for the
        habit (Influx, Jun-Oct 2026). The more of the past it took, the worse:
        the habit moves with the season - in June and July the shutter was
        shut against the sun.

        Who is home is taken the same way, the median on that weekday at that
        time of day over `habits`, not the last reading carried on - an
        evening at home put everyone in all the next day - nor the frame's day
        or two, whose weekdays a weekend morning was forecast from. Over 226
        forecasts (21 Aug - 16 Oct 2026, issued 7, 12, 18 and 22 h, 30 h
        ahead) that put 0.27 people wrong over a day against 0.45 by the time
        of day over the frame and 0.69 holding the last reading; 0.19 against
        0.42 once a fortnight away in August was three weeks past - the median,
        not the mean (0.40), lets one such week go. 75 W a person moves the
        zone little (0.352 against 0.357 K for 0.44 against 0.61 people wrong,
        Sep-Oct 2026).

        Each room also takes `room_heat_w`, the heat the filter found it
        gaining or losing beyond the model at that time of day over the past
        weeks (see room_heat_habit()). No sensor sees an open window or a
        shower, but they recur - the bedrooms' windows open on summer nights,
        and showers at 8 and 22 h -
        and they showed in it as some 60-95 W lost upstairs at night and
        150-230 W gained at those hours. Walk-forward, issued every 6 h on
        days the model was not fitted on (15-23 Oct and 10 Aug - 15 Sep
        2026), the living room came out 0.085/0.098 K off over six hours
        against 0.084/0.107 K without, and 0.146/0.181 against 0.163/0.211 K
        over a day; the rest 0.084/0.138 against 0.096/0.143, and 0.157/0.332
        against 0.201/0.318 K. The mean instead of the median did worse,
        a night's odd window taken as every night's; the habit made at
        midnight instead of at each forecast did as well (within 0.002 K).

        The rest is each sensor's last reading carried forward.
        """

        zone, last = self._trajectory(df, now, baseload_w, pv_w, habits, room_heat_w)

        return zone.iloc[last:]

    def trajectory(
        self,
        df: pd.DataFrame,
        now: datetime,
        baseload_w: pd.Series | None = None,
        pv_w: pd.Series | None = None,
        habits: pd.DataFrame | None = None,
        room_heat_w: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """The zone over the whole frame, indexed by time: the filter's estimate
        of every state up to now, then the free-running forecast from there (see
        forecast()), with the measured zone temperature and the gains beside
        them. One filter pass serves both halves - what a plan starts from and
        is driven by, and the history it is drawn against.
        """

        zone, _ = self._trajectory(df, now, baseload_w, pv_w, habits, room_heat_w)

        return zone

    def _trajectory(
        self,
        df: pd.DataFrame,
        now: datetime,
        baseload_w: pd.Series | None,
        pv_w: pd.Series | None,
        habits: pd.DataFrame | None,
        room_heat_w: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, int]:

        model = self.get_model()

        # The PV forecast stands in for the measurement ahead, before prepare()
        # turns the output into the sky (see forecast()).
        if pv_w is not None:
            df = df.copy()
            ahead = df["target_time"] > now
            df.loc[ahead, "pv_w"] = pv_w.reindex(df.loc[ahead, "target_time"]).to_numpy(
                dtype=float
            )

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

        local = prepared["time"].map(to_local_time)
        quarter = local.map(lambda t: t.hour * 60 + t.minute)

        # Who is home ahead, and the shutters that follow them, as on that
        # weekday at that time of day over the past weeks (see forecast()).
        history = (
            prepared.loc[known] if habits is None or habits.empty else habits
        ).set_index("time")
        past = history.index.map(to_local_time)
        habit = [
            past.map(lambda t: t.weekday()),
            past.map(lambda t: t.hour * 60 + t.minute),
        ]
        ahead = pd.MultiIndex.from_arrays(
            [local[~known].map(lambda t: t.weekday()), quarter[~known]]
        )

        def usual(values, last):
            return values.groupby(habit).median().reindex(ahead).fillna(last).to_numpy()

        # An open shutter as it stands for the rest of today.
        if self.shutter_columns:
            columns = self.shutter_columns
            last = (
                prepared.loc[known, columns]
                .apply(pd.to_numeric, errors="coerce")
                .iloc[-1]
            )
            forecast = usual(
                history[columns].apply(pd.to_numeric, errors="coerce"), last
            )
            today = (
                local[~known].map(lambda t: t.date()) == local[known].iloc[-1].date()
            ).to_numpy()
            is_open = last.to_numpy() > np.array(
                self.shutter_closed_positions or [0.0] * len(columns)
            )
            forecast[np.ix_(today, is_open)] = last.to_numpy()[is_open]
            prepared.loc[~known, columns] = forecast
            prepared["shutter_open_fraction"] = self._shutter_open_fraction(prepared)
            prepared["open_living"] = self._shutter_open_fraction(prepared, True)
            prepared["open_rest"] = self._shutter_open_fraction(prepared, False)

        # A tracker that did not report says nothing about who was home.
        if self.presence_columns:
            reported = history[self.presence_columns].notna().all(axis=1)
            prepared.loc[~known, "occupants"] = usual(
                self._occupants(history).where(reported),
                prepared.loc[known, "occupants"].iloc[-1],
            )

        if baseload_w is not None:
            aligned = baseload_w.reindex(prepared["time"]).to_numpy(dtype=float)
            prepared["baseload_w"] = np.where(
                np.isnan(aligned), prepared["baseload_w"], aligned
            )

        a, b = zone_state_space(model)
        inputs = self._inputs(model, prepared)
        dt_seconds = prepared["dt_seconds"].to_numpy(dtype=float)
        schedule = self._schedule(model, prepared)

        last = int(np.flatnonzero(known)[-1])

        estimates = kalman_states(
            a,
            b,
            prepared[["T_living", "T_rest"]].to_numpy(dtype=float)[: last + 1],
            inputs[: last + 1],
            dt_seconds[: last + 1],
            self.PROCESS_NOISE_W,
            self.SENSOR_RESOLUTION_K**2 / 12.0,
            zone_observations(model),
            self.DISTURBANCE_INPUTS,
            _window(schedule, 0, last + 1),
        )

        # Nothing is delivered to the zone: this is what happens if the heat
        # pump is left out of it.
        inputs = inputs.copy()
        inputs[last:, 4:6] = 0.0

        # What each room gains beyond the model at that time of day (see
        # room_heat_habit()).
        if room_heat_w is not None:
            inputs[last:, 2:4] += (
                room_heat_w.reindex(quarter.iloc[last:].to_numpy())
                .fillna(0.0)
                .to_numpy()
            )

        simulated = _rollout(
            a,
            b,
            initial_state=estimates[-1],
            inputs=inputs[last:],
            dt_seconds=dt_seconds[last:],
            schedule=_window(schedule, last, len(prepared)),
        )

        # The rollout starts from the filter's last estimate, so the two join
        # without a seam.
        states = np.vstack([estimates[:last], simulated])
        zone = self._states_frame(model, states, prepared)

        # What the thermostat would read of it, which is the only column that
        # can be drawn against - or scored on - its measurement (see
        # zone_observation).
        zone["reading"] = states @ zone_observation(model)
        zone["measured"] = np.where(
            known, prepared["T_living"].to_numpy(dtype=float), np.nan
        )
        # The heat no decision changes, split by where it lands, for a plan to
        # start from: the same gains the rollout was driven by.
        zone["internal_gain_w"] = inputs[:, 1]
        zone["living_gain_w"] = inputs[:, 2]
        zone["rest_gain_w"] = inputs[:, 3]
        # The envelope a plan's zone has each step (see _schedule).
        zone["envelope_w_per_k"] = self._envelope(model, prepared)
        # Which loops the floor's runs went through, and their flow, where
        # they were measured (see prepare and last_run_loops()).
        for column in ("floor_open_living", "floor_open_rest", "floor_flow_lpm"):
            zone[column] = np.where(known, prepared[column], np.nan)

        return zone, last

    @staticmethod
    def last_run_loops(zone: pd.DataFrame) -> tuple[float, float, float] | None:
        """The loops the last floor run in a trajectory() went through, what
        a plan takes its own runs to go through: (open_living, open_rest,
        flow_lpm), the set it held longest and its flow then - or None
        without a run. Not now: the rooms close theirs between runs. Not its
        last step: the rooms switch at a run's ends, one at a time (26 Oct
        2026: its last step had the bathroom alone open)."""

        columns = ["floor_open_living", "floor_open_rest"]
        ran = zone[columns[0]].notna().to_numpy()
        starts = ran & ~np.concatenate([[False], ran[:-1]])
        run = zone[ran & (np.cumsum(starts) == starts.sum())]
        loops = list(
            zip(run[columns[0]].round(3), run[columns[1]].round(3), strict=True)
        )

        if not loops:
            return None

        # The longest held, the latest of equally long.
        held = max(reversed(loops), key=loops.count)

        if sum(held) <= 0.0:
            return None

        flow = run["floor_flow_lpm"][[loop == held for loop in loops]].mean()

        return float(held[0]), float(held[1]), float(flow)

    def habit_dataset(self) -> DatasetDefinition:
        """The shutters and who is home alone, over the weeks forecast() takes
        their habit from (see HABIT_HISTORY_DAYS) - a few series, where the
        whole zone frame would load weeks of weather snapshots with them.
        After dataset(), which names them."""

        builder = DatasetBuilder()

        for name, sensor in zip(
            self.shutter_columns + self.presence_columns,
            self.shutter_sensors + self.presence_sensors,
            strict=True,
        ):
            builder = builder.timeseries(
                name, sensor, interval="15m", aggregation="last", fill="previous"
            )

        return builder.build()

    def dataset(self, config: Config) -> DatasetDefinition:
        self.room_areas_m2 = [room.area_m2 for room in config.building.rooms]
        self.states = config.heat_pump.states
        # Derived, not configured: the same areas already drive the weighting.
        self.volume_m3 = sum(self.room_areas_m2) * config.building.ceiling_height
        self.glazing_areas_m2 = [
            glazing.glass_m2 for glazing in config.building.south_glazing
        ]
        self.facade_azimuth_deg = config.building.facade_azimuth
        shaded = [
            glazing
            for glazing in config.building.south_glazing
            if glazing.cover is not None
        ]
        self.shutter_areas_m2 = [glazing.glass_m2 for glazing in shaded]
        self.shutter_closed_positions = [glazing.closed_position for glazing in shaded]
        self.shutter_sensors = [glazing.cover for glazing in shaded]
        self.glazing_in_thermostat_room = [
            glazing.thermostat_room for glazing in config.building.south_glazing
        ]
        self.shutter_in_thermostat_room = [
            glazing.thermostat_room for glazing in shaded
        ]
        configured = [
            glazing
            for glazing in config.building.south_glazing
            if glazing.extensions is not None
        ]

        if any(not glazing.thermostat_room for glazing in configured) or (
            len(configured) > 1
        ):
            raise ValueError(
                "building.south_glazing: extensions belong to one glazing, the "
                "thermostat's room's - the shade model has one opening between "
                "two walls, and the rest's glazing looks over them."
            )

        # The glazing floor to ceiling (see SouthGlazing.extensions).
        self.extensions = {
            side: depth_m * config.building.ceiling_height / glazing.glass_m2
            for glazing in configured
            for side, depth_m in zip(("east", "west"), glazing.extensions, strict=True)
        }
        thermostat = config.building.thermostat.temperature
        matching = [
            i
            for i, room in enumerate(config.building.rooms)
            if room.temperature.entity_id == thermostat.entity_id
        ]

        if not matching or len(config.building.rooms) < 2:
            raise ValueError(
                "building.rooms must include the thermostat's own room and at "
                "least one more: the zone is modelled as that room and the rest "
                "of the house."
            )

        self.thermostat_room_index = matching[0]
        self.room_temperature_columns = [
            f"room_temperature_{i}" for i in range(len(self.room_areas_m2))
        ]
        self.room_climate_columns = [
            None if room.climate is None else f"room_climate_{i}"
            for i, room in enumerate(config.building.rooms)
        ]
        self.shutter_columns = [f"shutter_{i}" for i in range(len(shaded))]
        self.presence_columns = [f"presence_{i}" for i in range(len(config.presence))]
        self.presence_sensors = list(config.presence)

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
                # On the PV array's plane, which Open-Meteo is configured with:
                # what its measured output is compared against (see prepare).
                "global_tilted_irradiance",
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
                "global_tilted_irradiance",
            ],
        )

        # Numeric series are averaged onto a 5 minute grid first and only then
        # onto the 15 minute model grid: with fill="previous" that makes each
        # model step a time-weighted average of a sensor that only reports on
        # change, rather than a mean over however many raw points happened to
        # land in the step.
        #
        # Temperatures are continuous and only reported on change, so the
        # previous reading is the current one.
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
            ("baseload_w", config.baseload, "mean", "previous"),
            # Rate-like and unfilled: a gap - and every row ahead of now - is
            # no measurement, and leaves the forecast's sun as it is.
            ("pv_w", config.solar, "mean", "none"),
        ]

        if config.heat_pump.outdoor_temperature is not None:
            numeric.append(
                ("T_out", config.heat_pump.outdoor_temperature, "mean", "previous")
            )

        # Cover position is an event-driven state - it only changes when a
        # shutter actually moves - so it is carried forward, not averaged.
        shutters: list[tuple[str, SensorReference, Aggregation, FillMethod]] = [
            (name, sensor, "last", "previous")
            for name, sensor in zip(
                self.shutter_columns, self.shutter_sensors, strict=True
            )
        ]
        numeric += shutters

        # The device trackers arrive as strings, which cannot be resampled by
        # averaging: they are taken directly on the model grid, keeping the
        # value in force at the step's end.
        textual: list[tuple[str, SensorReference]] = list(
            zip(self.presence_columns, config.presence, strict=True)
        )

        # The floor's heat is a product of three sensors that only counts while
        # the heat pump serves the space, and a run often turns from hot water
        # to the floor within a step: its four sensors arrive per 5 minute
        # sample (see prepare), so a step's heat is formed per sample. Averaged
        # first, a step that began on hot water put the tank's heat in the
        # floor (real data, 8 Oct 2026: 8.0 kW for a step that was 6 minutes
        # SWW, against 3-4 kW in the steps after it; 4.2 kW per sample).
        # Over time, not over readings: the Ecodan reports every few seconds
        # while the valve turns, so a reading mean is mostly the turn (that
        # step 6.1 kW). Flow holds its last reading too, which the state
        # zeroes once the pump stops.
        hydronic: list[tuple[str, SensorReference, Aggregation, FillMethod]] = [
            ("state", config.heat_pump.state, "last", "previous"),
            ("T_supply", config.heat_pump.supply_temperature, "time_mean", "none"),
            ("T_return", config.heat_pump.return_temperature, "time_mean", "none"),
            ("flow_lpm", config.heat_pump.flow, "time_mean", "none"),
            # Which loops the water runs through: per sample too, as the rooms
            # close theirs within a run (see prepare).
            *[
                (name, room.climate, "last", "previous")
                for name, room in zip(
                    self.room_climate_columns, config.building.rooms, strict=True
                )
                if name is not None
            ],
        ]

        for name, sensor, aggregation, fill in hydronic:
            builder = builder.timeseries(
                name,
                sensor,
                interval="5m",
                aggregation=aggregation,
                fill=fill,
                target_interval="15min",
                target_resample="samples",
            )

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

        for name in [entry[0] for entry in hydronic + numeric + textual]:
            builder = builder.join(
                left="weather",
                right=name,
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )

        return builder.build()
