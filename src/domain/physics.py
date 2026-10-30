"""Physical relations and balances of the modelled system.

The energy balances themselves, the constants they are written in, and the
heat flows that enter them - separated from how the parameters in them are
identified (features/) and from how they are numerically integrated
(domain/dynamics.py). Everything here is a statement about the physics, so it
is equally available to the identifiers that fit these models, to the filter
that estimates their states and to the planner that acts on them.
"""

import numpy as np
from scipy.special import expit, ndtr

from domain.models import (
    BoilerThermalModel,
    BuildingThermalModel,
    DewPointModel,
    FloorCircuitModel,
)

# Physical constants (water), not fit parameters.
RHO_WATER_KG_PER_L = 1.0
CP_WATER_J_PER_KG_K = 4186.0

# The climate the mains water follows (deg C): KNMI's 1991-2020 normals for De
# Bilt, the annual mean and the mean of the coldest (January) and warmest (July)
# months - the Netherlands' reference station, and within a degree of this
# installation's region. See mains_water_temperature_c.
CLIMATE_ANNUAL_MEAN_C = 10.5
CLIMATE_COLDEST_MONTH_C = 3.6
CLIMATE_WARMEST_MONTH_C = 18.2

# The temperature hot water is used at (deg C): EN 16147 rates a tank's hot
# water as V40, its volume mixed with mains water to 40 degC, a shower's
# temperature. Mixed from a tank above it, a draw takes the heat that lifts its
# whole volume from the mains to this, whatever the tank's own temperature.
MIXED_TAP_WATER_C = 40.0

# Physical constants of indoor air at room conditions, not fit parameters.
RHO_AIR_KG_PER_M3 = 1.2
CP_AIR_J_PER_KG_K = 1005.0

# Sensible heat released by one adult at rest / light activity (ASHRAE
# Fundamentals, Handbook chapter on internal heat gain). Only the sensible part
# enters a temperature balance; the ~45 W latent part adds moisture, which does
# not affect this ODE - stated here as an explicit modelling assumption rather
# than silently dropped.
Q_PERSON_SENSIBLE_W = 75.0


def tank_state_space(
    volume_l: float,
    ua_top_w_per_k: float,
    ua_bottom_w_per_k: float,
    ua_mix_w_per_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space matrices for dx/dt = A x + B u.

    x = [T_top, T_bottom], u = [T_ambient, Q_in]. Equal top/bottom volume split is the
    simplest unbiased assumption available: there is no sensor for the thermocline
    position, so both nodes share one capacity C_node.
    """

    c_node = RHO_WATER_KG_PER_L * (volume_l / 2.0) * CP_WATER_J_PER_KG_K

    a = (
        np.array(
            [
                [-(ua_mix_w_per_k + ua_top_w_per_k), ua_mix_w_per_k],
                [ua_mix_w_per_k, -(ua_mix_w_per_k + ua_bottom_w_per_k)],
            ]
        )
        / c_node
    )

    b = (
        np.array(
            [
                [ua_top_w_per_k, 0.0],
                [ua_bottom_w_per_k, 1.0],
            ]
        )
        / c_node
    )

    return a, b


def tank_stratification_k(top_c, bottom_c, mixing, held_before=None) -> np.ndarray:
    """The stratification the cold layer below the tank's sensors is read from
    (K, see BoilerThermalModel.mixed_temperature): the top sensor less the
    bottom one, held at its largest since the tank was last mixed. Arrays in
    time order; mixing where the tank heats.

    Tapping fills that layer from the bottom and only mixing empties it, so at
    rest it does not shrink - whereas the sensors' difference does, whenever
    the top steps down a sensor step (TANK_SENSOR_RESOLUTION_K) on standing
    loss. Read from the difference itself, a tank at rest warmed 1.2 K the
    moment its top stepped from 46.5 to 46.0 degC over a 45.5 degC bottom, and
    cooled 1.7 K again on the bottom's next step (real data, 30 September
    2026). While the tank heats, the difference as it is: what a run leaves is
    where the layer starts from after it. Before the first mixing the readings
    show, held from the first reading: the layer before them is not known.
    Missing readings hold what was held.

    held_before is what an earlier update held at the same readings (NaN where
    none): a quarter's reading is its last so far, so the larger difference an
    earlier one showed is otherwise lost and the tank warms again when the
    sensors step back (real data, 29 October 2026 at rest after a tap: 2.0 K,
    then 1.5 K on the next update, the SoC 74 -> 80 %).
    """

    spread = np.asarray(top_c, dtype=float) - np.asarray(bottom_c, dtype=float)
    mixing = np.asarray(mixing, dtype=bool)

    if held_before is not None:
        spread = np.where(
            mixing, spread, np.fmax(spread, np.asarray(held_before, dtype=float))
        )

    held = spread.copy()

    for i in range(1, len(held)):
        if not (mixing[i] or mixing[i - 1]):
            held[i] = np.fmax(held[i - 1], spread[i])

    return held


def tank_sensor_temperatures(
    model: BoilerThermalModel, hot_c, mixed_c, mains_c
) -> tuple[np.ndarray, np.ndarray]:
    """(top, bottom) the tank's sensors read (deg C) with its heat at mixed_c:
    water at hot_c above the mains water (mains_c) that draws let in at the
    bottom, as much of it as the heat drawn says - the mixed temperature is
    the volumes' mean - and the two apart over a front (thermocline)
    model.thermocline_width_l thick, the water mixed there as it came in. A
    sensor reads the water at its height: the volume below it
    (model.top/bottom_sensor_volume_l). A run stirs the tank to one
    temperature, which is hot_c = mixed_c here.

    Between runs the sensors stay where the run left them until the front
    reaches them: over 116 rests after runs (Jun-Oct 2026) the bottom first
    fell with 40-60 % of the tank's heat drawn, the top with 60-80 %. Read
    off the plan's tank this way, the sensors came out 0.9 K off on rests
    the fit had not seen, against 5.7 (top) and 2.3 K (bottom) for the mixed
    temperature itself. Scalars or arrays."""

    hot = np.asarray(hot_c, dtype=float)
    span = hot - mains_c
    cold_l = model.volume_l * np.clip(
        np.divide(hot - mixed_c, span, out=np.zeros_like(span), where=span > 0),
        0.0,
        1.0,
    )

    def at(volume_below_l):
        return mains_c + span * ndtr(
            (volume_below_l - cold_l) / model.thermocline_width_l
        )

    return at(model.top_sensor_volume_l), at(model.bottom_sensor_volume_l)


# A start dead time was tried for this planning model and rejected: no heat into
# the tank for a run's first 15 minutes, since real runs show the supply water
# 7-11 degC colder than the tank for ~10 minutes (the loop between heat pump and
# boiler cools down between runs, so heat first flows out of the tank). Scored
# per real run with BoilerThermalIdentifier._planner_run_errors() (72 runs over
# 90 days), it cut the error after the first 15-minute step from ~7.0 K to
# ~2.3 K - but with a heat input the heat pump actually delivers (4.8-6.2 kW)
# every run ended 4-6 K too cold, and matching run ends needed ~7.6 kW, more
# than measured calorimetric output: a fit factor, not physical heat input -
# mid-run the sensors read the tank being stirred from the bottom, where the
# plate exchanger's heat enters, not its mean. Capturing both the start dip and the
# run total needs more than the two-sensor average of a stratified tank.
def lumped_tank_state_space(
    volume_l: float,
    ua_total_w_per_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space for a single lumped tank node: dT/dt = A T + B u,
    u = [T_ambient, Q_in_effective, Q_tap_forecast].

    Used only for MPC planning (and validating that planning model), not for
    the calibrated two-node identification model. Mixing during active heating
    was found to saturate at the sampling-resolution ceiling (UA_mix_active
    pinned at its bound), meaning the tank is practically fully mixed within
    one MPC step - so a single node using the well-identified UA_top+UA_bottom
    sum is a defensible simplification. It also keeps these dynamics linear in
    the binary boiler_on decision: the full two-node model would need a
    disjunctive/big-M reformulation to let UA_mix switch with boiler_on, for
    precision in the individual UA_top/UA_bottom split that isn't there anyway.

    Q_tap_forecast is an additional heat-sink term (cold mains water entering,
    warm water drawn out) - the third B column carries a negative coefficient
    since, unlike Q_in, it removes energy from the tank: C dT/dt = Q_in -
    UA*(T-T_ambient) - Q_tap.
    """

    c_total = RHO_WATER_KG_PER_L * volume_l * CP_WATER_J_PER_KG_K

    a = np.array([[-ua_total_w_per_k / c_total]])
    b = np.array([[ua_total_w_per_k / c_total, 1.0 / c_total, -1.0 / c_total]])

    return a, b


def night_share(local_hour, start_hour: float, end_hour: float, spread_h: float):
    """Share (0-1) of nights a household's habit of the night - asleep, the
    windows open - is under way at a local hour of the day (0-24): begun
    around start_hour in the evening (one past midnight counted on past 24)
    and over around end_hour in the morning, each night within spread_h (h)
    of them. Counted on from noon, so a night is one stretch across
    midnight."""

    hour = np.asarray(local_hour, dtype=float)
    hour = np.where(hour < 12.0, hour + 24.0, hour)

    return expit((hour - start_hour) / spread_h) * expit(
        (end_hour + 24.0 - hour) / spread_h
    )


def envelope_conductance_w_per_k(
    model: BuildingThermalModel, day_outdoor_c, local_hour
):
    """The envelope's conductance (W/K) at a local hour of the day (0-24), on
    a day whose mean outdoor temperature (deg C, over the last 24 hours) is
    day_outdoor_c: the conductance with the windows shut, plus the open
    windows' on the share of nights they are open, while they are (see
    BuildingThermalModel.ua_window_w_per_k). Both shares are logistic: in the
    day - none on cold days, every night on hot ones, the occupants opening
    them to cool the house down - and in the hour, around the household's
    own times of opening and closing them. Their other airing, irregular (a
    bedroom window on an autumn night), is the rooms' heat habit's (see
    BuildingThermalIdentifier.room_heat_habit)."""

    nights = expit(
        (np.asarray(day_outdoor_c, dtype=float) - model.window_half_day_c)
        / model.window_spread_k
    )
    open_now = night_share(
        local_hour,
        model.window_open_hour,
        model.window_close_hour,
        model.window_hour_spread_h,
    )

    return model.ua_envelope_w_per_k + model.ua_window_w_per_k * nights * open_now


def zone_state_space(
    model: BuildingThermalModel,
    ua_envelope_w_per_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space of the zone for dx/dt = A x + B u - what the
    Kalman filter, a rollout and the MPC all drive (see BuildingThermalModel).

    x = [T_living, T_rest, T_slab_living, T_slab_rest] and, where the floor
    has a screed, [T_screed_living, T_screed_rest] after them;
    u = [T_outdoor, Q_internal, Q_living, Q_rest, Q_floor, Q_floor_moved].

    Q_internal and Q_floor are the dwelling's, divided by floor area; Q_living
    and Q_rest land in that room alone: its own glazing's sun, and the
    occupants' heat that is there beyond its floor area's share (asleep, in the
    bedrooms). The floor's heat enters each floor's top
    layer - its screed, or without one its slab - which is the surface its
    room sees; a screed passes it on to the concrete below. Q_floor_moved is
    the floor's heat that lands under the living room beyond its area share,
    out of the rest's: where rooms have closed their loops, the water heats
    only the floors still open (see floor_heat_moved_w) - zero with every loop
    open. The rest's slab exchanges with the living room below it too,
    through its ceiling.

    The envelope at ua_envelope_w_per_k where given - its conductance that
    day (see envelope_conductance_w_per_k) - else at the model's own.
    """

    if ua_envelope_w_per_k is None:
        ua_envelope_w_per_k = model.ua_envelope_w_per_k

    share = model.living_area_fraction
    shares = np.array([share, 1.0 - share])
    c_room = model.c_air_j_per_k * shares
    c_screed = model.screed_capacity_j_per_k * shares
    c_slab = model.c_mass_j_per_k * shares - c_screed

    if np.any(c_slab <= 0.0):
        raise ValueError("The screed holds more heat than the slabs it lies on.")

    ua_surface = model.ua_air_mass_w_per_k * shares
    ua_down = model.screed_to_slab_w_per_k * shares
    ua_env = ua_envelope_w_per_k * np.array(
        [model.living_envelope_fraction, 1.0 - model.living_envelope_fraction]
    )
    ua_rooms = model.ua_rooms_w_per_k
    screed = bool(model.screed_capacity_j_per_k)
    size = 6 if screed else 4

    a = np.zeros((size, size))
    b = np.zeros((size, 6))

    for room, other in ((0, 1), (1, 0)):
        slab = room + 2
        top = room + 4 if screed else slab
        c_top = c_screed[room] if screed else c_slab[room]

        a[room, room] = -(ua_env[room] + ua_rooms + ua_surface[room]) / c_room[room]
        a[room, other] = ua_rooms / c_room[room]
        a[room, top] = ua_surface[room] / c_room[room]
        a[top, room] = ua_surface[room] / c_top
        a[top, top] = -ua_surface[room] / c_top

        b[room, 0] = ua_env[room] / c_room[room]
        b[room, 1] = shares[room] / c_room[room]
        b[room, 2 + room] = 1.0 / c_room[room]
        b[top, 4] = shares[room] / c_top
        b[top, 5] = (1.0 if room == 0 else -1.0) / c_top

        if screed:
            a[top, top] -= ua_down[room] / c_top
            a[top, slab] = ua_down[room] / c_top
            a[slab, top] = ua_down[room] / c_slab[room]
            a[slab, slab] = -ua_down[room] / c_slab[room]

    ceiling = model.ua_ceiling_w_per_k
    a[0, 0] -= ceiling / c_room[0]
    a[0, 3] += ceiling / c_room[0]
    a[3, 3] -= ceiling / c_slab[1]
    a[3, 0] += ceiling / c_slab[1]

    return a, b


def zone_size(model: BuildingThermalModel) -> int:
    """How many states the zone has (see zone_state_space)."""

    return 6 if model.screed_capacity_j_per_k else 4


def _surfaces(model: BuildingThermalModel) -> tuple[int, int]:
    """Each floor's top layer, the surface its room sees."""

    return ZONE_SCREED_STATES if model.screed_capacity_j_per_k else ZONE_SLAB_STATES


def zone_observations(model: BuildingThermalModel) -> np.ndarray:
    """What the zone's thermometers read of its state, one row each: the
    living room's thermostat and the rest of the house's sensors averaged by
    floor area.

    Operative temperatures: a wall-mounted sensor exchanges longwave radiation
    with the floor as well as the room, so it follows the floor's surface in
    part (see BuildingThermalModel.sensor_mass_fraction). A measurement
    equation, not a heat balance: it moves no energy.
    """

    fraction = model.sensor_mass_fraction
    rows = np.zeros((2, zone_size(model)))

    for room, surface in enumerate(_surfaces(model)):
        rows[room, room] = 1.0 - fraction
        rows[room, surface] = fraction

    return rows


def zone_observation(model: BuildingThermalModel) -> np.ndarray:
    """Row vector h with T = h x: what the thermostat reads, the temperature
    comfort is set in and judged on."""

    return zone_observations(model)[0]


def zone_slab(model: BuildingThermalModel) -> np.ndarray:
    """Row vector of the slabs' mean temperature, by floor area: the
    building's thermal mass."""

    share = model.living_area_fraction
    row = np.zeros(zone_size(model))
    row[list(ZONE_SLAB_STATES)] = [share, 1.0 - share]

    return row


def zone_floor(
    model: BuildingThermalModel, living_fraction: float | None = None
) -> np.ndarray:
    """Row vector of the temperature the floor circuit delivers its heat
    against, the one water running through both floors: their screeds' mean
    by the share of the water under each (see floor_living_fraction; by floor
    area, every loop open, without one), or without screeds their slabs'."""

    share = model.living_area_fraction if living_fraction is None else living_fraction
    row = np.zeros(zone_size(model))
    row[list(_surfaces(model))] = [share, 1.0 - share]

    return row


def floor_living_fraction(open_living, open_rest, living_area_fraction: float):
    """The share of the floor circuit's water - its heat, and the screed it
    sees - that runs under the living room: by open loop area, taking every
    loop to take as much per m2. open_living and open_rest are each floor's
    open loop area as a share of the zone's floor area: (living_area_fraction,
    1 - it) with every loop open. With none open nothing flows, and floor
    area keeps it defined - as where none was measured (NaN)."""

    open_living = np.asarray(open_living, dtype=float)
    open_total = open_living + np.asarray(open_rest, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            open_total > 0.0, open_living / open_total, living_area_fraction
        )


def floor_heat_moved_w(floor_w, living_fraction, living_area_fraction: float):
    """The floor's heat under the living room beyond its floor-area share
    (W), out of the rest's: zone_state_space's Q_floor_moved. Zero with
    every loop open."""

    return floor_w * (living_fraction - living_area_fraction)


# The zone's slab and screed states (see zone_state_space): each must stay
# above the dew point while cooling - a screed is the floor's surface, the
# rest's slab the living room's ceiling.
ZONE_SLAB_STATES = (2, 3)
ZONE_SCREED_STATES = (4, 5)


def floor_heat_w(
    flow_lpm: np.ndarray,
    supply_temperature_c: np.ndarray,
    return_temperature_c: np.ndarray,
) -> np.ndarray:
    """Calorimetric heat delivered to the floor circuit: Q = m_dot * cp * dT.

    Signed by construction: during heating the supply is warmer than the return
    and Q is positive, during cooling it is colder and Q is negative. That is
    precisely why one model covers both modes - no mode flag enters here.
    """

    mass_flow_kg_per_s = RHO_WATER_KG_PER_L * flow_lpm / 60.0

    return (
        mass_flow_kg_per_s
        * CP_WATER_J_PER_KG_K
        * (supply_temperature_c - return_temperature_c)
    )


def floor_loops_conductance_w_per_k(pipe_ua_w_per_k, open_fraction, flow_lpm):
    """From the supply water to the floor (W/K), the water cooling through
    the loops included: Q = G (supply - floor). The loops taken as one heat
    exchanger against one floor temperature (effectiveness-NTU, see
    floor_pipe_conductance_w_per_k), G = m_dot c_p (1 - exp(-open UA /
    (m_dot c_p))): the pipes' UA, every loop open, by the open share of the
    floor area, and the flow through those open. Closing loops leaves the
    pump's flow to fewer of them, so each cools its water less (26 Oct 2026:
    18.0 to 16.0 L/min with 57% of the floor open). No flow, none.
    """

    capacity_w_per_k = (
        RHO_WATER_KG_PER_L * np.asarray(flow_lpm, dtype=float) / 60.0
    ) * CP_WATER_J_PER_KG_K

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            capacity_w_per_k > 0.0,
            capacity_w_per_k
            * (1.0 - np.exp(-open_fraction * pipe_ua_w_per_k / capacity_w_per_k)),
            0.0,
        )


def floor_circuit_conductance_w_per_k(
    model: FloorCircuitModel, open_fraction=1.0, flow_lpm=None
):
    """The floor circuit's conductance (see floor_loops_conductance_w_per_k):
    at the flow its runs ran at without one. A model from before its pipes'
    UA was identified scales its conductance by the open share alone."""

    if not model.pipe_ua_w_per_k:
        return model.conductance_w_per_k * open_fraction

    return floor_loops_conductance_w_per_k(
        model.pipe_ua_w_per_k,
        open_fraction,
        model.flow_lpm if flow_lpm is None else flow_lpm,
    )


def floor_pipe_conductance_w_per_k(
    flow_lpm: np.ndarray,
    supply_temperature_c: np.ndarray,
    return_temperature_c: np.ndarray,
    mass_temperature_c: np.ndarray,
) -> np.ndarray:
    """The conductance from the water to the mass that a run's temperatures
    imply (W/K), the loop taken as a heat exchanger against one temperature:
    return = mass + (supply - mass) * exp(-UA / (m_dot * cp)) (effectiveness-
    NTU). Either sign of heat. NaN where the return does not lie between the
    supply and the mass, which no conductance explains.
    """

    capacity_w_per_k = RHO_WATER_KG_PER_L * flow_lpm / 60.0 * CP_WATER_J_PER_KG_K
    remaining = (return_temperature_c - mass_temperature_c) / (
        supply_temperature_c - mass_temperature_c
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (remaining > 0.0) & (remaining < 1.0),
            -capacity_w_per_k * np.log(remaining),
            np.nan,
        )


def solar_gain_w(
    a_eff_m2: float,
    shutter_open_fraction: np.ndarray,
    facade_irradiance: np.ndarray,
) -> np.ndarray:
    """Q_sol = A_eff * f_shutter * I_facade.

    A roller shutter covers the glass from the top down, so the *unobstructed
    glass area* scales linearly with its open position - the linearity is
    geometric, not an assumed response curve. A fully closed shutter is taken as
    fully opaque; real slat gaps transmit a few percent, which would show up as
    an underprediction on sunny days with the shutter shut, and only then is
    there evidence for a residual-transmittance parameter.
    """

    return a_eff_m2 * shutter_open_fraction * facade_irradiance


def extension_shaded_fraction(
    sun_from_normal_deg: np.ndarray,
    east_depth_ratio: float,
    west_depth_ratio: float,
) -> np.ndarray:
    """Share of a glazing's width in the shadow of the side walls of the
    neighbours' extensions on either side of it, for the direct sun only.

    A wall perpendicular to the facade at the glazing's edge, at least as
    tall as the glazing (a single-storey extension beside ground-floor glass),
    casts a shadow depth * tan(gamma) wide across it, gamma the sun's
    horizontal angle from the facade's normal (positive towards the west):
    the east wall shades in the morning, the west wall in the afternoon. The
    sky the walls hide takes a fixed share of the diffuse light, which the
    effective aperture already absorbs.
    """

    gamma = np.radians(np.asarray(sun_from_normal_deg, dtype=float))
    ratio = np.where(gamma > 0.0, west_depth_ratio, east_depth_ratio)
    # From 90 degrees the sun is behind the facade, which has no direct sun
    # left to shade.
    shadow = np.where(np.abs(gamma) < np.pi / 2.0, ratio * np.tan(np.abs(gamma)), 1.0)

    return np.clip(shadow, 0.0, 1.0)


def internal_gain_w(
    baseload_w: np.ndarray,
    internal_gain_fraction: float,
    occupants: np.ndarray,
) -> np.ndarray:
    """Q_int = f_indoor * P_baseload + n_occupants * Q_PERSON_SENSIBLE_W.

    Household electricity ends up as heat inside the building, so the measured
    baseload power is a direct measurement of appliance and lighting gain rather
    than something to fit. Only its in-zone fraction is identified, because the
    sensor covers the whole house while the model covers one zone.
    """

    return internal_gain_fraction * baseload_w + occupants * Q_PERSON_SENSIBLE_W


# Magnus over water (Sonntag's coefficients), in both directions: the relation
# the Xiaomi sensors report their dew point by (their dew point is this formula
# on their own temperature and humidity to within 0.05 K on real data), so an
# indoor sensor's dew point and one computed from Open-Meteo's humidity are the
# same quantity.
MAGNUS_A = 17.62
MAGNUS_B_C = 243.12
MAGNUS_E0_PA = 611.2


def vapour_pressure_pa(dew_point_c):
    """The partial pressure of water vapour in air with this dew point (Pa):
    the saturation pressure there. Scalars or arrays."""

    return MAGNUS_E0_PA * np.exp(MAGNUS_A * dew_point_c / (MAGNUS_B_C + dew_point_c))


def dew_point_c(vapour_pressure_pa):
    """The dew point of air holding water vapour at this pressure (deg C)."""

    g = np.log(vapour_pressure_pa / MAGNUS_E0_PA)

    return MAGNUS_B_C * g / (MAGNUS_A - g)


def dew_point_from_humidity_c(temperature_c, relative_humidity_pct):
    """The dew point of air at this temperature and relative humidity (deg C)."""

    return dew_point_c(vapour_pressure_pa(temperature_c) * relative_humidity_pct / 100)


def indoor_dew_point_c(
    model: DewPointModel,
    dew_point_now_c,
    outdoor_dew_point_c,
    dt_hours: float,
) -> np.ndarray:
    """The indoor dew point at the start of each step, from the one now and the
    outdoor dew point over each step (steps along the last axis; one run per
    leading index, so many starts can be run at once).

    A moisture balance in vapour pressure, which is what ventilation mixes
    linearly: de/dt = (e_out + delta_e - e) / tau. tau is the ventilation's
    time constant stretched by what walls and furnishings buffer; delta_e is
    the surplus the occupants' own moisture keeps indoors, their production
    over the ventilation. Held constant over each step, so each advances
    exactly: e[k+1] = e[k] + (1 - exp(-dt / tau)) (e_out[k] + delta_e - e[k]).
    """

    weight = 1.0 - np.exp(-dt_hours / model.time_constant_hours)
    outdoor = vapour_pressure_pa(np.asarray(outdoor_dew_point_c, dtype=float))
    e = vapour_pressure_pa(np.asarray(dew_point_now_c, dtype=float))
    pressures = []

    for k in range(outdoor.shape[-1]):
        pressures.append(e)
        e = e + weight * (outdoor[..., k] + model.moisture_surplus_pa - e)

    return dew_point_c(np.stack(pressures, axis=-1))


def mains_water_temperature_c(day_of_year: int) -> float:
    """The cold mains water a tap draw lets into the tank (deg C) on that day:
    Burch and Christensen's correlation (NREL, 2007; EnergyPlus's
    Site:WaterMainsTemperature), the year's air temperature cycle damped and
    delayed by the ground the mains run through, in its own degrees F. Not
    measured here, and not readable from the tank either: the bottom sensor
    sits above the cold layer and was still falling 2 K a minute when the
    largest draws ended. Below the lowest it read in such draws in every month
    (Aug-Oct 2026: 17.3, 16.7 and 15.5 degC against 17.5, 17.8 and 16.5).
    """

    annual_f = CLIMATE_ANNUAL_MEAN_C * 9.0 / 5.0 + 32.0
    swing_f = (CLIMATE_WARMEST_MONTH_C - CLIMATE_COLDEST_MONTH_C) * 9.0 / 5.0
    ratio = 0.4 + 0.01 * (annual_f - 44.0)
    lag_days = 35.0 - 1.0 * (annual_f - 44.0)
    mains_f = (annual_f + 6.0) + ratio * swing_f / 2.0 * np.sin(
        np.radians(0.986 * (day_of_year - 15.0 - lag_days) - 90.0)
    )

    return float((mains_f - 32.0) * 5.0 / 9.0)
