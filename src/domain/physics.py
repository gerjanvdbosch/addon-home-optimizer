"""Physical relations and balances of the modelled system.

The energy balances themselves, the constants they are written in, and the
heat flows that enter them - separated from how the parameters in them are
identified (features/) and from how they are numerically integrated
(domain/dynamics.py). Everything here is a statement about the physics, so it
is equally available to the identifiers that fit these models, to the filter
that estimates their states and to the planner that acts on them.
"""

import numpy as np

from domain.models import BuildingThermalModel, DewPointModel

# Physical constants (water), not fit parameters.
RHO_WATER_KG_PER_L = 1.0
CP_WATER_J_PER_KG_K = 4186.0

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


def layered_tank_state_space(
    volume_l: float,
    ua_w_per_k: float,
    layer_fraction: float,
    k_top_bottom_w_per_k: float,
    k_bottom_layer_w_per_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space for dx/dt = A x + B T_ambient of a tank at rest,
    x = [T_top, T_bottom, T_layer]: the two sensed nodes and the cold layer
    below the bottom sensor (see BoilerThermalModel.cold_layer_fraction).

        C_top    dT_top/dt    = UA_top (T_amb - T_top) + K_tb (T_bottom - T_top)
        C_bottom dT_bottom/dt = UA_bottom (T_amb - T_bottom)
                                + K_tb (T_top - T_bottom) + K_bl (T_layer - T_bottom)
        C_layer  dT_layer/dt  = UA_layer (T_amb - T_layer) + K_bl (T_bottom - T_layer)

    The layer holds layer_fraction of the volume, the sensed nodes half of the
    rest each. The standing loss is spread over the nodes in proportion to their
    volume: the same loss per height of the cylinder, nothing being known that
    would make one part lose more. The internal couplings conserve the tank's
    energy, so only UA takes it out.
    """

    shares = np.array([(1 - layer_fraction) / 2, (1 - layer_fraction) / 2])
    shares = np.append(shares, layer_fraction)
    capacities = RHO_WATER_KG_PER_L * volume_l * CP_WATER_J_PER_KG_K * shares
    k_tb, k_bl = k_top_bottom_w_per_k, k_bottom_layer_w_per_k
    coupling = np.array(
        [[-k_tb, k_tb, 0.0], [k_tb, -k_tb - k_bl, k_bl], [0.0, k_bl, -k_bl]]
    )
    losses = ua_w_per_k * shares

    a = (coupling - np.diag(losses)) / capacities[:, None]
    b = (losses / capacities)[:, None]

    return a, b


def tank_stratification_k(top_c, bottom_c, mixing) -> np.ndarray:
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
    """

    spread = np.asarray(top_c, dtype=float) - np.asarray(bottom_c, dtype=float)
    mixing = np.asarray(mixing, dtype=bool)
    held = spread.copy()

    for i in range(1, len(held)):
        if not (mixing[i] or mixing[i - 1]):
            held[i] = np.fmax(held[i - 1], spread[i])

    return held


# A start dead time was tried for this planning model and rejected: no heat into
# the tank for a run's first 15 minutes, since real runs show the supply water
# 7-11 degC colder than the tank for ~10 minutes (the loop between heat pump and
# boiler cools down between runs, so heat first flows out of the tank). Scored
# per real run with BoilerThermalIdentifier._planner_run_errors() (72 runs over
# 90 days), it cut the error after the first 15-minute step from ~7.0 K to
# ~2.3 K - but with a heat input the heat pump actually delivers (4.8-6.2 kW)
# every run ended 4-6 K too cold, and matching run ends needed ~7.6 kW, more
# than measured calorimetric output: a fit factor compensating for the sensor
# nearest the coil running ahead of the rest of the tank, not physical heat
# input. Capturing both the start dip and the run total needs more than the
# two-sensor average of a stratified tank.
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


def zone_state_space(
    model: BuildingThermalModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space of the zone for dx/dt = A x + B u - what the
    Kalman filter, a rollout and the MPC all drive (see BuildingThermalModel).

    x = [T_living, T_rest, T_slab_living, T_slab_rest],
    u = [T_outdoor, Q_internal, Q_solar_living, Q_solar_rest, Q_floor].

    Q_internal and Q_floor are the dwelling's, divided by floor area; each
    room's sun is its own glazing's.
    """

    share = model.living_area_fraction
    shares = np.array([share, 1.0 - share])
    c_room = model.c_air_j_per_k * shares
    c_slab = model.c_mass_j_per_k * shares
    ua_slab = model.ua_air_mass_w_per_k * shares
    ua_env = model.ua_envelope_w_per_k * np.array(
        [model.living_envelope_fraction, 1.0 - model.living_envelope_fraction]
    )
    ua_rooms = model.ua_rooms_w_per_k

    a = np.zeros((4, 4))
    b = np.zeros((4, 5))

    for room, other in ((0, 1), (1, 0)):
        slab = room + 2
        a[room, room] = -(ua_env[room] + ua_rooms + ua_slab[room]) / c_room[room]
        a[room, other] = ua_rooms / c_room[room]
        a[room, slab] = ua_slab[room] / c_room[room]
        a[slab, room] = ua_slab[room] / c_slab[room]
        a[slab, slab] = -ua_slab[room] / c_slab[room]

        b[room, 0] = ua_env[room] / c_room[room]
        b[room, 1] = shares[room] / c_room[room]
        b[room, 2 + room] = 1.0 / c_room[room]
        b[slab, 4] = shares[room] / c_slab[room]

    return a, b


def zone_observations(model: BuildingThermalModel) -> np.ndarray:
    """What the zone's thermometers read of its state, one row each: the
    living room's thermostat and the rest of the house's sensors averaged by
    floor area.

    Operative temperatures: a wall-mounted sensor exchanges longwave radiation
    with the floor as well as the room, so it follows its slab in part (see
    BuildingThermalModel.sensor_mass_fraction). A measurement equation, not a
    heat balance: it moves no energy.
    """

    fraction = model.sensor_mass_fraction

    return np.array(
        [
            [1.0 - fraction, 0.0, fraction, 0.0],
            [0.0, 1.0 - fraction, 0.0, fraction],
        ]
    )


def zone_observation(model: BuildingThermalModel) -> np.ndarray:
    """Row vector h with T = h x: what the thermostat reads, the temperature
    comfort is set in and judged on."""

    return zone_observations(model)[0]


def zone_slab(model: BuildingThermalModel) -> np.ndarray:
    """Row vector of the slabs' mean temperature, by floor area: what the
    floor circuit delivers its heat against, the one water runs through both."""

    share = model.living_area_fraction

    return np.array([0.0, 0.0, share, 1.0 - share])


# The zone's slab states (see zone_state_space): each must stay above the dew
# point while cooling, since each has a floor surface the air touches.
ZONE_SLAB_STATES = (2, 3)


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
