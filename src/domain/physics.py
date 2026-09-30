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
    Kalman filter, a rollout and the MPC all drive.

    x = [T_air, T_mass], u = [T_outdoor, Q_internal, Q_solar, Q_floor]:

        C_air  dT_air/dt  = UA_env (T_out - T_air) + UA_am (T_mass - T_air) + Q_int
        C_mass dT_mass/dt = UA_am  (T_air - T_mass) + Q_sol + f_floor Q_floor

    Q_solar and Q_floor drive the mass node, not the air node. The floor
    circuit physically runs inside the screed, and air is effectively
    transparent to shortwave radiation, which is absorbed by floor and
    furnishings - this is the same structure as the boiler's, where heat is
    supplied at the bottom rather than uniformly. It is also what produces the
    observed lag between sun or compressor and room temperature, without any
    added delay term.
    """

    ua_env = model.ua_envelope_w_per_k
    ua_am = model.ua_air_mass_w_per_k
    c_air = model.c_air_j_per_k
    c_mass = model.c_mass_j_per_k

    a = np.array(
        [
            [-(ua_env + ua_am) / c_air, ua_am / c_air],
            [ua_am / c_mass, -ua_am / c_mass],
        ]
    )

    b = np.array(
        [
            [ua_env / c_air, 1.0 / c_air, 0.0, 0.0],
            [0.0, 0.0, 1.0 / c_mass, model.floor_heat_fraction / c_mass],
        ]
    )

    return a, b


def zone_observation(
    model: BuildingThermalModel,
) -> np.ndarray:
    """Row vector h with T_measured = h x: what a thermostat reads from the
    zone's states.

    An operative temperature: a wall-mounted sensor exchanges longwave
    radiation with the surfaces around it, so it follows the mass node as well
    as the air (see BuildingThermalModel.sensor_mass_fraction). This is a measurement
    equation, not a heat balance - it moves no energy, and dropping it would
    only mean claiming the sensor reads pure air.
    """

    fraction = model.sensor_mass_fraction

    return np.array([1.0 - fraction, fraction])


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
