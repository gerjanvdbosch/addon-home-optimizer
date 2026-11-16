"""The esphome-ecodan-hp auto-adaptive heating law
(components/optimizer/auto_adaptive.cpp), underfloor profile with the S-curve,
Smart Boost off. The room's error below setpoint + bias, as a share x of an
error range, sets the water's rise over the return:
rise = min_rise + (3x^2 - 2x^3)(max_rise - min_rise), both widened towards
the cold by a cold factor of the unit's outdoor sensor. With the room past
its target (error < 0) the rise is the base MIN_RISE_K, not the cold-widened
least: aiming just below the room asks for the heat pump's least heat.

The law sets the flow setpoint; the supply the heat pump holds sits above
it by an overshoot that shrinks as the rise asked grows (an installation's
own, fitted: see HeatPumpCOPModel.supply_overshoot_k), so the heat given is
the water's flow times the rise asked plus that overshoot.
"""

import math

import numpy as np

from domain.physics import CP_WATER_J_PER_KG_K, RHO_WATER_KG_PER_L

MIN_RISE_K = 2.0
COLD_MIN_RISE_K = 4.0
MAX_RISE_K = 6.5
ERROR_RANGE_K = 2.0
# The cold factor runs from 0 at MILD to 1.5 at COLD, quadratically.
MILD_C = 15.0
COLD_C = -5.0
COLD_FACTOR_MAX = 1.5
# The bias number's range and step.
BIAS_LIMIT_K = 2.5
BIAS_STEP_K = 0.1


def _cold_factor(outdoor_c: float) -> float:
    return (
        COLD_FACTOR_MAX
        * ((MILD_C - min(max(outdoor_c, COLD_C), MILD_C)) / (MILD_C - COLD_C)) ** 2
    )


def water_w_per_k(flow_lpm: float) -> float:
    """The heat the water carries per K of rise (W/K) at this flow."""

    return flow_lpm / 60.0 * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K


# (K over the flow setpoint at the base rise, rise K asked where it is gone):
# HeatPumpCOPModel.supply_overshoot. None: the supply at the setpoint.
NO_OVERSHOOT = (0.0, 0.0)


def supply_overshoot_k(
    rise_k: float, overshoot: tuple[float, float] = NO_OVERSHOOT
) -> float:
    """How far (K) the supply sits above the flow setpoint, the controller
    asking for rise_k over the return: linear from the base rise to none.
    Scalars or arrays."""

    at_base_k, gone_k = overshoot

    if not at_base_k:
        return 0.0 * rise_k

    return at_base_k * np.maximum((gone_k - rise_k) / (gone_k - MIN_RISE_K), 0.0)


def rise_heat_w(
    rise_k: float, flow_lpm: float, overshoot: tuple[float, float] = NO_OVERSHOOT
) -> float:
    """The heat (W) the floor takes with the controller asking for rise_k
    over the return: the rise asked plus the supply's overshoot."""

    return (rise_k + supply_overshoot_k(rise_k, overshoot)) * water_w_per_k(flow_lpm)


def asked_rise_k(
    heat_w: float, flow_lpm: float, overshoot: tuple[float, float] = NO_OVERSHOOT
) -> float:
    """The rise over the return (K) to ask for heat_w into the floor: the
    inverse of rise_heat_w, the overshoot falling with the rise asked."""

    given_k = heat_w / water_w_per_k(flow_lpm)
    at_base_k, gone_k = overshoot

    if not at_base_k or given_k >= gone_k:
        return given_k

    per_k = at_base_k / (gone_k - MIN_RISE_K)

    return (given_k - per_k * gone_k) / (1.0 - per_k)


def min_rise_k(outdoor_c: float) -> float:
    """The least rise over the return (K) the controller asks for with the
    room at or below its target: 2 K when mild, widening towards the cold
    (3.4 K at 1.5 degC, ~4.2 kW at 18 L/min). Past the target it asks for
    MIN_RISE_K."""

    return MIN_RISE_K + _cold_factor(outdoor_c) * (COLD_MIN_RISE_K - MIN_RISE_K)


def least_bias_k(room_c: float, setpoint_c: float) -> float:
    """The setpoint bias (K) at which the controller asks for its least: the
    target strictly below the room, within a step of it, so the base rise."""

    bias_k = (math.ceil(round((room_c - setpoint_c) / BIAS_STEP_K, 6)) - 1) * (
        BIAS_STEP_K
    )

    return round(min(max(bias_k, -BIAS_LIMIT_K), BIAS_LIMIT_K), 1)


def setpoint_bias_k(
    heat_w: float,
    flow_lpm: float,
    outdoor_c: float,
    room_c: float,
    setpoint_c: float,
    maximum_c: float,
    least_heat_w: float = 0.0,
    round_up: bool = True,
    overshoot: tuple[float, float] = NO_OVERSHOOT,
) -> float:
    """The setpoint bias (K) at which the controller asks for heat_w into the
    floor: the law inverted for the room's error, the room target it needs
    never past the comfort band's maximum. Rounded up to the bias step, so at
    least that heat - or down (round_up False), so at most that heat.

    No more than the heat pump's least heat (least_heat_w, its compressor at
    the lowest speed) - rounded down also less than the least rise at or
    below the target, which in the cold is more than that (4.1 kW at 2 degC,
    18 L/min) - is asked for with the target one step below the room: the
    base rise, which
    the compressor at its lowest speed gives at least. Asked for through the
    curve instead, the least heat sets a supply the lowest speed only just
    reaches over a cool return, and the controller speeds the compressor up to
    reach it (8 Nov 2026: 26 -> 39 Hz for 20 minutes at -0.7/-0.8 K) - at a
    lower COP than its lowest speed (steady heating 10 Oct-8 Nov: 7% above
    the fitted COP at 26 Hz, 4-6% below it from 38 Hz).
    """

    least_k = min_rise_k(outdoor_c)
    # Between the base rise below the target and the cold-widened least rise
    # at it the controller asks for nothing: rounded down that is the least
    # below the room, rounded up the least rise, the target at the room (the
    # curve's start below).
    rise_k = least_k if not round_up else MIN_RISE_K

    # In whole watts: at its least the plan's heat comes back from the solver
    # a fraction of a watt either side of it.
    if round(heat_w) <= max(rise_heat_w(rise_k, flow_lpm, overshoot), least_heat_w):
        return least_bias_k(room_c, setpoint_c)

    error_range_k = ERROR_RANGE_K * (1.0 - 0.5 * min(_cold_factor(outdoor_c), 1.0))
    share = (asked_rise_k(heat_w, flow_lpm, overshoot) - least_k) / (
        MAX_RISE_K - least_k
    )
    share = min(max(share, 0.0), 1.0)
    # The inverse of 3x^2 - 2x^3 on [0, 1].
    x = 0.5 - math.sin(math.asin(1.0 - 2.0 * share) / 3.0)
    target_c = min(room_c + x * error_range_k, maximum_c)

    steps = round((target_c - setpoint_c) / BIAS_STEP_K, 6)
    bias_k = (math.ceil(steps) if round_up else math.floor(steps)) * BIAS_STEP_K

    return round(min(max(bias_k, -BIAS_LIMIT_K), BIAS_LIMIT_K), 1)
