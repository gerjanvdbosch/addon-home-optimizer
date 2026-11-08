"""The esphome-ecodan-hp auto-adaptive heating law
(components/optimizer/auto_adaptive.cpp), underfloor profile with the S-curve,
Smart Boost off. The room's error below setpoint + bias, as a share x of an
error range, sets the water's rise over the return:
rise = min_rise + (3x^2 - 2x^3)(max_rise - min_rise), both widened towards
the cold by a cold factor of the unit's outdoor sensor. With the room past
its target (error < 0) the rise is the base MIN_RISE_K, not the cold-widened
least: aiming just below the room asks for the heat pump's least heat.
"""

import math

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


def min_rise_k(outdoor_c: float) -> float:
    """The least rise over the return (K) the controller asks for with the
    room at or below its target: 2 K when mild, widening towards the cold
    (3.4 K at 1.5 degC, ~4.2 kW at 18 L/min). Past the target it asks for
    MIN_RISE_K."""

    return MIN_RISE_K + _cold_factor(outdoor_c) * (COLD_MIN_RISE_K - MIN_RISE_K)


def setpoint_bias_k(
    heat_w: float,
    flow_lpm: float,
    outdoor_c: float,
    room_c: float,
    setpoint_c: float,
    maximum_c: float,
    least_heat_w: float = 0.0,
) -> float:
    """The setpoint bias (K) at which the controller asks for heat_w into the
    floor: the law inverted for the room's error, the room target it needs
    never past the comfort band's maximum. Rounded up to the bias step, so at
    least that heat.

    No more than the heat pump's least heat (least_heat_w, its compressor at
    the lowest speed) or less than the least rise at or below the target -
    which in the cold is more than that (4.1 kW at 2 degC, 18 L/min) - is
    asked for with the target one step below the room: the base rise, which
    the compressor at its lowest speed gives at least. Asked for through the
    curve instead, the least heat sets a supply the lowest speed only just
    reaches over a cool return, and the controller speeds the compressor up to
    reach it (8 Nov 2026: 26 -> 39 Hz for 20 minutes at -0.7/-0.8 K) - at a
    lower COP than its lowest speed (steady heating 10 Oct-8 Nov: 7% above
    the fitted COP at 26 Hz, 4-6% below it from 38 Hz).
    """

    least_k = min_rise_k(outdoor_c)

    # In whole watts: at its least the plan's heat comes back from the solver
    # a fraction of a watt either side of it.
    if round(heat_w) <= max(least_k * water_w_per_k(flow_lpm), least_heat_w):
        # Strictly below the room, within a step of it.
        bias_k = (math.ceil(round((room_c - setpoint_c) / BIAS_STEP_K, 6)) - 1) * (
            BIAS_STEP_K
        )

        return round(min(max(bias_k, -BIAS_LIMIT_K), BIAS_LIMIT_K), 1)

    error_range_k = ERROR_RANGE_K * (1.0 - 0.5 * min(_cold_factor(outdoor_c), 1.0))
    share = (heat_w / water_w_per_k(flow_lpm) - least_k) / (MAX_RISE_K - least_k)
    share = min(max(share, 0.0), 1.0)
    # The inverse of 3x^2 - 2x^3 on [0, 1].
    x = 0.5 - math.sin(math.asin(1.0 - 2.0 * share) / 3.0)
    target_c = min(room_c + x * error_range_k, maximum_c)

    bias_k = math.ceil(round((target_c - setpoint_c) / BIAS_STEP_K, 6)) * BIAS_STEP_K

    return round(min(max(bias_k, -BIAS_LIMIT_K), BIAS_LIMIT_K), 1)
