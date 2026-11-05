"""The esphome-ecodan-hp auto-adaptive heating law
(components/optimizer/auto_adaptive.cpp), underfloor profile with the S-curve,
Smart Boost off. The room's error below setpoint + bias, as a share x of an
error range, sets the water's rise over the return:
rise = min_rise + (3x^2 - 2x^3)(max_rise - min_rise), both widened towards
the cold by a cold factor of the unit's outdoor sensor.
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


def setpoint_bias_k(
    heat_w: float,
    flow_lpm: float,
    outdoor_c: float,
    room_c: float,
    setpoint_c: float,
    maximum_c: float,
) -> float:
    """The setpoint bias (K) at which the controller asks for heat_w into the
    floor: the law inverted for the room's error, the room target it needs
    never past the comfort band's maximum. Rounded up to the bias step, so at
    least that heat.

    The controller's least rise is a floor no bias goes under (2.4 K, ~3 kW at
    8 degC); in the cold it can exceed the heat pump's least heat.
    """

    water_w_per_k = flow_lpm / 60.0 * RHO_WATER_KG_PER_L * CP_WATER_J_PER_KG_K
    cold_factor = (
        COLD_FACTOR_MAX
        * ((MILD_C - min(max(outdoor_c, COLD_C), MILD_C)) / (MILD_C - COLD_C)) ** 2
    )
    min_rise_k = MIN_RISE_K + cold_factor * (COLD_MIN_RISE_K - MIN_RISE_K)
    error_range_k = ERROR_RANGE_K * (1.0 - 0.5 * min(cold_factor, 1.0))
    share = (heat_w / water_w_per_k - min_rise_k) / (MAX_RISE_K - min_rise_k)
    share = min(max(share, 0.0), 1.0)
    # The inverse of 3x^2 - 2x^3 on [0, 1].
    x = 0.5 - math.sin(math.asin(1.0 - 2.0 * share) / 3.0)
    target_c = min(room_c + x * error_range_k, maximum_c)

    bias_k = math.ceil(round((target_c - setpoint_c) / BIAS_STEP_K, 6)) * BIAS_STEP_K

    return round(min(max(bias_k, -BIAS_LIMIT_K), BIAS_LIMIT_K), 1)
