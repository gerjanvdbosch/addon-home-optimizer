"""Numerical integration and state estimation for a linear state space.

Separated from the balances in domain/physics.py on purpose: nothing here
knows what the states mean. Both take the (A, B) of any continuous model and
turn it into discrete steps - which is what keeps the discrete implementation
identical in meaning to the continuous formulation rather than a second,
slightly different model.
"""

import numpy as np
from scipy.linalg import expm


def discretize_zoh(
    a: np.ndarray,
    b: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization via the matrix exponential (Van Loan).

    Keeps the discrete step physically identical to the continuous ODE for any dt,
    instead of introducing Euler-integration error.
    """

    n, m = b.shape

    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = a
    augmented[:n, n:] = b

    exponent = expm(augmented * dt_seconds)

    return exponent[:n, :n], exponent[:n, n:]


def kalman_states(
    a: np.ndarray,
    b: np.ndarray,
    measured: np.ndarray,
    inputs: np.ndarray,
    dt_seconds: np.ndarray,
    process_noise_w: float,
    measurement_variance: float,
) -> np.ndarray:
    """Filtered state estimate at every sample, from the measured air
    temperature alone.

    Only the first state is observed - the room thermometer - so any other
    state has to be inferred from how the measured one moves relative to what
    the model predicted. That is what a Kalman filter does, and it is the
    honest way to start a rollout: hard-resetting the measured state while
    letting an unmeasured one free-run leaves the two inconsistent with each
    other, which flatters a single-state model and penalises a multi-state one.

    It is also not only an evaluation device. An MPC must know where it starts
    from at solve time, including a screed temperature nothing measures, so
    this is a missing part of the system rather than a test harness.

    Process noise is expressed as an unmodelled HEAT FLOW (W) entering through
    the same channel as the internal gains, not as an abstract covariance: the
    disturbance this is standing in for - ventilation through an opened window,
    a wood stove, a visitor - is a heat flow, so its magnitude can be reasoned
    about physically.
    """

    n = a.shape[0]

    state = np.concatenate(([measured[0]], np.full(n - 1, measured[0])))
    covariance = np.eye(n) * measurement_variance

    estimates = np.empty((len(measured), n))
    estimates[0] = state

    cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for i in range(1, len(measured)):
        dt = float(dt_seconds[i])
        key = round(dt, 3)

        if key not in cache:
            a_d, b_d = discretize_zoh(a, b, dt)
            # The disturbance enters where an unmodelled indoor heat flow
            # would, and its effect on the state is what the DISCRETE input
            # matrix says an input of that size does over one step. Building it
            # from the continuous B instead understates it by a factor dt^2 -
            # here about a million - which silently turns the filter into a
            # free-running simulation that ignores the measurement.
            disturbance = b_d[:, 1:2]
            cache[key] = (
                a_d,
                b_d,
                (process_noise_w**2) * (disturbance @ disturbance.T),
            )

        a_d, b_d, process_covariance = cache[key]

        state = a_d @ state + b_d @ inputs[i - 1]
        covariance = a_d @ covariance @ a_d.T + process_covariance

        # The air temperature is the first state and the only measured one, so
        # the observation matrix is a unit vector and the usual matrix products
        # reduce to indexing.
        innovation = measured[i] - state[0]
        innovation_covariance = covariance[0, 0] + measurement_variance

        gain = covariance[:, 0] / innovation_covariance

        state = state + gain * innovation
        covariance = covariance - np.outer(gain, covariance[0, :])

        estimates[i] = state

    return estimates
