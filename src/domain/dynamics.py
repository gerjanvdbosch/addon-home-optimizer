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
    observation: np.ndarray | None = None,
    disturbance_inputs: tuple[int, ...] = (1,),
) -> np.ndarray:
    """Filtered state estimate at every sample, from one scalar measurement.

    That measurement covers at most a part of each state - the room
    thermometer reads the air, and something of the surfaces around it - so
    the rest has to be inferred from how the reading moves relative to what
    the model predicted. That is what a Kalman filter does, and it is the
    honest way to start a rollout: hard-resetting the measured state while
    letting an unmeasured one free-run leaves the two inconsistent with each
    other, which flatters a single-state model and penalises a multi-state one.

    It is also not only an evaluation device. An MPC must know where it starts
    from at solve time, including a screed temperature nothing measures, so
    this is a missing part of the system rather than a test harness.

    Process noise is expressed as an unmodelled HEAT FLOW (W) entering through
    the input channels named by `disturbance_inputs`, not as an abstract
    covariance: the disturbances this stands in for - ventilation through an
    opened window, a wood stove, a visitor, heat that never reached the node
    it was measured into - are heat flows, so their magnitude can be reasoned
    about physically. One channel per node that can be disturbed
    independently; naming them is the caller's job, since nothing here knows
    what an input means.

    `observation` is the row vector the thermometer sees, defaulting to the
    first state alone. A sensor reading a mixture of states (an operative
    temperature, say) makes the others partly observable, which is the only
    way the filter can correct a state nothing measures directly. Several
    thermometers are a matrix of such rows, with `measured` holding a column
    each; a reading that is missing (NaN) simply corrects nothing that step.
    """

    n = a.shape[0]
    # Default: the first state alone, the one a thermometer normally reads.
    h = np.atleast_2d(np.eye(1, n)[0] if observation is None else observation)
    readings = np.asarray(measured, float).reshape(len(measured), -1)

    # Unmeasured parts start where the readings put them on average, and each
    # reading's departure from that is spread over the states it sees.
    first = readings[0]
    level = float(np.nanmean(first))
    state = np.full(n, level) + np.linalg.pinv(h) @ np.nan_to_num(first - level)
    covariance = np.eye(n) * measurement_variance

    estimates = np.empty((len(readings), n))
    estimates[0] = state

    cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for i in range(1, len(readings)):
        dt = float(dt_seconds[i])
        key = round(dt, 3)

        if key not in cache:
            a_d, b_d = discretize_zoh(a, b, dt)
            # The disturbances enter where unmodelled heat flows would, and
            # their effect on the state is what the DISCRETE input matrix says
            # inputs of that size do over one step. Building it from the
            # continuous B instead understates it by a factor dt^2 - here about
            # a million - which silently turns the filter into a free-running
            # simulation that ignores the measurement. Independent channels, so
            # their covariances add.
            disturbance = b_d[:, list(disturbance_inputs)]
            cache[key] = (
                a_d,
                b_d,
                (process_noise_w**2) * (disturbance @ disturbance.T),
            )

        a_d, b_d, process_covariance = cache[key]

        state = a_d @ state + b_d @ inputs[i - 1]
        covariance = a_d @ covariance @ a_d.T + process_covariance

        seen = ~np.isnan(readings[i])

        if seen.any():
            h_seen = h[seen]
            innovation = readings[i][seen] - h_seen @ state
            innovation_covariance = (
                h_seen @ covariance @ h_seen.T
                + np.eye(int(seen.sum())) * measurement_variance
            )
            gain = covariance @ h_seen.T @ np.linalg.inv(innovation_covariance)

            state = state + gain @ innovation
            covariance = covariance - gain @ h_seen @ covariance

        estimates[i] = state

    return estimates
