"""Numerical integration and state estimation for a linear state space.

Separated from the balances in domain/physics.py on purpose: nothing here
knows what the states mean. Both take the (A, B) of any continuous model and
turn it into discrete steps - which is what keeps the discrete implementation
identical in meaning to the continuous formulation rather than a second,
slightly different model.
"""

from collections.abc import Callable

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


class ScheduledSystem:
    """A linear system whose matrices follow one value per step - a
    conductance that changes with the weather, say: `system_at(value)` gives
    the continuous (A, B) at a value, `values` the value over the interval
    starting at each step.

    Each step is discretized exactly (see discretize_zoh) at the whole values
    around its own and linearly between them: whole values 1 apart (1 W/K for
    a conductance), where the discrete matrices hardly bend, and smooth in
    the value, as a fit's gradients need - one exact discretization per whole
    value rather than one per step.
    """

    def __init__(
        self,
        system_at: Callable[[float], tuple[np.ndarray, np.ndarray]],
        values: np.ndarray,
        cache: dict | None = None,
    ) -> None:
        self.system_at = system_at
        self.values = np.asarray(values, dtype=float)
        self._cache = {} if cache is None else cache

    def window(self, start: int, end: int) -> "ScheduledSystem":
        """The same system over steps start to end, sharing its discretizations."""

        return ScheduledSystem(self.system_at, self.values[start:end], self._cache)

    def discrete(self, step: int, dt_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """The discrete (A, B) over one step."""

        a_d, b_d = self.steps(np.array([dt_seconds]), start=step)

        return a_d[0], b_d[0]

    def steps(
        self, dt_seconds: np.ndarray, start: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """The discrete (A, B) of steps start, start + 1, ... over the given
        intervals, stacked - one blend for all of them."""

        values = self.values[start : start + len(dt_seconds)]
        low = np.floor(values).astype(int)
        weight = (values - low)[:, None, None]
        dt = np.round(np.asarray(dt_seconds, dtype=float), 3)
        # Each whole value and interval once, then gathered per step.
        keys = np.column_stack([np.concatenate([low, low + 1]), np.tile(dt, 2)])
        unique, index = np.unique(keys, axis=0, return_inverse=True)
        wholes = [self._whole(int(value), float(step)) for value, step in unique]
        a_all = np.stack([a for a, _ in wholes])[index.ravel()]
        b_all = np.stack([b for _, b in wholes])[index.ravel()]
        n = len(values)
        a_low, a_high, b_low, b_high = a_all[:n], a_all[n:], b_all[:n], b_all[n:]

        return (
            a_low + weight * (a_high - a_low),
            b_low + weight * (b_high - b_low),
        )

    def _whole(self, value: int, dt_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        key = (value, round(dt_seconds, 3))

        if key not in self._cache:
            self._cache[key] = discretize_zoh(*self.system_at(float(value)), dt_seconds)

        return self._cache[key]


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
    schedule: ScheduledSystem | None = None,
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

    With a `schedule` the system is its, step by step, in place of (a, b).
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

    if schedule is not None:
        scheduled_a, scheduled_b = schedule.steps(np.asarray(dt_seconds, float)[1:])
        disturbances = scheduled_b[:, :, list(disturbance_inputs)]
        scheduled_covariance = (process_noise_w**2) * np.einsum(
            "kij,klj->kil", disturbances, disturbances
        )

    for i in range(1, len(readings)):
        dt = float(dt_seconds[i])
        key = round(dt, 3)

        if key not in cache and schedule is None:
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

        if schedule is None:
            a_d, b_d, process_covariance = cache[key]
        else:
            a_d, b_d = scheduled_a[i - 1], scheduled_b[i - 1]
            process_covariance = scheduled_covariance[i - 1]

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
