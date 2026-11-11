"""Numerical integration and state estimation for a linear state space.

Separated from the balances in domain/physics.py on purpose: nothing here
knows what the states mean. Both take the (A, B) of any continuous model and
turn it into discrete steps - which is what keeps the discrete implementation
identical in meaning to the continuous formulation rather than a second,
slightly different model.
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import expm, solve_discrete_are


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
        intervals, interval = np.unique(
            np.round(np.asarray(dt_seconds, dtype=float), 3), return_inverse=True
        )
        # Each whole value and interval once, then gathered per step - as one
        # integer per pair, which np.unique sorts far faster than rows.
        keys = np.concatenate([low, low + 1]) * len(intervals) + np.tile(interval, 2)
        unique, index = np.unique(keys, return_inverse=True)
        wholes = [
            self._whole(
                int(key // len(intervals)), float(intervals[key % len(intervals)])
            )
            for key in unique
        ]
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


def discrete_steps(
    a: np.ndarray,
    b: np.ndarray,
    dt_seconds: np.ndarray,
    schedule: ScheduledSystem | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The discrete (A, B) of the step into each sample after the first,
    stacked - the schedule's, or (a, b)'s once per interval length."""

    dt_seconds = np.asarray(dt_seconds, float)[1:]

    # A single sample takes no step.
    if not len(dt_seconds):
        return np.empty((0, *a.shape)), np.empty((0, *b.shape))

    if schedule is not None:
        return schedule.steps(dt_seconds)

    _, first, index = np.unique(
        np.round(dt_seconds, 3), return_index=True, return_inverse=True
    )
    steps = [discretize_zoh(a, b, dt) for dt in dt_seconds[first]]

    return (
        np.stack([a_d for a_d, _ in steps])[index],
        np.stack([b_d for _, b_d in steps])[index],
    )


def kalman_states(
    a: np.ndarray,
    b: np.ndarray,
    measured: np.ndarray,
    inputs: np.ndarray,
    dt_seconds: np.ndarray,
    process_noise_w: float,
    measurement_variance: float | Sequence[float],
    observation: np.ndarray | None = None,
    disturbance_inputs: tuple[int, ...] = (1,),
    schedule: ScheduledSystem | None = None,
    reading_drift_k: tuple[float, ...] = (),
    reading_recovery_seconds: float | None = None,
    observation_per_step: bool = False,
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

    `measurement_variance` is the readings' noise (K2), one for all or one
    per reading. With `observation_per_step` the observation has a matrix
    per step (steps, readings, states): a reading that sees a different
    mixture of states each step.

    With a `schedule` the system is its, step by step, in place of (a, b).

    `reading_drift_k` is, per thermometer, the spread (K) of a departure of
    its reading from the state it reads that moves no heat - a draught over
    the sensor - and fades over `reading_recovery_seconds`; 0 for none.
    """

    n = a.shape[0]
    # Default: the first state alone, the one a thermometer normally reads.
    h = np.atleast_2d(np.eye(1, n)[0] if observation is None else observation)
    a_steps, b_steps = discrete_steps(a, b, dt_seconds, schedule)

    return kalman_filter(
        a_steps,
        b_steps,
        measured,
        inputs,
        process_noise_w,
        measurement_variance,
        h,
        disturbance_inputs,
        reading_drift_k,
        (
            np.exp(-np.asarray(dt_seconds, float)[1:] / reading_recovery_seconds)
            if any(reading_drift_k)
            else None
        ),
        observation_per_step=observation_per_step,
    )


def _settled_covariance(
    a: np.ndarray, q: np.ndarray, h: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """The predicted covariance a filter on step (a, q), read by h with
    `noise`, settles to - per system along any leading axes."""

    h = np.broadcast_to(h, a.shape[:-2] + h.shape[-2:])
    systems = zip(
        a.reshape(-1, *a.shape[-2:]),
        q.reshape(-1, *q.shape[-2:]),
        h.reshape(-1, *h.shape[-2:]),
        strict=True,
    )

    # A covariance is symmetric; its mean over the steps is so only to
    # rounding (8e-15 seen), which the solver refuses.
    return np.stack(
        [
            solve_discrete_are(a_s.T, h_s.T, (q_s + q_s.T) / 2.0, noise)
            for a_s, q_s, h_s in systems
        ]
    ).reshape(a.shape)


def kalman_filter(
    a_steps: np.ndarray,
    b_steps: np.ndarray,
    measured: np.ndarray,
    inputs: np.ndarray,
    process_noise_w: float,
    measurement_variance: float | Sequence[float],
    observation: np.ndarray,
    disturbance_inputs: tuple[int, ...],
    reading_drift_k: tuple[float, ...] = (),
    reading_decay: np.ndarray | None = None,
    same: Sequence[int] | None = None,
    stationary: bool = False,
    observation_per_step: bool = False,
) -> np.ndarray:
    """kalman_states over steps discrete_steps() gives - or over many systems
    at once against the same readings, each along a leading axis of the steps,
    the inputs and the observation (a fit's parameter sets, say).

    `same` gives, for each of those, the first with the very same steps and
    observation: the covariance and the gain depend on nothing else, so they
    are carried for the first alone - a Jacobian's sets mostly move only the
    inputs (an aperture, a horizon), and the covariance's products were most
    of a fit's time.

    `stationary` holds each system's covariance where its mean step settles it
    (the discrete Riccati equation) instead of carrying it step by step: only
    the states are left to the loop. The building's gain moves hardly at all
    once settled - within 0.6% of its median over 150 days of steps with every
    step's system its own (real data, 5 Nov 2026), settled within 50 steps -
    and carrying it was most of a fit's time.

    A reading's drift (see kalman_states) is a state of the filter's own, one
    per thermometer, decaying by `reading_decay` each step (exp(-dt/tau)) and
    renewed so that its spread stays reading_drift_k: an AR(1) departure added
    to what the thermometer reads. It is no part of the system, so it is left
    out of the estimate returned.

    `observation_per_step` (see kalman_states) is for one system, its
    covariance carried step by step.
    """

    if observation_per_step and (stationary or same is not None):
        raise ValueError("A per-step observation is for one system, step by step.")

    h = observation
    n = a_steps.shape[-1]
    readings = np.asarray(measured, float).reshape(len(measured), -1)
    drift_k = np.asarray(reading_drift_k, float)
    drifting = bool(drift_k.any())

    # Unmeasured parts start where the readings put them on average, and each
    # reading's departure from that is spread over the states it sees. States
    # are columns, so one product serves one system or many.
    first = readings[0]
    level = float(np.nanmean(first))
    first_h = h[0] if observation_per_step else h
    state = level + np.linalg.pinv(first_h) @ np.nan_to_num(first - level)[:, None]
    variances = np.broadcast_to(np.asarray(measurement_variance, float), (h.shape[-2],))
    covariance = np.broadcast_to(np.eye(n) * variances.min(), state.shape[:-2] + (n, n))

    # The drift, after the system's states: driven by nothing, read in its
    # thermometer's reading alone, starting at none at its own spread.
    if drifting:
        size = n + len(drift_k)
        own = n + np.arange(len(drift_k))
        augmented = np.zeros(a_steps.shape[:-2] + (size, size))
        augmented[..., :n, :n] = a_steps
        augmented[..., own, own] = reading_decay[:, None]
        a_steps = augmented
        augmented = np.zeros(b_steps.shape[:-2] + (size, b_steps.shape[-1]))
        augmented[..., :n, :] = b_steps
        b_steps = augmented
        state = np.concatenate(
            [state, np.zeros(state.shape[:-2] + (len(drift_k), 1))], axis=-2
        )
        grown = np.zeros(covariance.shape[:-2] + (size, size))
        grown[..., :n, :n] = covariance
        grown[..., own, own] = drift_k**2
        covariance = grown
        h = np.concatenate(
            [h, np.broadcast_to(np.eye(len(drift_k)), h.shape[:-1] + (len(drift_k),))],
            axis=-1,
        )

    # The disturbances enter where unmodelled heat flows would, and their
    # effect on the state is what the DISCRETE input matrix says inputs of
    # that size do over one step. Building it from the continuous B instead
    # understates it by a factor dt^2 - here about a million - which silently
    # turns the filter into a free-running simulation that ignores the
    # measurement. Independent channels, so their covariances add.
    disturbances = b_steps[..., list(disturbance_inputs)]
    # Everything that does not depend on the previous step, for all steps at
    # once, step first: only the recursion itself is left to the loop.
    process_covariances = np.moveaxis(
        (process_noise_w**2) * disturbances @ np.swapaxes(disturbances, -1, -2), -3, 0
    )

    if drifting:
        renewal = drift_k**2 * (1.0 - reading_decay[:, None] ** 2)
        process_covariances[..., own, own] += renewal.reshape(
            (len(renewal),) + (1,) * (process_covariances.ndim - 3) + (len(drift_k),)
        )
    drives = np.moveaxis(b_steps @ inputs[..., : len(readings) - 1, :, None], -3, 0)
    a_steps = np.moveaxis(a_steps, -3, 0)
    a_transposed = np.swapaxes(a_steps, -1, -2)

    estimates = np.empty((len(readings), *state.shape))
    estimates[0] = state

    # The covariance and the gain over the distinct systems alone, the gain
    # handed back to every set by `spread`.
    if same is not None:
        same = np.asarray(same)
        distinct = np.flatnonzero(same == np.arange(len(same)))
        spread = np.searchsorted(distinct, same)
    else:
        distinct = spread = slice(None)
    h_distinct = h[distinct]
    h_distinct_t = np.swapaxes(h_distinct, -1, -2)
    noise = np.diag(variances)

    if stationary:
        # The covariance a step's prediction settles to, and its gain.
        covariance = _settled_covariance(
            a_steps.mean(axis=0)[distinct],
            process_covariances.mean(axis=0)[distinct],
            h_distinct,
            noise,
        )
        covariance_h = covariance @ h_distinct_t
        settled_gain = covariance_h @ np.linalg.inv(h_distinct @ covariance_h + noise)
    else:
        covariance = covariance[distinct]
        a_distinct = a_steps[:, distinct]
        a_transposed = a_transposed[:, distinct]
        process_covariances = process_covariances[:, distinct]

    seen_rows = ~np.isnan(readings)
    all_seen = seen_rows.all(axis=1).tolist()
    any_seen = seen_rows.any(axis=1).tolist()

    for i in range(1, len(readings)):
        state = a_steps[i - 1] @ state + drives[i - 1]

        if not stationary:
            covariance = (
                a_distinct[i - 1] @ covariance @ a_transposed[i - 1]
                + process_covariances[i - 1]
            )

        if any_seen[i]:
            h_now, h_now_t = (
                (h[i], np.swapaxes(h[i], -1, -2))
                if observation_per_step
                else (h_distinct, h_distinct_t)
            )

            if all_seen[i]:
                h_seen = h[i] if observation_per_step else h
                noise_seen = noise
                own_h, own_h_t = h_now, h_now_t
                reading = readings[i][:, None]
            else:
                seen = seen_rows[i]
                h_seen = (h[i] if observation_per_step else h)[..., seen, :]
                own_h, own_h_t = h_now[..., seen, :], h_now_t[..., seen]
                noise_seen = noise[np.ix_(seen, seen)]
                reading = readings[i][seen][:, None]

            innovation = reading - h_seen @ state

            if stationary and all_seen[i]:
                gain = settled_gain
            else:
                covariance_h = covariance @ own_h_t
                gain = covariance_h @ np.linalg.inv(own_h @ covariance_h + noise_seen)

            state = state + gain[spread] @ innovation

            # G (H P), not (G H) P: the same product through the few readings
            # rather than a full state-by-state one, ~10% of a fit's filter.
            if not stationary:
                covariance = covariance - gain @ (own_h @ covariance)

        estimates[i] = state

    # Back to the systems first, a row per reading.
    return np.moveaxis(estimates[..., :n, 0], 0, -2)
