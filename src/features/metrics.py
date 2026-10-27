import numpy as np
from numpy.typing import ArrayLike


def mean_absolute_error(measured: ArrayLike, predicted: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(measured) - np.asarray(predicted))))


def root_mean_squared_error(measured: ArrayLike, predicted: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(measured) - np.asarray(predicted)) ** 2)))


def r2_score(measured: ArrayLike, predicted: ArrayLike) -> float:
    """The share of the measured variance the prediction explains; NaN where
    the measurement does not vary, which leaves nothing to explain."""

    measured = np.asarray(measured, dtype=float)
    total = float(np.sum((measured - measured.mean()) ** 2))

    if total == 0.0:
        return float("nan")

    return 1.0 - float(np.sum((measured - np.asarray(predicted)) ** 2)) / total
