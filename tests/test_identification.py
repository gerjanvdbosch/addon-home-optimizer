import logging
from pathlib import Path

import pandas as pd
import pytest

from app.identification import Identification
from domain.jobs import CalibrateConfig, ValidateConfig


class _Identifier:
    """Minimal stand-in: only what Identification actually calls."""

    def __init__(self, name: str, error: Exception | None = None):
        self._name = name
        self._error = error
        self.calibrated = False
        self.saved = False
        self.validated = False
        self.attempts = 0

    @property
    def name(self) -> str:
        return self._name

    def load(self, path: Path) -> None: ...

    def dataset(self, config):
        return object()

    def calibrate(self, df: pd.DataFrame) -> None:
        self.attempts += 1

        if self._error is not None:
            raise self._error

        self.calibrated = True

    def save(self, path: Path) -> None:
        self.saved = True

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        if self._error is not None:
            raise self._error

        self.validated = True

        return {"mae": 0.1}


class _Loader:
    def load(self, dataset, start, end) -> pd.DataFrame:
        return pd.DataFrame({"time": []})


class _ConfigRepository:
    def load(self):
        return object()


def _identification(identifiers, tmp_path) -> Identification:
    return Identification(
        loader=_Loader(),
        backtest_repository=None,
        config_repository=_ConfigRepository(),
        state_manager=None,
        path=tmp_path,
        identifiers=identifiers,
    )


def test_one_unfittable_model_does_not_take_down_the_batch(tmp_path, caplog):
    """A mode the heat pump has not run yet is normal in a batch run.

    It has twice aborted the whole loop here - once on a cooling COP that could
    not be fitted, once on a missing configuration field - leaving models that
    were perfectly fittable uncalibrated and silently stale.
    """

    first = _Identifier("boiler")
    broken = _Identifier("cop_heating", ValueError("no measurements remain"))
    last = _Identifier("solar")

    with caplog.at_level(logging.ERROR):
        _identification([first, broken, last], tmp_path).calibrate(
            CalibrateConfig(days=60)
        )

    assert first.calibrated and first.saved
    assert last.calibrated and last.saved, "the failure stopped the models after it"
    assert not broken.saved, "a model that could not be fitted must not be saved"
    assert "cop_heating" in caplog.text


def test_asking_for_one_model_by_name_still_fails_loudly(tmp_path):
    """A direct instruction is not a batch: it reports why it could not run."""

    broken = _Identifier("cop_heating", ValueError("no measurements remain"))

    with pytest.raises(ValueError, match="no measurements remain"):
        _identification([_Identifier("boiler"), broken], tmp_path).calibrate(
            CalibrateConfig(target="cop_heating", days=60)
        )


def test_a_model_with_nothing_saved_is_not_attempted_twice(tmp_path, caplog):
    """The retry exists to drop a stale saved model, so without one it is waste.

    A mode the heat pump has not run yet fails every batch run. Loading its
    window of history a second time to fail identically costs real time and
    tells no one anything.
    """

    broken = _Identifier("cop_heating", ValueError("no measurements remain"))

    with caplog.at_level(logging.ERROR):
        _identification([broken], tmp_path).calibrate(CalibrateConfig(days=60))

    assert broken.attempts == 1
    assert "dropping it" not in caplog.text


def test_a_stale_saved_model_is_dropped_and_refitted(tmp_path, caplog):
    """With something saved, one failure is worth a second attempt from scratch."""

    calls = {"n": 0}

    class _StaleOnce(_Identifier):
        def calibrate(self, df: pd.DataFrame) -> None:
            self.attempts += 1
            calls["n"] += 1
            # Fails only while the saved model is still there.
            if (tmp_path / "boiler.joblib").exists():
                raise ValueError("feature mismatch")
            self.calibrated = True

    (tmp_path / "boiler.joblib").write_bytes(b"stale")
    identifier = _StaleOnce("boiler")

    with caplog.at_level(logging.WARNING):
        _identification([identifier], tmp_path).calibrate(CalibrateConfig(days=60))

    assert identifier.attempts == 2
    assert identifier.calibrated and identifier.saved
    assert "dropping it and fitting from scratch" in caplog.text


def test_validate_without_a_target_validates_every_model(tmp_path, caplog):
    """Without a target it once crashed on None instead of running a batch,
    as calibrate does: every model, one failure reported and the rest run."""

    first = _Identifier("boiler")
    broken = _Identifier("space_heating", ValueError("no run held out"))
    last = _Identifier("solar")

    with caplog.at_level(logging.ERROR):
        _identification([first, broken, last], tmp_path).validate(
            ValidateConfig(days=60)
        )

    assert first.validated and last.validated
    assert "space_heating" in caplog.text


def test_validate_by_name_fails_loudly(tmp_path):
    broken = _Identifier("space_heating", ValueError("no run held out"))

    with pytest.raises(ValueError, match="no run held out"):
        _identification([_Identifier("boiler"), broken], tmp_path).validate(
            ValidateConfig(target="space_heating", days=60)
        )
