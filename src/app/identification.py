import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from domain.jobs import CalibrateConfig, ValidateConfig
from features.dataset import DatasetLoader
from features.identifier import SystemIdentifier
from infrastructure.repositories import ConfigRepository

logger = logging.getLogger(__name__)


class Identification:
    def __init__(
        self,
        loader: DatasetLoader,
        config_repository: ConfigRepository,
        state_manager: Any,
        path: Path,
        identifiers: list[SystemIdentifier],
    ):
        self.loader = loader
        self.config_repository = config_repository
        self.state_manager = state_manager
        self.path = path
        self.identifiers = identifiers
        # The last frame loaded, and what for: models that ask for the same
        # data over the same window - the COP of each mode, the building and
        # its floor circuits - share one load. Only the last, so memory holds
        # no more than one frame at a time, as before.
        self._loaded: tuple[Any, pd.DataFrame] | None = None

    def calibrate(self, config: CalibrateConfig) -> None:
        # One window for the whole job, so models asking for the same data
        # ask for the same rows.
        end = datetime.now(timezone.utc)

        for identifier in self.identifiers:
            if config.target and identifier.name != config.target:
                continue

            saved = self.path / f"{identifier.name}.joblib"

            # Two attempts, and only where there is something to retry with: a
            # previously saved model is the one thing here that can be stale, so
            # a first failure drops it and refits from scratch. Deciding that by
            # matching the message text was worse than useless - it recognised
            # one library's wording and would have gone quietly unhandled the
            # day that wording changed.
            for attempt in range(2):
                try:
                    identifier, df = self._prepare(identifier, config.days, end)
                    identifier.calibrate(df)
                    identifier.save(self.path)
                    break
                except Exception as error:
                    if attempt == 0 and saved.exists():
                        logger.warning(
                            "Calibrating %s failed with a saved model in "
                            "place, dropping it and fitting from scratch: %s",
                            identifier.name,
                            error,
                        )
                        saved.unlink(missing_ok=True)
                        identifier.model = None
                        continue

                    # Asking for one model by name is a direct instruction: it
                    # fails loudly, with the traceback that says why.
                    # Calibrating everything is a batch, and one model that
                    # cannot be fitted is normal there - a mode the heat pump
                    # has not run yet, a sensor not configured. Letting that
                    # abort the loop has twice taken down models that were
                    # perfectly fittable, so here it is reported and the rest
                    # still run.
                    if config.target:
                        raise

                    logger.error(
                        "Calibrating %s failed, continuing with the rest: %s",
                        identifier.name,
                        error,
                    )
                    break

    def validate(self, config: ValidateConfig) -> None:
        # Without a target every model, as calibrate() does - and as a batch
        # too: one model that cannot be validated (never calibrated, no runs in
        # the window) is reported and the rest still run.
        end = config.end or datetime.now(timezone.utc)

        for identifier in self.identifiers:
            if config.target and identifier.name != config.target:
                continue

            try:
                identifier, df = self._prepare(identifier, config.days, end)
                result = identifier.validate(df)
            except Exception as error:
                if config.target:
                    raise

                logger.error(
                    "Validating %s failed, continuing with the rest: %s",
                    identifier.name,
                    error,
                )
                continue

            logger.info(
                "Validate finished (%s): %s",
                identifier.name,
                " ".join(f"{key}={value:.4g}" for key, value in result.items()),
            )

    def _prepare(
        self,
        identifier: SystemIdentifier,
        days: int,
        end: datetime,
    ) -> tuple[SystemIdentifier, pd.DataFrame]:
        identifier.load(self.path)

        config = self.config_repository.load()

        start = end - timedelta(days=days)
        key = (identifier.dataset(config), start, end)

        if self._loaded is None or self._loaded[0] != key:
            # The old frame goes before the new one comes.
            self._loaded = None
            self._loaded = (key, self.loader.load(*key))

        # Shallow: under copy-on-write what one model adds or changes in its
        # frame never reaches the next one's.
        return identifier, self._loaded[1].copy(deep=False)
