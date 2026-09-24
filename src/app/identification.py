import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.jobs import CalibrateConfig, ValidateConfig
from features.dataset import DatasetLoader
from features.identifier import SystemIdentifier
from infrastructure.repositories import BacktestRepository, ConfigRepository

logger = logging.getLogger(__name__)


class Identification:
    def __init__(
        self,
        loader: DatasetLoader,
        backtest_repository: BacktestRepository,
        config_repository: ConfigRepository,
        state_manager: Any,
        path: Path,
        identifiers: list[SystemIdentifier],
    ):
        self.loader = loader
        self.backtest_repository = backtest_repository
        self.config_repository = config_repository
        self.state_manager = state_manager
        self.path = path
        self.identifiers = identifiers

    def calibrate(self, config: CalibrateConfig) -> None:
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
                    identifier, df = self._prepare(identifier, config.days)
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
        for identifier in self.identifiers:
            if config.target and identifier.name != config.target:
                continue

            try:
                identifier, df = self._prepare(identifier, config.days, end=config.end)
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
        end: datetime | None = None,
    ) -> tuple[SystemIdentifier, Any]:
        identifier.load(self.path)

        config = self.config_repository.load()

        end = end or datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        dataset = identifier.dataset(config)
        df = self.loader.load(dataset, start, end)

        return identifier, df
