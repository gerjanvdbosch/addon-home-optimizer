import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.types import CalibrateConfig, ValidateConfig
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

            try:
                identifier, df = self._prepare(identifier, config.days)
                identifier.calibrate(df)
            except ValueError as error:
                if "feature names should match" not in str(error):
                    raise

                logger.warning(
                    "Saved %s model does not match the current features, deleting "
                    "it and calibrating a new one: %s",
                    identifier.name,
                    error,
                )
                (self.path / f"{identifier.name}.joblib").unlink(missing_ok=True)
                identifier.model = None

                identifier, df = self._prepare(identifier, config.days)
                identifier.calibrate(df)

            identifier.save(self.path)

    def validate(self, config: ValidateConfig) -> None:
        identifier, df = self._prepare(config.target, config.days, end=config.end)

        result = identifier.validate(df)

        logger.info(
            "Validate finished (%s): %s",
            identifier.name,
            " ".join(f"{key}={value:.4g}" for key, value in result.items()),
        )

    def _prepare(
        self,
        identifier: str | SystemIdentifier,
        days: int,
        end: datetime | None = None,
    ) -> tuple[SystemIdentifier, Any]:
        if isinstance(identifier, str):
            identifier = self._get_identifier(identifier)

        identifier.load(self.path)

        config = self.config_repository.load()

        end = end or datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        dataset = identifier.dataset(config)
        df = self.loader.load(dataset, start, end)

        return identifier, df

    def _get_identifier(self, name: str) -> SystemIdentifier:
        for identifier in self.identifiers:
            if identifier.name == name:
                return identifier

        raise ValueError(f"Unknown system identifier: {name}")
