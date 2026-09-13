import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from domain.types import BacktestConfig, CalibrateConfig
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

            identifier, df = self._prepare(identifier, config.days)

            identifier.calibrate(df)

            identifier.save(self.path)

    def validate(self, config: BacktestConfig) -> None:
        identifier, df = self._prepare(config.target, config.days)

        identifier.validate(df)

    def _prepare(
        self,
        identifier: str | SystemIdentifier,
        days: int,
    ) -> tuple[SystemIdentifier, Any]:
        if isinstance(identifier, str):
            identifier = self._get_identifier(identifier)

        identifier.load(self.path)

        config = self.config_repository.load()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        dataset = identifier.dataset(config)
        df = self.loader.load(dataset, start, end)

        return identifier, df

    def _get_identifier(self, name: str) -> SystemIdentifier:
        for identifier in self.identifiers:
            if identifier.name == name:
                return identifier

        raise ValueError(f"Unknown system identifier: {name}")
