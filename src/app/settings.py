"""Settings and file repositories: everything the web process needs.

Kept apart from app.bootstrap, which imports the modelling libraries, so the
web process stays small; those libraries only live in a job's own process (see
app.worker).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from domain.types import Settings
from infrastructure.home_assistant import HomeAssistant
from infrastructure.repositories import (
    BacktestRepository,
    ConfigRepository,
    StateRepository,
)
from infrastructure.storage import JsonStorage


@dataclass(slots=True)
class Repositories:
    config: ConfigRepository
    state: StateRepository
    backtest: BacktestRepository


def create_repositories(settings: Settings) -> Repositories:
    return Repositories(
        config=ConfigRepository(JsonStorage(settings.data_path / "config.json")),
        state=StateRepository(JsonStorage(settings.data_path / "state.json")),
        backtest=BacktestRepository(JsonStorage(settings.data_path / "backtest.json")),
    )


def load_settings() -> Settings:
    options = Path("/data/options.json")

    if options.exists():
        latitude, longitude = HomeAssistant(os.environ["SUPERVISOR_TOKEN"]).location()

        return Settings(
            **json.loads(options.read_text()),
            data_path=Path("/data"),
            latitude=latitude,
            longitude=longitude,
        )

    load_dotenv()

    return Settings(
        influx_host=os.getenv("INFLUX_HOST", "homeassistant.local"),
        influx_port=int(os.getenv("INFLUX_PORT", 8086)),
        influx_username=os.getenv("INFLUX_USERNAME", ""),
        influx_password=os.getenv("INFLUX_PASSWORD", ""),
        influx_database=os.getenv("INFLUX_DATABASE", "home_assistant"),
        log_level=os.getenv("LOG_LEVEL", "DEBUG"),
        latitude=_required_float_env("LATITUDE"),
        longitude=_required_float_env("LONGITUDE"),
    )


def _required_float_env(name: str) -> float:
    value = os.getenv(name)

    if value is None:
        raise ValueError(
            f"{name} must be set (e.g. in .env): the installation's location "
            "is required for the solar position."
        )

    return float(value)


def configure_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.INFO)
