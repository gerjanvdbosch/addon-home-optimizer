import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.forecasting import Forecasting
from app.identification import Identification
from app.optimization import Optimization
from app.state import StateManager
from domain.types import Settings
from features.baseload import BaseloadForecaster
from features.boiler import BoilerThermalIdentifier
from features.cop import HeatPumpCOPIdentifier
from features.dataset import (
    AttributeSeriesLoader,
    AttributeTimeSeriesLoader,
    DatasetLoader,
    TimeSeriesLoader,
)
from features.solar import SolarBiasIdentifier
from features.tap import TapForecaster
from infrastructure.home_assistant import HomeAssistant
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.repositories import (
    BacktestRepository,
    ConfigRepository,
    StateRepository,
)
from infrastructure.storage import JsonStorage


@dataclass(slots=True)
class Container:
    config_repository: ConfigRepository
    state_manager: StateManager
    forecasting: Forecasting
    identification: Identification
    optimization: Optimization
    backtest_repository: BacktestRepository


def create_container() -> Container:
    settings = load_settings()

    configure_logger(settings.log_level)

    influx = InfluxDatabase(settings)
    resolver = InfluxSensorResolver(influx)

    dataset_loader = DatasetLoader(
        loaders=[
            TimeSeriesLoader(influx, resolver),
            AttributeSeriesLoader(influx, resolver),
            AttributeTimeSeriesLoader(influx, resolver),
        ],
    )

    config_repository = ConfigRepository(
        JsonStorage(settings.data_path / "config.json"),
    )

    state_repository = StateRepository(
        JsonStorage(
            settings.data_path / "state.json",
        )
    )

    backtest_repository = BacktestRepository(
        JsonStorage(settings.data_path / "backtest.json"),
    )

    models_path = settings.data_path / "models"

    state_manager = StateManager(
        loader=dataset_loader,
        state_repository=state_repository,
        config_repository=config_repository,
        models_path=models_path,
        latitude=settings.latitude,
        longitude=settings.longitude,
    )

    forecasting = Forecasting(
        loader=dataset_loader,
        backtest_repository=backtest_repository,
        config_repository=config_repository,
        state_manager=state_manager,
        path=models_path,
        study_storage=f"sqlite:///{models_path / 'optuna.db'}",
        forecasters=[
            BaseloadForecaster(),
            TapForecaster(models_path=models_path),
        ],
    )

    identification = Identification(
        loader=dataset_loader,
        backtest_repository=backtest_repository,
        config_repository=config_repository,
        state_manager=state_manager,
        path=models_path,
        identifiers=[
            BoilerThermalIdentifier(),
            HeatPumpCOPIdentifier(
                mode=BoilerThermalIdentifier.DHW_ACTIVE_STATE, key="dhw"
            ),
            SolarBiasIdentifier(
                latitude=settings.latitude, longitude=settings.longitude
            ),
        ],
    )

    optimization = Optimization(
        state_manager=state_manager,
        config_repository=config_repository,
        models_path=models_path,
        # Only the add-on has a Supervisor token; without one (a local run) the
        # client logs its writes instead of sending them.
        home_assistant=HomeAssistant(os.environ.get("SUPERVISOR_TOKEN")),
    )

    container = Container(
        config_repository=config_repository,
        state_manager=state_manager,
        forecasting=forecasting,
        identification=identification,
        optimization=optimization,
        backtest_repository=backtest_repository,
    )

    return container


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
