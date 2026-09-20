import os
from dataclasses import dataclass

from app.forecasting import Forecasting
from app.identification import Identification
from app.optimization import Optimization
from app.settings import configure_logger, create_repositories, load_settings
from app.state import StateManager
from features.baseload import BaseloadForecaster
from features.boiler import BoilerThermalIdentifier
from features.building import (
    BuildingLumpedIdentifier,
    BuildingThermalIdentifier,
)
from features.cop import HeatPumpCOPIdentifier
from features.dataset import DatasetLoader
from features.solar import SolarBiasIdentifier
from features.tap import TapForecaster
from infrastructure.home_assistant import HomeAssistant
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.loaders import (
    AttributeSeriesLoader,
    AttributeTimeSeriesLoader,
    TimeSeriesLoader,
)
from infrastructure.repositories import BacktestRepository, ConfigRepository


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

    repositories = create_repositories(settings)
    config_repository = repositories.config
    state_repository = repositories.state
    backtest_repository = repositories.backtest

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
            # Space heating runs the compressor at a far lower supply
            # temperature than DHW, so it needs its own fit - one instance per
            # mode is what HeatPumpCOPIdentifier is built for. Registered
            # before the season starts so it calibrates from the first runs
            # rather than from whatever is left when someone remembers.
            # Cooling is deliberately absent: there the water is the COLD side
            # and this model does not describe it (see the class docstring).
            HeatPumpCOPIdentifier(mode="Verwarmen", key="heating"),
            BuildingThermalIdentifier(
                latitude=settings.latitude, longitude=settings.longitude
            ),
            # Both building structures are calibrated against the same data on
            # purpose: their own skill_vs_persistence and implausible_aperture
            # metrics then decide which one the data supports, instead of the
            # choice being an untested assumption.
            BuildingLumpedIdentifier(
                latitude=settings.latitude, longitude=settings.longitude
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
