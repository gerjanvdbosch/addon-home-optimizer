import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from app.settings import configure_logger, create_repositories, load_settings
from app.state import StateManager
from domain.config import Settings
from features.dataset import DatasetLoader
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.loaders import (
    AttributeSeriesLoader,
    AttributeTimeSeriesLoader,
    TimeSeriesLoader,
)
from infrastructure.repositories import BacktestRepository, ConfigRepository

if TYPE_CHECKING:
    from app.forecasting import Forecasting
    from app.identification import Identification
    from app.optimization import Optimization


@dataclass
class Container:
    """What a job runs with. Every job starts a fresh process (see
    app.worker), so the services only some jobs use are built - and their
    libraries imported - when a job first asks for them: forecasting's
    skforecast, sklearn and optuna and the optimizer's pyomo took 1.9 of a
    state update's 5 seconds while it used none of them."""

    settings: Settings
    config_repository: ConfigRepository
    state_manager: StateManager
    backtest_repository: BacktestRepository
    dataset_loader: DatasetLoader
    models_path: Path

    @cached_property
    def forecasting(self) -> "Forecasting":
        from app.forecasting import Forecasting
        from features.baseload import BaseloadForecaster
        from features.tap import TapForecaster

        return Forecasting(
            loader=self.dataset_loader,
            backtest_repository=self.backtest_repository,
            config_repository=self.config_repository,
            state_manager=self.state_manager,
            path=self.models_path,
            study_storage=f"sqlite:///{self.models_path / 'optuna.db'}",
            forecasters=[
                BaseloadForecaster(),
                TapForecaster(models_path=self.models_path),
            ],
        )

    @cached_property
    def identification(self) -> "Identification":
        from app.identification import Identification
        from features.boiler import BoilerThermalIdentifier
        from features.building import (
            BuildingLumpedIdentifier,
            BuildingThermalIdentifier,
        )
        from features.cop import HeatPumpCOPIdentifier
        from features.solar import SolarBiasIdentifier

        latitude, longitude = self.settings.latitude, self.settings.longitude

        return Identification(
            loader=self.dataset_loader,
            backtest_repository=self.backtest_repository,
            config_repository=self.config_repository,
            state_manager=self.state_manager,
            path=self.models_path,
            identifiers=[
                BoilerThermalIdentifier(),
                HeatPumpCOPIdentifier(key="dhw"),
                # Space heating runs the compressor at a far lower supply
                # temperature than DHW, so it needs its own fit - one instance
                # per mode is what HeatPumpCOPIdentifier is built for.
                # Registered before the season starts so it calibrates from the
                # first runs rather than from whatever is left when someone
                # remembers. Cooling is deliberately absent: there the water is
                # the COLD side and this model does not describe it (see the
                # class docstring).
                HeatPumpCOPIdentifier(key="heating"),
                BuildingThermalIdentifier(latitude=latitude, longitude=longitude),
                # Both building structures are calibrated against the same data
                # on purpose: their own skill_vs_persistence and
                # implausible_aperture metrics then decide which one the data
                # supports, instead of the choice being an untested assumption.
                BuildingLumpedIdentifier(latitude=latitude, longitude=longitude),
                SolarBiasIdentifier(latitude=latitude, longitude=longitude),
            ],
        )

    @cached_property
    def optimization(self) -> "Optimization":
        from app.optimization import Optimization
        from infrastructure.home_assistant import HomeAssistant

        return Optimization(
            state_manager=self.state_manager,
            config_repository=self.config_repository,
            models_path=self.models_path,
            # Only the add-on has a Supervisor token; without one (a local run)
            # the client logs its writes instead of sending them.
            home_assistant=HomeAssistant(os.environ.get("SUPERVISOR_TOKEN")),
        )


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
    models_path = settings.data_path / "models"

    return Container(
        settings=settings,
        config_repository=repositories.config,
        state_manager=StateManager(
            loader=dataset_loader,
            state_repository=repositories.state,
            config_repository=repositories.config,
            models_path=models_path,
            latitude=settings.latitude,
            longitude=settings.longitude,
        ),
        backtest_repository=repositories.backtest,
        dataset_loader=dataset_loader,
        models_path=models_path,
    )
