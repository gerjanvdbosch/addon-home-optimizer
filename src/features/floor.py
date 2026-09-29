"""How this heat pump runs the floor circuit when it is left to itself.

The heat pump chooses its own supply temperature - from its heating curve and
modulation, running long at low power rather than cycling - so the heat a
space-heating run delivers is not a decision a plan can make. What a plan
decides is when the zone is heated; how much heat that brings follows from
the curve and from the floor it heats. This identifies both, and the run
length the heat pump keeps to, from its own heating runs.

Cooling is the same floor run the other way, identified the same way from
cooling runs: the heat is negative, the supply below the mass, and the supply
is the one the heat pump holds by itself (real data: 0.5 K below its zone
flow setpoint once settled). The conductance is the same pipes in the same
screed, so both fits estimate one physical quantity (real data: 768 W/K from
heating runs, 674 W/K from three cooling runs).
"""

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.models import FloorCircuitModel
from features.building import BuildingThermalIdentifier
from features.cop import HeatPumpCOPIdentifier
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)


class FloorCircuitIdentifier(SystemIdentifier[FloorCircuitModel]):
    # Chronological split, as for every other identifier here - but by whole
    # runs: half a run in each split would validate a run on itself.
    TRAIN_RATIO = 0.80

    # The shortest runs the heat pump makes by itself, not its typical one: a
    # plan may not ask for shorter runs than it ever runs, but it may well ask
    # for its short ones. The 10th percentile rather than the minimum, so one
    # run cut short by a defrost or a restart does not set it.
    MIN_RUNTIME_QUANTILE = 0.10
    # The least heat it moves, for the same reason a low quantile rather than
    # the minimum: a single reading caught mid-defrost or mid-stop is not the
    # compressor's lowest speed.
    MIN_HEAT_QUANTILE = 0.10

    def __init__(
        self,
        latitude: float,
        longitude: float,
        models_path: Path,
        key: Literal["heating", "cooling"] = "heating",
    ) -> None:
        super().__init__()
        # Which HeatPumpStates mode this instance fits, as in
        # HeatPumpCOPIdentifier.
        self.key = key
        # The floor's heat lands in the building's thermal mass, which only the
        # building model can estimate - so it is fitted against that estimate,
        # the same state the MPC plans with.
        self.building = BuildingThermalIdentifier(latitude, longitude)
        self.models_path = models_path
        # The configured state labels; overwritten by dataset().
        self.states = HeatPumpStates()

    @property
    def mode(self) -> str:
        return getattr(self.states, self.key)

    @property
    def sign(self) -> float:
        """The sign of the heat this mode brings into the floor."""

        return -1.0 if self.key == "cooling" else 1.0

    @property
    def name(self) -> str:
        return f"space_{self.key}"

    @property
    def label(self) -> str:
        return f"Space {self.key}"

    @property
    def unit(self) -> str:
        return "W"

    def dataset(self, config: Config) -> DatasetDefinition:
        self.states = config.heat_pump.states

        return self.building.dataset(config)

    def runs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every reading of a run in this mode, with the thermal mass's estimate, the
        run it belongs to, and whether the heat pump had settled by then.

        Past its start-up only (see HeatPumpCOPIdentifier.STARTUP): while the
        compressor ramps up, the heat delivered says nothing about the operating
        point it settles at.
        """

        self.building.load(self.models_path)

        if self.building.model is None:
            raise RuntimeError(
                "The building model must be calibrated before space heating: "
                "the floor's heat is fitted against its estimate of the mass."
            )

        prepared = self.building.prepare(df).reset_index(drop=True)
        prepared["T_mass"] = self.building.estimate(df)["mass"].to_numpy()

        running = prepared["state"] == self.mode
        prepared["run"] = (running & ~running.shift(fill_value=False)).cumsum()
        start = (
            prepared["time"].where(running).groupby(prepared["run"]).transform("min")
        )
        prepared["settled"] = running & (
            prepared["time"] - start >= HeatPumpCOPIdentifier.STARTUP
        )

        return prepared.loc[running]

    def fit(self, rows: pd.DataFrame, dt_hours: float) -> FloorCircuitModel:
        """The model from this mode's run readings (see runs()).

        The heating curve: supply = a + b * T_outdoor, over settled readings.
        Flat at their mean supply when the runs do not show the slope - a
        supply rising with the outdoor temperature, or a slope within its own
        standard error, says the outdoor range was too narrow, not that the
        curve is shaped that way. Cooling too: a warmer day asks for a colder
        supply, as a colder one asks for a warmer supply while heating.
        The floor: Q = G * (supply - T_mass), a conductance from the supply
        water to the mass. It covers both the water warming the screed and the
        water cooling on its way through the loop - G = 1 / (1 / UA_floor +
        1 / (2 m_dot c_p)) - and is fitted through the origin, because no heat
        flows at no temperature difference.

        The run length from complete runs only: one cut off by the edge of the
        data says nothing about how long it would have lasted, and one that
        never settled is a start-up that failed, not a run the heat pump
        chose.
        """

        settled = rows[rows["settled"] & (self.sign * rows["Q_floor_w"] > 0.0)]

        if settled.empty:
            raise ValueError(f"{self.label} needs at least one settled run.")

        slope, intercept = 0.0, float(settled["T_supply"].mean())

        if len(settled) > 3 and settled["T_out"].nunique() > 1:
            (fit_slope, fit_intercept), covariance = np.polyfit(
                settled["T_out"], settled["T_supply"], 1, cov=True
            )

            if 0.0 >= fit_slope and np.sqrt(covariance[0, 0]) < -fit_slope:
                slope, intercept = float(fit_slope), float(fit_intercept)
            else:
                logger.warning(
                    "%s: supply slope %.2f K/K not shown by these "
                    "runs - a flat curve at their mean supply %.1f degC.",
                    self.label,
                    fit_slope,
                    intercept,
                )

        lift = (settled["T_supply"] - settled["T_mass"]).to_numpy(dtype=float)
        heat = settled["Q_floor_w"].to_numpy(dtype=float)
        conductance = float(np.dot(lift, heat) / np.dot(lift, lift))

        lengths = (
            rows[rows["run"].isin(settled["run"])].groupby("run").size() * dt_hours
        )
        complete = lengths.iloc[1:-1] if len(lengths) > 2 else lengths
        min_runtime_hours = float(
            math.floor(complete.quantile(self.MIN_RUNTIME_QUANTILE) / dt_hours)
            * dt_hours
        )

        return FloorCircuitModel(
            supply_at_zero_outdoor_c=intercept,
            supply_per_outdoor_k=slope,
            conductance_w_per_k=conductance,
            min_runtime_hours=max(min_runtime_hours, dt_hours),
            min_heat_w=float(np.quantile(np.abs(heat), self.MIN_HEAT_QUANTILE)),
        )

    def _in_test(self, rows: pd.DataFrame) -> pd.Series:
        """Whether each reading belongs to the runs held out for validation:
        the last fifth of those that settled, but never the only one."""

        runs = rows.loc[rows["settled"], "run"].unique()
        train_runs = max(1, int(len(runs) * self.TRAIN_RATIO))

        return rows["run"].isin(runs[train_runs:])

    def calibrate(self, df: pd.DataFrame) -> FloorCircuitModel:
        rows = self.runs(df)
        train = rows[~self._in_test(rows)]
        dt_hours = float(rows["dt_seconds"].median()) / 3600.0

        self.model = self.fit(train, dt_hours)

        logger.info(
            "%s calibrated: supply = %.1f %+.2f * T_out degC, "
            "G = %.0f W/K, runs of at least %.2f h and %.0f W",
            self.label,
            self.model.supply_at_zero_outdoor_c,
            self.model.supply_per_outdoor_k,
            self.model.conductance_w_per_k,
            self.model.min_runtime_hours,
            self.model.min_heat_w,
        )

        return self.model

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        """How well the operating point predicts the heat delivered on runs it
        was not fitted on, from the outdoor temperature and the mass alone."""

        model = self.get_model()
        rows = self.runs(df)
        test = rows[
            self._in_test(rows)
            & rows["settled"]
            & (self.sign * rows["Q_floor_w"] > 0.0)
        ]

        if test.empty:
            raise ValueError(
                "No settled run held out to validate on - the model is "
                "fitted on every run there is."
            )

        predicted = model.heat_w(
            test["T_out"].to_numpy(dtype=float), test["T_mass"].to_numpy(dtype=float)
        )
        error = predicted - test["Q_floor_w"].to_numpy(dtype=float)

        metrics = {
            "mae_w": float(np.mean(np.abs(error))),
            "bias_w": float(np.mean(error)),
            "runs": float(test["run"].nunique()),
        }

        logger.info(
            "%s validation: MAE %.0f W, bias %+.0f W over %d runs",
            self.label,
            metrics["mae_w"],
            metrics["bias_w"],
            int(metrics["runs"]),
        )

        return metrics
