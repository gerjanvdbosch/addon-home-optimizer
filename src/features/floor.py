"""How this heat pump runs the floor circuit when it is left to itself.

The heat pump chooses its own supply temperature - from its heating curve and
modulation, running long at low power rather than cycling - so the heat a
space-heating run delivers is not a decision a plan can make. What a plan
decides is when the zone is heated; how much heat that brings follows from
the curve and from the floor it heats. This identifies both from its own
heating runs. How long a run lasts is not identified here: a run's length
shows the demand of its day (real cooling runs: 20 min to 5 h), not a limit
of the machine - that is the compressor's own minimum runtime, the same in
every mode (MPCConfig.compressor_min_runtime_steps).

Cooling is the same floor run the other way, identified the same way from
cooling runs: the heat is negative, the supply below the mass, and the supply
is the one the heat pump holds by itself (real data: 0.5 K below its zone
flow setpoint once settled). The conductance is the same pipes in the same
screed, so both fits estimate one physical quantity. It is to the screed the
building model estimates where it has one (see
BuildingThermalIdentifier._fit_screed), which warms within the hour: against
the slabs instead, one conductance read too little heat at a run's start and
too much later (real data, 33 runs: 424 W off on runs left out, 207 W with
the screed).

The conductance is the pipes' UA and the water's cooling along them (see
physics.floor_circuit_conductance_w_per_k): the UA the whole floor's, every
loop open, and where rooms have closed theirs (see Room.climate) the open
loops' share of it by floor area, at the flow the pump then drives through
them. Fewer loops open, more water through each: 26 Oct 2026, 57% of the
floor open, the flow fell from 18.0 to 16.0 L/min only. Over the runs with
loops closed (Oct 2026, 15 settled steps) the heat was predicted 234 W off
(+59 W) this way, 294 W (-232 W) with the conductance by area alone - the
same 234 W on the runs left out with every loop open, the flow there
hardly varying (18.0 +- 0.1 L/min).
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.models import FloorCircuitModel
from domain.physics import (
    floor_circuit_conductance_w_per_k,
    floor_pipe_conductance_w_per_k,
)
from features.building import BuildingThermalIdentifier
from features.cop import HeatPumpCOPIdentifier
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)


class FloorCircuitIdentifier(SystemIdentifier[FloorCircuitModel]):
    # Chronological split, as for every other identifier here - but by whole
    # runs: half a run in each split would validate a run on itself.
    TRAIN_RATIO = 0.80

    # The least heat it moves: a low quantile rather than the minimum, as a
    # single reading caught mid-defrost or mid-stop is not the compressor's
    # lowest speed.
    MIN_HEAT_QUANTILE = 0.10
    # The most it moves, the same way: a high quantile, so one reading caught
    # in a transient does not set it.
    MAX_HEAT_QUANTILE = 0.90

    # How long the screed between the pipes takes to settle against the water:
    # lateral conduction over half a pipe spacing, (7.5 cm)^2 / 6e-7 m2/s, some
    # 2.5 hours at a usual 15 cm spacing - assumed, this floor's is not known.
    # Before it the water sees screed nearer its own temperature than the mass
    # node's mean, so the conductance a run implies is higher early on (real
    # data, Aug-Oct 2026: about 1300 W/K in the first half hour, 950-1000 W/K
    # after two hours).
    SCREED_SETTLE = pd.Timedelta(hours=2)

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
        # What the water delivers against: the screed it lies in where the
        # building has one, else its slabs (see physics.zone_floor) - of the
        # floors whose loops were open.
        prepared["T_mass"] = self.building.estimate(df)["floor"].to_numpy()
        # The share of the circuit the water ran through, its flow in
        # floor_flow_lpm (see prepare).
        prepared["open"] = prepared["floor_open_living"] + prepared["floor_open_rest"]

        running = prepared["state"] == self.mode
        prepared["run"] = (running & ~running.shift(fill_value=False)).cumsum()
        start = (
            prepared["time"].where(running).groupby(prepared["run"]).transform("min")
        )
        prepared["settled"] = running & (
            prepared["time"] - start >= HeatPumpCOPIdentifier.STARTUP
        )

        return prepared.loc[running]

    def fit(self, rows: pd.DataFrame) -> FloorCircuitModel:
        """The model from this mode's run readings (see runs()).

        The heating curve: supply = a + b * T_outdoor, over settled readings.
        Flat at their mean supply when the runs do not show the slope - a
        supply rising with the outdoor temperature, or a slope within its own
        standard error, says the outdoor range was too narrow, not that the
        curve is shaped that way. Cooling too: a warmer day asks for a colder
        supply, as a colder one asks for a warmer supply while heating.
        The floor: Q = G (supply - T_mass), a conductance from the supply
        water to the mass, through the origin because no heat flows at no
        temperature difference. It covers both the pipes warming the screed
        and the water cooling on its way through them: G = m_dot c_p (1 -
        exp(-open UA / (m_dot c_p))) (see
        physics.floor_circuit_conductance_w_per_k), UA fitted at each
        reading's open share and flow.
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
        open_ = settled["open"].to_numpy(dtype=float)
        flow = settled["floor_flow_lpm"].to_numpy(dtype=float)
        model = FloorCircuitModel(
            supply_at_zero_outdoor_c=intercept,
            supply_per_outdoor_k=slope,
            conductance_w_per_k=0.0,
            min_heat_w=float(np.quantile(np.abs(heat), self.MIN_HEAT_QUANTILE)),
            max_heat_w=float(np.quantile(np.abs(heat), self.MAX_HEAT_QUANTILE)),
            flow_lpm=float(np.median(flow)),
        )

        def residuals(ua: np.ndarray) -> np.ndarray:
            conductance = floor_circuit_conductance_w_per_k(
                replace(model, pipe_ua_w_per_k=float(ua[0])), open_, flow
            )

            # In kW, so the residuals sit near 1.
            return (conductance * lift - heat) / 1000.0

        # From the conductance by area alone, which UA is never below: the
        # water cooling along the pipes only takes from it.
        by_area = float(np.dot(open_ * lift, heat) / np.dot(open_ * lift, open_ * lift))
        fit = least_squares(residuals, [by_area], bounds=(0.0, np.inf))
        model.pipe_ua_w_per_k = float(fit.x[0])
        model.conductance_w_per_k = float(floor_circuit_conductance_w_per_k(model))

        return model

    def _in_test(self, rows: pd.DataFrame) -> pd.Series:
        """Whether each reading belongs to the runs held out for validation:
        the last fifth of those that settled, but never the only one."""

        runs = rows.loc[rows["settled"], "run"].unique()
        train_runs = max(1, int(len(runs) * self.TRAIN_RATIO))

        return rows["run"].isin(runs[train_runs:])

    def calibrate(self, df: pd.DataFrame) -> FloorCircuitModel:
        rows = self.runs(df)
        train = rows[~self._in_test(rows)]
        self.model = self.fit(train)

        logger.info(
            "%s calibrated: supply = %.1f %+.2f * T_out degC, "
            "G = %.0f W/K (UA %.0f W/K at %.1f L/min), %.0f-%.0f W",
            self.label,
            self.model.supply_at_zero_outdoor_c,
            self.model.supply_per_outdoor_k,
            self.model.conductance_w_per_k,
            self.model.pipe_ua_w_per_k,
            self.model.flow_lpm,
            self.model.min_heat_w,
            self.model.max_heat_w,
        )

        return self.model

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        """How well the model predicts the heat delivered on runs it was not
        fitted on: Q = G (supply - mass) at the supply the heat pump ran,
        since the plan chooses the supply itself wherever its mode has a COP
        model (see MPCOptimizer._add_space_heating). The heating curve is
        scored on its own, as the supply it predicts: through the heat it
        mixed the curve's miss into G's (real data, Oct 2026: the heat pump's
        own room-led flow control, not the outdoor temperature, set the
        supply, 642 W off through the curve against 256 W through G)."""

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

        supply = test["T_supply"].to_numpy(dtype=float)
        predicted = floor_circuit_conductance_w_per_k(
            model,
            test["open"].to_numpy(dtype=float),
            test["floor_flow_lpm"].to_numpy(dtype=float),
        ) * (supply - test["T_mass"].to_numpy(dtype=float))
        error = predicted - test["Q_floor_w"].to_numpy(dtype=float)
        supply_error = model.supply_c(test["T_out"].to_numpy(dtype=float)) - supply

        metrics = {
            "mae_w": float(np.mean(np.abs(error))),
            "bias_w": float(np.mean(error)),
            "supply_mae_k": float(np.mean(np.abs(supply_error))),
            "runs": float(test["run"].nunique()),
            **self._pipe_conductance(rows),
        }

        logger.info(
            "%s validation: MAE %.0f W, bias %+.0f W over %d runs, the curve's "
            "supply %.2f K off; the water implies %.0f W/K to the mass (%.0f "
            "before the screed settles, %.0f after) over %d runs",
            self.label,
            metrics["mae_w"],
            metrics["bias_w"],
            int(metrics["runs"]),
            metrics["supply_mae_k"],
            metrics["pipe_ua_w_per_k"],
            metrics["pipe_ua_early_w_per_k"],
            metrics["pipe_ua_late_w_per_k"],
            int(metrics["pipe_ua_runs"]),
        )

        return metrics

    def _pipe_conductance(self, rows: pd.DataFrame) -> dict[str, float]:
        """The building model's mass checked against the return, the one sensor
        that sees the floor: the conductance from the water to the mass every
        settled run implies (see physics.floor_pipe_conductance_w_per_k), by
        median, over all runs - a check of the mass estimate, not of this
        model's fit.

        The pipes are the same either way, so heating and cooling should
        agree; a mass estimate off by d K moves them apart, by about
        m_dot c_p (1 / (return - mass) - 1 / (supply - mass)) d each - some
        300 W/K per K in either direction here. Within a run it falls until
        the screed settles (see SCREED_SETTLE): what the mass node, one
        temperature for the whole slab, cannot show, and why the return is
        not an observation of it. The whole floor's, every loop open, as
        FloorCircuitModel.pipe_ua_w_per_k is.
        """

        settled = rows[rows["settled"] & (self.sign * rows["Q_floor_w"] > 0.0)]
        conductance = pd.Series(
            floor_pipe_conductance_w_per_k(
                settled["floor_flow_lpm"].to_numpy(dtype=float),
                settled["T_supply"].to_numpy(dtype=float),
                settled["T_return"].to_numpy(dtype=float),
                settled["T_mass"].to_numpy(dtype=float),
            )
            / settled["open"].to_numpy(dtype=float),
            index=settled.index,
        )
        start = rows["time"].groupby(rows["run"]).transform("min")
        early = (settled["time"] - start[settled.index]) < self.SCREED_SETTLE

        return {
            "pipe_ua_w_per_k": float(conductance.median()),
            "pipe_ua_early_w_per_k": float(conductance[early].median()),
            "pipe_ua_late_w_per_k": float(conductance[~early].median()),
            "pipe_ua_runs": float(settled.loc[conductance.notna(), "run"].nunique()),
        }
