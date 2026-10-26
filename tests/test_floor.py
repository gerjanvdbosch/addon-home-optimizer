from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domain.physics import (
    CP_WATER_J_PER_KG_K,
    RHO_WATER_KG_PER_L,
    floor_circuit_conductance_w_per_k,
    floor_pipe_conductance_w_per_k,
)
from features.floor import FloorCircuitIdentifier

SUPPLY_AT_ZERO_C = 34.0
SUPPLY_PER_OUTDOOR_K = -0.5
CONDUCTANCE_W_PER_K = 900.0


def _runs(lengths: list[int], rng: np.random.Generator) -> pd.DataFrame:
    """Heating runs as runs() returns them: each at its own outdoor
    temperature, on the heat pump's curve, delivering what the floor takes."""

    rows = []
    time = pd.Timestamp("2026-11-01T00:00Z")

    for run, length in enumerate(lengths, start=1):
        outdoor = rng.uniform(-5.0, 12.0)

        for step in range(length):
            supply = SUPPLY_AT_ZERO_C + SUPPLY_PER_OUTDOOR_K * outdoor
            supply += rng.normal(0.0, 0.2)
            mass = 20.0 + rng.normal(0.0, 0.3)
            rows.append(
                {
                    "time": time,
                    "run": run,
                    "settled": step > 0,
                    "T_out": outdoor,
                    "T_supply": supply,
                    "T_mass": mass,
                    "open": 1.0,
                    "floor_flow_lpm": 18.0,
                    "Q_floor_w": CONDUCTANCE_W_PER_K * (supply - mass),
                }
            )
            time += pd.Timedelta(minutes=15)

        time += pd.Timedelta(hours=3)

    return pd.DataFrame(rows)


def _identifier() -> FloorCircuitIdentifier:
    return FloorCircuitIdentifier(52.0, 5.0, Path("."))


def test_fit_recovers_the_curve_and_the_floor():
    # The first and the last run are cut off by the edges of the data.
    lengths = [8, 12, 16, 10, 20, 6, 14, 18]
    model = _identifier().fit(_runs(lengths, np.random.default_rng(3)))

    assert model.supply_per_outdoor_k == pytest.approx(SUPPLY_PER_OUTDOOR_K, abs=0.05)
    assert model.supply_at_zero_outdoor_c == pytest.approx(SUPPLY_AT_ZERO_C, abs=0.5)
    assert model.conductance_w_per_k == pytest.approx(CONDUCTANCE_W_PER_K, rel=0.05)


def test_one_run_gives_a_flat_curve_at_its_supply():
    """One run sits at one outdoor temperature: it shows the floor, but no
    slope for the curve."""

    rows = _runs([14], np.random.default_rng(4))
    model = _identifier().fit(rows)

    assert model.supply_per_outdoor_k == 0.0
    assert model.supply_at_zero_outdoor_c == pytest.approx(
        rows.loc[rows["settled"], "T_supply"].mean()
    )
    assert model.conductance_w_per_k == pytest.approx(CONDUCTANCE_W_PER_K, rel=0.05)


def test_split_keeps_runs_whole_and_the_only_run_for_training():
    identifier = _identifier()
    rows = _runs([8, 12, 16, 10, 20], np.random.default_rng(5))

    assert set(rows.loc[identifier._in_test(rows), "run"]) == {5}
    assert not identifier._in_test(_runs([14], np.random.default_rng(6))).any()


def test_a_run_through_part_of_the_loops_takes_their_share_of_the_pipes():
    """Rooms that close their loops leave the water the rest of the floor:
    half its pipes, with more of the pump's flow through each. The water
    cools less along them, so half the loops pass more than half the heat -
    and the pipes' own UA comes out, as the whole circuit at its flow."""

    pipe_ua_w_per_k = 1500.0
    rows = _runs([8, 12, 16, 10, 20, 6, 14, 18], np.random.default_rng(3))
    half = rows["run"] % 2 == 0
    rows.loc[half, "open"] = 0.5
    rows.loc[half, "floor_flow_lpm"] = 16.0
    capacity = RHO_WATER_KG_PER_L * rows["floor_flow_lpm"] / 60.0 * CP_WATER_J_PER_KG_K
    rows["Q_floor_w"] = (
        capacity
        * (1.0 - np.exp(-rows["open"] * pipe_ua_w_per_k / capacity))
        * (rows["T_supply"] - rows["T_mass"])
    )

    model = _identifier().fit(rows)
    every_loop = floor_circuit_conductance_w_per_k(model)
    half_open = floor_circuit_conductance_w_per_k(model, 0.5, 16.0)

    assert model.pipe_ua_w_per_k == pytest.approx(pipe_ua_w_per_k, rel=0.01)
    assert model.flow_lpm == 18.0
    assert model.conductance_w_per_k == pytest.approx(every_loop)
    assert 0.5 * every_loop < half_open < every_loop


def test_cooling_fit_reads_runs_that_take_heat_from_the_floor():
    """Cooling runs mirror heating ones: supply below the mass, heat negative,
    and a warmer day on a colder supply. The same floor comes out."""

    rows = _runs([8, 12, 16, 10, 20, 6, 14, 18], np.random.default_rng(3))
    rows["T_out"] = 35.0 - rows["T_out"]
    rows["T_supply"] = 40.0 - rows["T_supply"]
    rows["Q_floor_w"] = CONDUCTANCE_W_PER_K * (rows["T_supply"] - rows["T_mass"])

    identifier = FloorCircuitIdentifier(52.0, 5.0, Path("."), key="cooling")
    model = identifier.fit(rows)

    assert identifier.name == "space_cooling"
    assert model.supply_per_outdoor_k == pytest.approx(SUPPLY_PER_OUTDOOR_K, abs=0.05)
    assert model.conductance_w_per_k == pytest.approx(CONDUCTANCE_W_PER_K, rel=0.05)
    assert model.heat_w(30.0, 20.0) < 0.0


def test_the_water_implies_the_conductance_it_was_given_either_way():
    """A return set by a known conductance gives that conductance back, while
    heating and while cooling; one beyond the supply or the mass, none."""

    flow = np.array([18.0, 18.0, 18.0])
    capacity = RHO_WATER_KG_PER_L * 18.0 / 60.0 * CP_WATER_J_PER_KG_K
    supply = np.array([25.0, 14.0, 25.0])
    mass = np.array([20.5, 19.5, 20.5])
    remaining = np.exp(-1100.0 / capacity)
    ret = mass + (supply - mass) * remaining
    ret[2] = 26.0

    conductance = floor_pipe_conductance_w_per_k(flow, supply, ret, mass)

    assert conductance[:2] == pytest.approx([1100.0, 1100.0])
    assert np.isnan(conductance[2])
