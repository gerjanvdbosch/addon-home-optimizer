from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
