import numpy as np
import pandas as pd
import pytest

from domain.models import DewPointModel
from domain.physics import (
    dew_point_c,
    dew_point_from_humidity_c,
    indoor_dew_point_c,
    vapour_pressure_pa,
)
from features.dew_point import DewPointIdentifier

TRUE = DewPointModel(time_constant_hours=30.0, moisture_surplus_pa=120.0)


def test_magnus_matches_a_known_dew_point_both_ways():
    # 20 degC at 50%: 9.3 degC in any psychrometric table.
    assert dew_point_from_humidity_c(20.0, 50.0) == pytest.approx(9.26, abs=0.02)
    assert dew_point_c(vapour_pressure_pa(13.7)) == pytest.approx(13.7)


def test_the_indoor_dew_point_relaxes_to_the_outdoor_one_plus_its_surplus():
    """Its first step is the exact exponential one, and after many time
    constants it sits at the outdoor vapour pressure plus the surplus."""

    outdoor = np.full(400, 10.0)
    run = indoor_dew_point_c(TRUE, 16.0, outdoor, 1.0)

    weight = 1.0 - np.exp(-1.0 / TRUE.time_constant_hours)
    e0, e_out = vapour_pressure_pa(16.0), vapour_pressure_pa(10.0)
    first = e0 + weight * (e_out + TRUE.moisture_surplus_pa - e0)

    assert run[0] == pytest.approx(16.0)
    assert run[1] == pytest.approx(dew_point_c(first))
    assert run[-1] == pytest.approx(
        dew_point_c(e_out + TRUE.moisture_surplus_pa), abs=1e-3
    )

    # Many starts at once give the runs each would alone.
    starts = np.array([12.0, 16.0])
    batch = indoor_dew_point_c(TRUE, starts, np.tile(outdoor[:24], (2, 1)), 1.0)
    assert batch[1] == pytest.approx(run[:24])


def _hours(days: int = 30) -> pd.DataFrame:
    """A month of hourly readings from the true model: outdoor dew point
    swinging daily and over days, indoor following it. The first half has no
    stored outdoor dew point, only the temperature and humidity it follows
    from - the fallback."""

    hours = np.arange(days * 24)
    outdoor = 12.0 + 3.0 * np.sin(2 * np.pi * hours / 24) + 2.5 * np.sin(hours / 50.0)
    indoor = indoor_dew_point_c(TRUE, 14.0, outdoor, 1.0)

    temperature = np.full(len(hours), 22.0)
    humidity = 100.0 * vapour_pressure_pa(outdoor) / vapour_pressure_pa(temperature)
    stored = np.where(hours < len(hours) // 2, np.nan, outdoor)
    time = pd.date_range("2026-08-01", periods=len(hours), freq="1h", tz="UTC")

    return pd.DataFrame(
        {
            "time": time,
            "dew_point_0": indoor,
            "time_open_meteo": time,
            "temperature": temperature,
            "relative_humidity": humidity,
            "dew_point": stored,
        }
    )


def test_calibration_recovers_the_balance_and_beats_holding_it():
    identifier = DewPointIdentifier()
    df = _hours()

    model = identifier.calibrate(df)
    metrics = identifier.validate(df)

    assert model.time_constant_hours == pytest.approx(30.0, rel=0.02)
    assert model.moisture_surplus_pa == pytest.approx(120.0, rel=0.02)
    assert metrics["rmse_24h"] < 0.01
    assert metrics["rmse_24h"] < metrics["persistence_rmse_24h"]


def test_prepare_keeps_the_forecast_known_by_then():
    """Of two forecasts for the same hour, the later one published before it
    counts; one published after it was not known then."""

    time = pd.Timestamp("2026-08-01T12:00Z")
    df = pd.DataFrame(
        {
            "time": [time] * 3,
            "dew_point_0": [15.0] * 3,
            "time_open_meteo": [
                time - pd.Timedelta(hours=6),
                time - pd.Timedelta(hours=1),
                time + pd.Timedelta(hours=1),
            ],
            "dew_point": [10.0, 11.0, 12.0],
        }
    )

    prepared = DewPointIdentifier().prepare(df)

    assert prepared["outside"].iloc[0] == 11.0
    assert prepared["inside"].iloc[0] == 15.0
