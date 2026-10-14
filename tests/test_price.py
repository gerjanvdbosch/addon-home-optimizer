from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.optimization import Optimization
from domain.config import PricesConfig
from domain.sensors import SensorReference
from domain.state import SeriesPoint, State
from features.price import (
    PRICE_PERIOD,
    PriceIdentifier,
    on_quarters,
    predict_price,
    wind_capacity_factor,
)

START = datetime(2026, 8, 31, 22, 0, tzinfo=UTC)  # local midnight, a Monday
DAYS = 42
SOLAR_EUR_PER_KWH_PER_KW = -0.02
WIND_EUR_PER_KWH = -0.10


def _prices(times: pd.DatetimeIndex, solar_kw: np.ndarray, wind_kmh: np.ndarray):
    """A clock-driven price - an evening peak, cheaper weekends - lowered by
    sun and wind, as the model assumes."""

    local = times.tz_convert("Europe/Amsterdam")
    hour = np.asarray(local.hour + local.minute / 60)
    clock = 0.22 + 0.08 * np.exp(-(((hour - 18.5) / 1.5) ** 2))
    weekend = np.where(np.asarray(local.weekday) >= 5, -0.03, 0.0)

    return (
        clock
        + weekend
        + SOLAR_EUR_PER_KWH_PER_KW * solar_kw
        + WIND_EUR_PER_KWH * wind_capacity_factor(wind_kmh)
    )


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    """Six weeks as the joined snapshots come in: each quarter hour published
    two hours ahead, with a sunny or dull, calm or windy day at random."""

    rng = np.random.default_rng(1)
    times = pd.date_range(START, periods=DAYS * 96, freq="15min")
    hour = np.asarray(times.tz_convert("Europe/Amsterdam").hour)
    day = np.arange(len(times)) // 96
    sun = rng.uniform(0.0, 3.0, DAYS)[day] * np.clip(
        np.sin((hour - 7) / 12 * np.pi), 0, 1
    )
    wind = rng.uniform(0.0, 40.0, DAYS)[day]
    price = _prices(times, sun, wind) + rng.normal(0.0, 0.005, len(times))

    return pd.DataFrame(
        {
            "time": times - pd.Timedelta(hours=2),
            "target_time": times,
            "price": price,
            "p50": sun * 1000.0,
            "wind_speed": wind,
        }
    )


@pytest.fixture(scope="module")
def identifier(history: pd.DataFrame) -> PriceIdentifier:
    identifier = PriceIdentifier()
    identifier.calibrate(history)

    return identifier


def test_turbines_give_nothing_in_a_calm_and_all_in_a_gale():
    factor = wind_capacity_factor(np.array([0.0, 5.0, 15.0, 30.0, 60.0]))

    assert factor[0] == factor[1] == 0.0
    assert np.all(np.diff(factor) >= 0.0)
    assert factor[-1] == 1.0


def test_a_published_price_holds_over_its_period_only():
    """An hourly price over its four quarter hours; the last of quarter-hour
    prices over its own quarter hour only."""

    times = [START + timedelta(minutes=15 * i) for i in range(5)]
    hourly = pd.Series([0.20], index=pd.DatetimeIndex([START]))
    quarterly = pd.Series([0.20, 0.25], index=pd.DatetimeIndex(times[:2]))

    hourly_values = on_quarters(hourly, times, PRICE_PERIOD)
    quarterly_values = on_quarters(quarterly, times, PRICE_PERIOD)

    assert hourly_values[:4].tolist() == [0.20] * 4
    assert np.isnan(hourly_values[4])
    assert quarterly_values[:2].tolist() == [0.20, 0.25]
    assert np.isnan(quarterly_values[2:]).all()


def test_sun_and_wind_are_found_as_they_lower_the_price(identifier):
    model = identifier.get_model()

    assert model.solar_eur_per_kwh_per_kw == pytest.approx(
        SOLAR_EUR_PER_KWH_PER_KW, rel=0.2
    )
    assert model.wind_eur_per_kwh == pytest.approx(WIND_EUR_PER_KWH, rel=0.2)


def test_the_forecast_beats_the_medians_and_yesterday(identifier, history):
    metrics = identifier.validate(history)

    assert metrics["mae"] < metrics["median_mae"]
    assert metrics["mae"] < metrics["day_earlier_mae"]


def test_unknown_weather_leaves_the_typical_price(identifier):
    model = identifier.get_model()
    # Monday local noon: a weekday.
    noon = [START + timedelta(days=DAYS, hours=12)]

    price = predict_price(model, noon, np.array([np.nan]), np.array([np.nan]))

    assert price[0] == model.price_profile[0, 48]


def test_import_and_export_each_take_a_price_sensor():
    """Dynamic, they follow the market price apart: the import with tax and
    margin, the export without."""

    prices = PricesConfig.model_validate(
        {"import": ["sensor.import_prices", "price"], "export": "sensor.export_prices"}
    )

    assert prices.import_ == SensorReference(
        entity_id="sensor.import_prices", attribute="price"
    )
    assert prices.export == SensorReference(entity_id="sensor.export_prices")


class _Published:
    """Today's prices published, tomorrow's not yet."""

    def __init__(self, times: list[datetime], prices: list[float]) -> None:
        self.frame = pd.DataFrame({"time": times, "price": prices})

    def load(self, dataset, start, end) -> pd.DataFrame:
        return self.frame


def _sensor_price(
    models_path: Path,
    times: list[datetime],
    published: int,
    state: State,
    attribute: str | None = "price",
) -> tuple[float, ...]:
    optimization = Optimization(
        loader=_Published(times[:published], [0.30] * published),  # type: ignore[arg-type]
        state_manager=None,  # type: ignore[arg-type]
        config_repository=None,  # type: ignore[arg-type]
        models_path=models_path,
        home_assistant=None,  # type: ignore[arg-type]
    )

    return optimization._price(
        "import",
        SensorReference(entity_id="sensor.prices", attribute=attribute),
        state,
        times,
    )


def _weather(times: list[datetime]) -> State:
    """Half a kW of sun and 20 km/h of wind throughout."""

    state = State()
    state.forecast.solcast.p50 = [SeriesPoint(time=t, value=500.0) for t in times]
    state.forecast.open_meteo.wind_speed = [
        SeriesPoint(time=t, value=20.0) for t in times
    ]

    return state


def test_steps_not_published_are_forecast(identifier, tmp_path):
    """Today's prices published, tomorrow's not yet."""

    times = [START + timedelta(days=DAYS, minutes=15 * i) for i in range(8)]
    identifier.save(tmp_path)

    prices = _sensor_price(tmp_path, times, 4, _weather(times))

    assert prices[:4] == (0.30,) * 4
    assert prices[4:] == tuple(
        predict_price(
            identifier.get_model(), times[4:], np.full(4, 0.5), np.full(4, 20.0)
        ).tolist()
    )


def test_without_a_forecast_the_published_mean_stands_in(tmp_path):
    times = [START + timedelta(minutes=15 * i) for i in range(8)]

    assert _sensor_price(tmp_path, times, 4, State()) == pytest.approx((0.30,) * 8)


def test_a_sensor_without_its_price_list_leaves_flat_prices(tmp_path):
    times = [START + timedelta(minutes=15 * i) for i in range(4)]

    assert _sensor_price(tmp_path, times, 4, State(), attribute=None) == ()
