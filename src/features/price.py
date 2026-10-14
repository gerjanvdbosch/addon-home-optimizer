"""Dynamic electricity prices beyond those published.

The day-ahead market publishes tomorrow's prices around 13:00, so until then a
plan reaching to the end of tomorrow has to guess them. A price follows the
clock - the night's trough, the evening peak, the weekend - and the supply
that costs nothing to run: sun and wind push dearer plants out of the merit
order. A quarter hour's price is taken as the median of the same quarter hour
over the last weeks, moved by how much more sun and wind its forecast has than
is typical for it:

    price = P_med + a (S - S_med) + b (W - W_med)

with S Solcast's p50 for the own panels (kW) and W the wind's capacity factor
(-) from Open-Meteo's wind at 10 m, both standing in for the country's.
Weekdays and weekend days each have their own medians: their demand differs.
Medians rather than a single day: copying yesterday carries a Friday into a
Saturday, and one day's price spike into the next.
"""

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.optimize import lsq_linear

from domain.config import Config
from domain.dataset import DatasetDefinition
from domain.models import PriceModel
from domain.sensors import SensorAttributesReference, SensorReference
from domain.time import LOCAL_TIMEZONE
from features.dataset import DatasetBuilder
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)

STEP = pd.Timedelta(minutes=15)
QUARTERS_PER_DAY = 96

# A published price holds over its period: an hour on the day-ahead market
# until October 2025, a quarter hour since. Solcast's p50 is the mean over its
# half hour; Open-Meteo's wind the instant at each quarter hour.
PRICE_PERIOD = pd.Timedelta(hours=1)
SOLCAST_PERIOD = pd.Timedelta(minutes=30)
WIND_PERIOD = STEP

# Four weeks of each kind of day - 20 weekdays and 8 weekend days per quarter
# hour - so a median one price spike does not move, recent enough to follow
# the gas price and the season. At least a week of them.
WINDOW_WEEKS = 4
MIN_WEEKS = 1
# Weekdays and weekend days in a week.
DAYS_PER_WEEK = (5, 2)

# Open-Meteo's default wind unit (wind_speed_unit=kmh), which the sensor uses:
# a median 9.7 over June-October, a light breeze at 10 m.
KMH_PER_M_PER_S = 3.6
# The wind at hub height from that at 10 m by the power law of the wind
# profile, with exponent 1/7: open, flat land in a neutral atmosphere. Hubs at
# about 100 m, as Dutch onshore turbines have.
FORECAST_HEIGHT_M = 10.0
HUB_HEIGHT_M = 100.0
SHEAR_EXPONENT = 1.0 / 7.0
# A generic turbine: nothing below cut-in, the wind's power (v^3) above it, up
# to its rated output. Cut-out (about 25 m/s at the hub) is left out: the wind
# at 10 m here stayed below a third of what that takes (at most 40 km/h over
# June-October), and a country's turbines do not all stop at once.
CUT_IN_M_PER_S = 3.0
RATED_M_PER_S = 12.0


class PriceAttributes(BaseModel):
    price: str

    def items(self):
        return (("price", self.price),)


def price_list(sensor: SensorReference) -> SensorAttributesReference[PriceAttributes]:
    """The prices a sensor publishes ahead in its attribute, their periods'
    starts in its `time` attribute (see config.Price)."""

    return SensorAttributesReference[PriceAttributes](
        entity_id=sensor.entity_id,
        attributes=PriceAttributes(price=str(sensor.attribute)),
    )


def published_prices(sensor: SensorReference) -> DatasetDefinition:
    """The list of prices the sensor published last."""

    return (
        DatasetBuilder()
        .attribute_series("price", price_list(sensor), attributes=["price"])
        .build()
    )


def wind_capacity_factor(wind_kmh: np.ndarray) -> np.ndarray:
    """The share of their rated output wind turbines give in this wind (-)."""

    hub_m_per_s = (
        np.asarray(wind_kmh, dtype=float)
        / KMH_PER_M_PER_S
        * (HUB_HEIGHT_M / FORECAST_HEIGHT_M) ** SHEAR_EXPONENT
    )

    return np.clip(
        (hub_m_per_s**3 - CUT_IN_M_PER_S**3) / (RATED_M_PER_S**3 - CUT_IN_M_PER_S**3),
        0.0,
        1.0,
    )


def on_quarters(
    values: pd.Series,
    times: Sequence[datetime] | pd.DatetimeIndex,
    period: pd.Timedelta,
) -> np.ndarray:
    """`values`, by the start of the period each holds over, at `times`: the
    value of the period each lies in, NaN where none does. A value holds until
    the next, at most `period`; the last as long as the one before it, so the
    last of quarter-hour prices does not hold for an hour."""

    values = values.dropna().sort_index()

    if values.empty:
        return np.full(len(times), np.nan)

    values = values[~values.index.duplicated(keep="last")]
    starts = pd.DatetimeIndex(values.index).asi8
    spacing = np.diff(starts)
    held = np.append(spacing, spacing[-1] if len(spacing) else period.value)
    ends = starts + np.minimum(held, period.value)
    at = pd.DatetimeIndex(times).asi8
    i = np.clip(np.searchsorted(starts, at, side="right") - 1, 0, None)

    return np.where(
        (at >= starts[i]) & (at < ends[i]), values.to_numpy(dtype=float)[i], np.nan
    )


def _slots(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(0 on a weekday, 1 on a weekend day; the local quarter hour of the day)
    of each time."""

    local = times.tz_convert(LOCAL_TIMEZONE)

    return (
        (local.weekday >= 5).astype(int),
        np.asarray(local.hour * 4 + local.minute // 15),
    )


def _medians(series: pd.Series) -> dict[int, pd.DataFrame]:
    """Per kind of day: for every local day of that kind (rows), the median of
    each quarter hour (columns) over it and the days of its kind before it,
    WINDOW_WEEKS weeks of them."""

    kind, slot = _slots(series.index)
    day = series.index.tz_convert(LOCAL_TIMEZONE).normalize()
    table = series.groupby([kind, day, slot]).first().unstack()
    medians = {}

    for k, days in table.groupby(level=0):
        n = DAYS_PER_WEEK[k]
        medians[k] = (
            days.droplevel(0)
            .rolling(WINDOW_WEEKS * n, min_periods=MIN_WEEKS * n)
            .median()
            .reindex(columns=range(QUARTERS_PER_DAY))
        )

    return medians


def _typical(series: pd.Series) -> np.ndarray:
    """For every time, the median of its quarter hour as known before its day
    began (see _medians); NaN without enough days before it."""

    kind, slot = _slots(series.index)
    day = series.index.tz_convert(LOCAL_TIMEZONE).normalize()
    typical = np.full(len(series), np.nan)

    for k, table in _medians(series).items():
        at = kind == k
        before = table.shift(1).reindex(day[at]).to_numpy()
        typical[at] = before[np.arange(at.sum()), slot[at]]

    return typical


def _profile(series: pd.Series) -> np.ndarray:
    """The medians as they stand at the end of `series`: (weekday, weekend) x
    quarter hour of the day."""

    profile = np.full((2, QUARTERS_PER_DAY), np.nan)

    for k, table in _medians(series).items():
        profile[k] = table.iloc[-1].to_numpy()

    return profile


def predict_price(
    model: PriceModel,
    times: Sequence[datetime] | pd.DatetimeIndex,
    solar_kw: np.ndarray,
    wind_kmh: np.ndarray,
) -> np.ndarray:
    """The price (EUR/kWh) at `times`, given Solcast's p50 (kW) and
    Open-Meteo's wind at 10 m (km/h) there. Where either is unknown (NaN) it
    counts as typical: the price is then the quarter hour's median."""

    kind, slot = _slots(pd.DatetimeIndex(times))
    solar = np.nan_to_num(solar_kw - model.solar_profile_kw[kind, slot])
    wind = np.nan_to_num(
        wind_capacity_factor(wind_kmh) - model.wind_profile[kind, slot]
    )

    return (
        model.price_profile[kind, slot]
        + model.solar_eur_per_kwh_per_kw * solar
        + model.wind_eur_per_kwh * wind
    )


class PriceIdentifier(SystemIdentifier[PriceModel]):
    TRAIN_RATIO = 0.80
    # How often the history of each forecast is sampled: hourly, holding the
    # last one published, so all three share their snapshot times.
    SNAPSHOT_INTERVAL = "1h"

    def __init__(self, key: Literal["import", "export"] = "import") -> None:
        super().__init__()
        self.key = key

    @property
    def name(self) -> str:
        return f"price_{self.key}"

    @property
    def label(self) -> str:
        return f"{self.key.capitalize()} price"

    @property
    def unit(self) -> str:
        return "EUR/kWh"

    def dataset(self, config: Config) -> DatasetDefinition:
        # Each its own fit: both follow the market price, by a factor and an
        # offset of their own, which the medians and the fit scale with.
        sensor = config.prices.import_ if self.key == "import" else config.prices.export

        if not isinstance(sensor, SensorReference) or sensor.attribute is None:
            raise ValueError(
                f"No price sensor with the attribute it publishes its prices in "
                f"configured (prices.{self.key})."
            )

        # Every forecast published, the prices too, held between publications
        # and matched on when it was known and the time it is for: prepare()
        # keeps what was known by then.
        builder = DatasetBuilder()

        for name, source, attributes in (
            ("open_meteo", config.forecast.open_meteo, ["wind_speed"]),
            ("solcast", config.forecast.solcast, ["p50"]),
            ("price", price_list(sensor), ["price"]),
        ):
            builder = builder.attribute_timeseries(
                name,
                source,
                attributes=attributes,
                interval=self.SNAPSHOT_INTERVAL,
                aggregation="last",
                fill="previous",
            )

        return (
            builder.join(left="open_meteo", right="solcast", on=("time", "target_time"))
            .join(left="open_meteo", right="price", on=("time", "target_time"))
            .build()
        )

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per quarter hour: the price (EUR/kWh), Solcast's p50 (kW) and the
        wind's capacity factor (-), each as last published by then. A snapshot
        holds the last publication within its hour, so it is known by the
        hour's end. A published price is the same in every snapshot; the
        weather is the forecast closest to the time, which is somewhat better
        than a plan has for tomorrow."""

        df = df[df["time"] + pd.Timedelta(self.SNAPSHOT_INTERVAL) <= df["target_time"]]
        # An attribute never published leaves its column out (see
        # resolve_attributes).
        latest = (
            df.sort_values("time")
            .groupby("target_time")
            .last()
            .reindex(columns=["price", "p50", "wind_speed"])
        )
        known = latest["price"].last_valid_index()

        if known is None:
            raise ValueError("No price known.")

        # Up to the last price known: the weather is forecast beyond it.
        latest = latest.loc[:known]

        index = pd.date_range(latest.index.min(), latest.index.max(), freq=STEP)

        return pd.DataFrame(
            {
                "price": on_quarters(latest["price"], index, PRICE_PERIOD),
                "solar_kw": on_quarters(latest["p50"], index, SOLCAST_PERIOD) / 1000.0,
                "wind": wind_capacity_factor(
                    on_quarters(latest["wind_speed"], index, WIND_PERIOD)
                ),
            },
            index=index,
        )

    def _deviations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Each quantity less its typical value at that quarter hour."""

        return pd.DataFrame(
            {column: data[column] - _typical(data[column]) for column in data},
            index=data.index,
        )

    def _split(self, data: pd.DataFrame) -> pd.Timestamp:
        return data.index[int(len(data) * self.TRAIN_RATIO)]

    def calibrate(self, df: pd.DataFrame) -> PriceModel:
        data = self.prepare(df)
        deviations = self._deviations(data)
        train = deviations[deviations.index < self._split(data)].dropna()

        if len(train) < QUARTERS_PER_DAY * 7:
            raise ValueError(
                f"Prices need at least a week of quarter hours with a typical "
                f"price, sun and wind to fit on, found {len(train)}."
            )

        profile = _profile(data["price"])

        if np.isnan(profile).any():
            raise ValueError(
                "Prices need at least a week of weekdays and weekend days for "
                "the typical price of every quarter hour."
            )

        # More supply that costs nothing can only lower the price.
        fit = lsq_linear(
            train[["solar_kw", "wind"]].to_numpy(),
            train["price"].to_numpy(),
            bounds=([-np.inf, -np.inf], [0.0, 0.0]),
        )
        self.model = PriceModel(
            price_profile=profile,
            solar_profile_kw=_profile(data["solar_kw"]),
            wind_profile=_profile(data["wind"]),
            solar_eur_per_kwh_per_kw=float(fit.x[0]),
            wind_eur_per_kwh=float(fit.x[1]),
        )

        logger.info(
            "Price (%s) calibrated on %d quarter hours: %.4f EUR/kWh per kW of "
            "sun, %.4f EUR/kWh per unit of wind capacity factor",
            self.key,
            len(train),
            self.model.solar_eur_per_kwh_per_kw,
            self.model.wind_eur_per_kwh,
        )

        return self.model

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        """The quarter hours held out, forecast from the medians known before
        their day and the weather, beside the medians alone and the price a
        day earlier (MAE, EUR/kWh)."""

        model = self.get_model()
        data = self.prepare(df)
        deviations = self._deviations(data)
        typical = pd.Series(_typical(data["price"]), index=data.index)
        forecast = (
            typical
            + model.solar_eur_per_kwh_per_kw * deviations["solar_kw"].fillna(0.0)
            + model.wind_eur_per_kwh * deviations["wind"].fillna(0.0)
        )
        day_earlier = data["price"].shift(freq=pd.Timedelta(days=1)).reindex(data.index)
        test = (data.index >= self._split(data)) & typical.notna() & day_earlier.notna()

        if not test.any():
            raise ValueError("No quarter hour held out to validate on.")

        price = data["price"][test]
        metrics = {
            "quarters": float(test.sum()),
            "mae": float((forecast[test] - price).abs().mean()),
            "median_mae": float((typical[test] - price).abs().mean()),
            "day_earlier_mae": float((day_earlier[test] - price).abs().mean()),
        }

        logger.info(
            "Price (%s) validation over %d quarter hours: MAE %.4f EUR/kWh, "
            "medians alone %.4f, a day earlier %.4f",
            self.key,
            test.sum(),
            metrics["mae"],
            metrics["median_mae"],
            metrics["day_earlier_mae"],
        )

        return metrics
