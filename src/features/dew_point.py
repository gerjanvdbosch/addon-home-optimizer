"""How the indoor dew point follows the outdoor one.

A cooled floor, and uninsulated pipes carrying the chilled water, may not be
colder than the dew point of the air around them, so a plan needs that dew
point over its horizon, not just now. Nothing forecasts it indoors, but
ventilation carries outdoor humidity in, which Open-Meteo does forecast: the
indoor dew point is identified here as a moisture balance driven by the outdoor
one (see physics.indoor_dew_point_c). On real data the indoor dew point moved a
median 1.0 K over 24 hours (p90 2.3 K) and tracked the outdoor one, 1 K above
it, with a correlation of 0.84 three hours later.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from domain.config import Config
from domain.dataset import DatasetDefinition
from domain.models import DewPointModel
from domain.physics import dew_point_from_humidity_c, indoor_dew_point_c
from features.dataset import DatasetBuilder
from features.identifier import SystemIdentifier

logger = logging.getLogger(__name__)


class DewPointIdentifier(SystemIdentifier[DewPointModel]):
    TRAIN_RATIO = 0.80
    # Hourly: the dew point changes over hours, not minutes (real data: a
    # median 0.3 K in 3 hours).
    STEP_HOURS = 1.0
    # Fitted and scored on what a plan needs: free runs over its day ahead.
    HORIZON_STEPS = 24
    SCORED_HOURS = (6, 12, 24)

    # From an hour - a house with every window open - to six weeks, beyond
    # which the indoor air would not follow the weather at all; the fit on
    # real data sits at a day or two.
    MIN_TIME_CONSTANT_HOURS = 1.0
    MAX_TIME_CONSTANT_HOURS = 1000.0
    # Occupants only add moisture; 3 kPa above the outdoor air would put the
    # indoor dew point some 15 K above it.
    MAX_MOISTURE_SURPLUS_PA = 3000.0

    @property
    def name(self) -> str:
        return "dew_point"

    @property
    def label(self) -> str:
        return "Dew point"

    @property
    def unit(self) -> str:
        return "°C"

    def dataset(self, config: Config) -> DatasetDefinition:
        sensors = config.building.dew_points

        if not sensors:
            raise ValueError("No indoor dew point configured (building.dew_points).")

        builder = DatasetBuilder()

        # Dew points are states that report on change: the hour's mean, from
        # the last reading where an hour has none.
        for i, sensor in enumerate(sensors):
            builder = builder.timeseries(
                f"dew_point_{i}",
                sensor,
                interval="1h",
                aggregation="mean",
                fill="previous",
            )

        # Every forecast Open-Meteo published, matched on the hour it is for:
        # prepare() keeps the one known by then. The dew point where it was
        # stored, and the humidity it follows from before that.
        builder = builder.attribute_timeseries(
            "open_meteo",
            config.forecast.open_meteo,
            attributes=["temperature", "dew_point", "relative_humidity"],
            interval="1h",
            aggregation="last",
        )

        for i in range(1, len(sensors)):
            builder = builder.join(
                left="dew_point_0", right=f"dew_point_{i}", on=("time",), how="left"
            )

        return builder.join(
            left="dew_point_0",
            right="open_meteo",
            left_on=("time",),
            right_on=("target_time",),
            how="left",
        ).build()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hourly indoor and outdoor dew point (deg C), on an unbroken hourly
        index so that consecutive rows are consecutive hours.

        Indoors, the highest of the configured rooms: the air most likely to
        condense. Outdoors, the latest forecast published by that hour - the
        best the outdoor air was known then - as its stored dew point, or from
        its temperature and humidity where none was stored yet.
        """

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)

        if "time_open_meteo" in df.columns:
            published = pd.to_datetime(df["time_open_meteo"], utc=True)
            df = df[published.isna() | (published <= df["time"])]
            df = df.assign(published=published).sort_values(["time", "published"])

        hourly = df.groupby("time").last()

        def column(name: str) -> pd.Series:
            if name not in hourly.columns:
                return pd.Series(np.nan, index=hourly.index)

            return pd.to_numeric(hourly[name], errors="coerce")

        inside = hourly[[c for c in hourly.columns if c.startswith("dew_point_")]]
        outside = column("dew_point").fillna(
            dew_point_from_humidity_c(
                column("temperature"), column("relative_humidity")
            )
        )
        prepared = pd.DataFrame(
            {
                "inside": inside.apply(pd.to_numeric, errors="coerce").max(axis=1),
                "outside": outside,
            }
        )

        return prepared.asfreq(f"{int(self.STEP_HOURS)}h")

    def _starts(self, data: pd.DataFrame) -> np.ndarray:
        """Hours a full horizon can be scored from: it and the horizon after
        it all measured indoors and forecast outdoors."""

        known = data["inside"].notna() & data["outside"].notna()
        complete = (
            known[::-1]
            .rolling(self.HORIZON_STEPS + 1, min_periods=self.HORIZON_STEPS + 1)
            .sum()[::-1]
            == self.HORIZON_STEPS + 1
        )

        return np.flatnonzero(complete.to_numpy())

    def _runs(
        self, model: DewPointModel, data: pd.DataFrame, starts: np.ndarray
    ) -> np.ndarray:
        """Free runs from every start, one row per start, one column per hour
        ahead (1 ... HORIZON_STEPS)."""

        window = starts[:, None] + np.arange(self.HORIZON_STEPS + 1)

        return indoor_dew_point_c(
            model,
            data["inside"].to_numpy()[starts],
            data["outside"].to_numpy()[window],
            self.STEP_HOURS,
        )[:, 1:]

    def _measured(self, data: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
        inside = data["inside"].to_numpy()

        return np.column_stack(
            [inside[starts + h] for h in range(1, self.HORIZON_STEPS + 1)]
        )

    def _split(self, data: pd.DataFrame) -> int:
        return int(len(data) * self.TRAIN_RATIO)

    def calibrate(self, df: pd.DataFrame) -> DewPointModel:
        data = self.prepare(df)
        split = self._split(data)
        starts = self._starts(data)
        # Training windows end before the split, so no hour held out for
        # validation is fitted on.
        train = starts[starts + self.HORIZON_STEPS < split]

        if len(train) < self.HORIZON_STEPS:
            raise ValueError(
                f"Dew point needs at least {self.HORIZON_STEPS} complete training "
                f"windows, found {len(train)}."
            )

        measured = self._measured(data, train)

        def residuals(parameters: np.ndarray) -> np.ndarray:
            model = DewPointModel(*parameters)
            return (self._runs(model, data, train) - measured).ravel()

        result = least_squares(
            residuals,
            x0=np.array([24.0, 100.0]),
            bounds=(
                [self.MIN_TIME_CONSTANT_HOURS, 0.0],
                [self.MAX_TIME_CONSTANT_HOURS, self.MAX_MOISTURE_SURPLUS_PA],
            ),
        )
        self.model = DewPointModel(*(float(x) for x in result.x))

        logger.info(
            "Dew point calibrated on %d windows: time constant %.1f h, moisture "
            "surplus %.0f Pa",
            len(train),
            self.model.time_constant_hours,
            self.model.moisture_surplus_pa,
        )

        return self.model

    def validate(self, df: pd.DataFrame) -> dict[str, float]:
        """Free runs from every hour held out, scored against the dew point
        measured then, beside holding the one measured at the start."""

        model = self.get_model()
        data = self.prepare(df)
        starts = self._starts(data)
        test = starts[starts >= self._split(data)]

        if not len(test):
            raise ValueError("No complete window held out to validate on.")

        measured = self._measured(data, test)
        error = self._runs(model, data, test) - measured
        held = data["inside"].to_numpy()[test][:, None] - measured
        metrics: dict[str, float] = {"windows": float(len(test))}

        for hours in self.SCORED_HOURS:
            h = int(hours / self.STEP_HOURS) - 1
            metrics[f"rmse_{hours}h"] = float(np.sqrt(np.mean(error[:, h] ** 2)))
            metrics[f"persistence_rmse_{hours}h"] = float(
                np.sqrt(np.mean(held[:, h] ** 2))
            )

        logger.info(
            "Dew point validation over %d windows: RMSE %s, holding it %s",
            len(test),
            ", ".join(
                f"{h} h {metrics[f'rmse_{h}h']:.2f} K" for h in self.SCORED_HOURS
            ),
            ", ".join(
                f"{h} h {metrics[f'persistence_rmse_{h}h']:.2f} K"
                for h in self.SCORED_HOURS
            ),
        )

        return metrics
