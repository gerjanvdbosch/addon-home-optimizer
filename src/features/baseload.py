import numpy as np
import pandas as pd

from domain.config import Config
from domain.dataset import DatasetDefinition
from domain.time import LOCAL_TIMEZONE
from features.dataset import DatasetBuilder


class BaseloadForecaster:
    """The household's baseload (W) ahead as the habit of its week: per local
    quarter hour, the median of the same quarter hour on the same weekday over
    the past MEDIAN_WEEKS, and the band (BAND_QUANTILES, the MPC's low and high
    baseload scenarios) of that weekday's hour over the past BAND_WEEKS.

    Baseload is a ~160 W base with appliance spikes whose timing no forecast
    can know; when they come follows the household's week. Walk-forward on
    real data (each day 15 Sep - 27 Oct 2026 from the weeks before it), the
    median came out 126.8 W off against 130.8 W for gradient boosting on the
    calendar and the same quarter yesterday and last week (refitted weekly),
    better in 6 of 7 weeks and on every weekday but Monday (+1.2 W); a day's
    energy 1.8 against 2.3 kWh off. Over 6 weeks 127.7 W, 4 weeks 132.2 W,
    per hour instead of quarter hour 128.2 W. The band's pinball loss came out
    17.0 W (p10) and 53.2 W (p90) against 17.9 and 52.7 W for the boosting
    median plus its own errors' quantiles per weekday and hour over 12 weeks;
    13.7% below p10 and 12.1% above p90. Per hour, as a quarter hour's 12
    values are too few for a tail. Recent weeks weighted more did not help.
    """

    MEDIAN_WEEKS = 8
    BAND_WEEKS = 12
    BAND_QUANTILES = (0.1, 0.9)
    QUARTERS_PER_HOUR = 4
    QUARTERS_PER_WEEK = 7 * 24 * QUARTERS_PER_HOUR

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_baseload",
                config.baseload,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .build()
        )

    def habit(self, df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
        """(p10, p50, p90) per quarter hour of the local week (see _slot), from
        the whole weeks before the last local midnight in `df`: today adds
        one day of eight, so it is made once a day."""

        y = pd.Series(
            df["P_baseload"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df["time"]).tz_convert(LOCAL_TIMEZONE),
        ).sort_index()
        y = y[y.notna()]
        midnight = pd.Timestamp(y.index[-1]).normalize()
        low, high = self.BAND_QUANTILES

        return (
            self._quantile(y, midnight, self.BAND_WEEKS, low, self.QUARTERS_PER_HOUR),
            self._quantile(y, midnight, self.MEDIAN_WEEKS, 0.5, 1),
            self._quantile(y, midnight, self.BAND_WEEKS, high, self.QUARTERS_PER_HOUR),
        )

    def predict(
        self,
        habit: tuple[list[float], list[float], list[float]],
        times: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """The habit at `times`, as the predictions baseload_p10, baseload and
        baseload_p90. The band holds the median: the quarter hour's own median
        can fall outside its hour's band (0.2% of quarter hours on real
        data)."""

        slot = self._slot(times.tz_convert(LOCAL_TIMEZONE))
        p10, p50, p90 = (np.asarray(quantile)[slot] for quantile in habit)

        return pd.DataFrame(
            {
                "baseload_p10": np.clip(np.minimum(p10, p50), 0.0, None),
                "baseload": p50,
                "baseload_p90": np.maximum(p90, p50),
            },
            index=times,
        )

    def _quantile(
        self,
        y: pd.Series,
        midnight: pd.Timestamp,
        weeks: int,
        quantile: float,
        pooled_quarters: int,
    ) -> list[float]:
        # Wall-clock weeks, so each quarter hour of the week has `weeks` values
        # across a DST change too.
        recent = y.loc[
            (y.index >= midnight - pd.DateOffset(weeks=weeks)) & (y.index < midnight)
        ]
        groups = self.QUARTERS_PER_WEEK // pooled_quarters
        table = (
            recent.groupby(
                self._slot(pd.DatetimeIndex(recent.index)) // pooled_quarters
            )
            .quantile(quantile)
            .reindex(range(groups))
            # A slot without any value yet (a short history): the week's.
            .fillna(recent.quantile(quantile))
        )

        return np.repeat(table.to_numpy(), pooled_quarters).tolist()

    def _slot(self, index: pd.DatetimeIndex) -> np.ndarray:
        """The quarter hour of the local week, Monday 00:00 = 0."""

        return np.asarray(
            (index.day_of_week * 24 + index.hour) * self.QUARTERS_PER_HOUR
            + index.minute // 15
        )
