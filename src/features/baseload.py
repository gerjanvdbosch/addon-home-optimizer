import numpy as np
import pandas as pd

from domain.config import Config
from domain.dataset import DatasetDefinition
from domain.time import LOCAL_TIMEZONE
from features.dataset import DatasetBuilder


class BaseloadForecaster:
    """The household's baseload (W) ahead as the habit of its week: per local
    quarter hour, the mean of the same quarter hour on the same weekday over
    the past MEAN_WEEKS, and the band (BAND_QUANTILES, the MPC's low and high
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

    The mean, not that median: what a plan weighs is energy - the house's
    draw against the sun's, the heat it leaves in the rooms - and a quarter
    hour's energy ahead is its mean. The median leaves out the cooker's and
    the washing's peaks, on in some weeks and not in others: walk-forward
    (each day 1 Aug - 9 Nov 2026 from the weeks before it) it came out 59 W
    low, 142 W from 17 to 21 h, and a day's energy 1.60 kWh off, against 5
    and 26 W low and 1.12 kWh for the mean (per quarter hour 135 against
    118 W off: it spreads a peak no forecast can time). The zone 12 h ahead,
    its floor's heat and all else as measured, in the heating season (from
    10 Oct): the living room 0.103 against 0.108 K off and -0.04 against
    -0.07 K low, 0.100 and -0.02 K with the baseload as measured. Per hour or
    over 4 weeks instead did no better.
    """

    MEAN_WEEKS = 8
    BAND_WEEKS = 12
    BAND_QUANTILES = (0.1, 0.9)
    QUARTERS_PER_HOUR = 4
    QUARTERS_PER_WEEK = 7 * 24 * QUARTERS_PER_HOUR

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .sum_series("P_baseload", config.baseload_terms, interval="15m")
            .build()
        )

    def habit(self, df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
        """(p10, mean, p90) per quarter hour of the local week (see _slot), from
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
            self._habit(y, midnight, self.BAND_WEEKS, low, self.QUARTERS_PER_HOUR),
            self._habit(y, midnight, self.MEAN_WEEKS, None, 1),
            self._habit(y, midnight, self.BAND_WEEKS, high, self.QUARTERS_PER_HOUR),
        )

    def predict(
        self,
        habit: tuple[list[float], list[float], list[float]],
        times: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """The habit at `times`, as the predictions baseload_p10, baseload and
        baseload_p90. The band holds the mean: a quarter hour's own mean can
        fall outside its hour's band, a peak in one week of eight."""

        slot = self._slot(times.tz_convert(LOCAL_TIMEZONE))
        p10, mean, p90 = (np.asarray(values)[slot] for values in habit)

        return pd.DataFrame(
            {
                "baseload_p10": np.clip(np.minimum(p10, mean), 0.0, None),
                "baseload": mean,
                "baseload_p90": np.maximum(p90, mean),
            },
            index=times,
        )

    def _habit(
        self,
        y: pd.Series,
        midnight: pd.Timestamp,
        weeks: int,
        quantile: float | None,
        pooled_quarters: int,
    ) -> list[float]:
        """A quantile of each slot's values, or without one their mean."""

        def of(values):
            return values.mean() if quantile is None else values.quantile(quantile)

        # Wall-clock weeks, so each quarter hour of the week has `weeks` values
        # across a DST change too.
        recent = y.loc[
            (y.index >= midnight - pd.DateOffset(weeks=weeks)) & (y.index < midnight)
        ]
        groups = self.QUARTERS_PER_WEEK // pooled_quarters
        table = (
            of(
                recent.groupby(
                    self._slot(pd.DatetimeIndex(recent.index)) // pooled_quarters
                )
            )
            .reindex(range(groups))
            # A slot without any value yet (a short history): the week's.
            .fillna(of(recent))
        )

        return np.repeat(table.to_numpy(), pooled_quarters).tolist()

    def _slot(self, index: pd.DatetimeIndex) -> np.ndarray:
        """The quarter hour of the local week, Monday 00:00 = 0."""

        return np.asarray(
            (index.day_of_week * 24 + index.hour) * self.QUARTERS_PER_HOUR
            + index.minute // 15
        )
