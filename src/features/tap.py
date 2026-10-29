from pathlib import Path

import numpy as np
import pandas as pd

from domain.config import Config, HeatPumpStates
from domain.dataset import DatasetDefinition
from domain.physics import MIXED_TAP_WATER_C, mains_water_temperature_c
from domain.time import LOCAL_TIMEZONE
from features.boiler import BoilerThermalIdentifier
from features.dataset import DatasetBuilder


class TapForecaster:
    """Forecasts hot-water tap demand as the expected excess-heat-loss power (W):
    the calibrated boiler thermal model's own residual diagnostic
    (BoilerThermalIdentifier.excess_loss_w) of the past weeks, averaged per
    quarter hour of the day.

    This does NOT predict a validated tap event or volume - no flow meter exists,
    and the underlying signal is the same unproven "candidate excess heat loss"
    used during calibration (see boiler.py). It is a forecast of that same
    residual quantity, useful as an exogenous input for MPC planning, not a claim of
    validated hot-water usage.

    Further, the target itself is a mix of two effects, not purely tap draws:
    real data shows the excess-loss flag rate is far more sensitive to the
    tank's own starting temperature (51.8% hot vs. 19.2% cool) than to presence
    (39.0% vs. 34.7%) - see BoilerThermalIdentifier.excess_loss_w's docstring
    for the underlying finding (a temperature-dependent heat-transfer effect
    the constant-UA model cannot represent). This forecaster therefore predicts
    "excess loss," a quantity influenced by both tap draws and that modeling
    gap - not tap draws alone.
    """

    # The planning model is linear, so its expected tank temperature follows from
    # the expected heat sink: this forecasts the mean, the past weeks' draws
    # per quarter hour of the day. Yesterday's evening says nothing of
    # today's (correlation -0.03 over Jun-Oct 2026), and the mean of the past
    # weeks follows a household's level as it changes. Walk-forward from 09:00
    # on real data (32 evenings from 15 Sep 2026, and 16 from 1 Oct), the
    # evening's draws (21-24 h) came out 0.72 and 0.83 kWh off, 0.14 short and
    # 0.18 over, against 1.06 and 1.16, 0.59 and 0.64 short, for gradient
    # boosting on the same quarter yesterday and the calendar, trained up to
    # then - which also forecast a draw at 13:30 for one the day before's
    # data made up. Two or eight weeks did about as well; weekends apart, no
    # better.
    PROFILE_DAYS = 28

    # What the forecast expects for the rest of today is held to what a heavy
    # day of the past week - one in four drew more - leaves after what has been
    # drawn since midnight: a household that showered early does not shower
    # again (real data, 16 Oct 2026: the evening's draws at 20:30-21:30, then
    # the usual 22:00 draw forecast on top, 2.3 kWh the tank never lost). Over
    # 136 forecasts issued at 18-22 h (13 Sep - 16 Oct 2026) the rest of the
    # day came out 1.38 kWh off against 1.52 for the forecast alone, 0.19 kWh
    # short on average; held to the week's mean day 1.34, but 0.53 short - the
    # heavier days' draws cut. A night's tank from 21:30 and 22:30 came out
    # 2.6 K off by the next run against 3.1 K, and on 16 Oct 1 K against 8.5 K
    # (88 nights, Sep-Oct). Only ever lowered: raised to the budget where the
    # forecast expected little, it put up to 41 times its own draw into a
    # quarter hour.
    BUDGET_DAYS = 7
    BUDGET_QUANTILE = 0.75

    QUARTERS_PER_DAY = 96

    # Averaged per kelvin the drawn water is heated above the mains (see
    # MIXED_TAP_WATER_C), and turned back into heat with the mains of the day
    # forecast: a household's habit is the water it uses, and the colder mains
    # of winter take more heat for the same showers.

    def __init__(self, models_path: Path) -> None:
        self.models_path = models_path
        # The configured state labels, handed to the boiler identifier this
        # reads the draws from; overwritten by dataset().
        self.states = HeatPumpStates()

    def predict(
        self,
        prepared: pd.DataFrame,
        steps: int = 48,
        habit: tuple[pd.Series, float | None] | None = None,
    ) -> pd.Series:
        """The draws ahead from `habit` (see habit()), made from `prepared`
        (see prepare()) without one - which then need only reach back to
        today's midnight."""

        profile, heavy_day_w = habit if habit is not None else self._habit(prepared)
        future_index = pd.date_range(
            start=prepared.index[-1], periods=steps + 1, freq="15min"
        )[1:]

        # The target is signed (noise, model error), but a tap draw can only
        # remove heat.
        forecast = pd.Series(
            profile.reindex(self._quarter_of_day(future_index)).fillna(0.0).to_numpy()
            * self._heated_k(future_index),
            index=future_index,
        ).clip(lower=0.0)

        return self._within_todays_budget(
            forecast, prepared["excess_loss_w"].dropna(), heavy_day_w
        )

    def habit(self, df: pd.DataFrame) -> tuple[pd.Series, float | None]:
        """(the draws per K above the mains per quarter hour of the day over
        the PROFILE_DAYS before today, a heavy day's draws over the BUDGET_DAYS
        before it or None without a full week): what predict() needs of the
        past weeks, which today hardly moves (the next day's draws came out
        0.861 kWh off made at midnight against 0.866 at each forecast, 179
        forecasts 8 Sep - 22 Oct 2026), so it can be made once a day."""

        return self._habit(self.prepare(df))

    def _habit(self, prepared: pd.DataFrame) -> tuple[pd.Series, float | None]:
        midnight = prepared.index[-1].normalize()
        past = prepared[prepared.index < midnight]
        y = past["draw_w_per_k"].dropna()
        recent = y[y.index >= midnight - pd.Timedelta(days=self.PROFILE_DAYS)]
        profile = (
            recent.groupby(self._quarter_of_day(recent.index))
            .mean()
            .reindex(range(self.QUARTERS_PER_DAY), fill_value=0.0)
        )
        drawn = past["excess_loss_w"].dropna()
        week_start = midnight - pd.Timedelta(days=self.BUDGET_DAYS)

        if drawn.empty or drawn.index[0] > week_start:
            return profile, None

        week = drawn[drawn.index >= week_start]

        return profile, float(
            np.quantile(week.groupby(week.index.date).sum(), self.BUDGET_QUANTILE)
        )

    @staticmethod
    def _quarter_of_day(index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.hour * 4 + index.minute // 15)

    @staticmethod
    def _heated_k(index: pd.DatetimeIndex) -> np.ndarray:
        """How far drawn water is heated above the mains at each time (K)."""

        return MIXED_TAP_WATER_C - np.array(
            [mains_water_temperature_c(day) for day in index.day_of_year]
        )

    @staticmethod
    def drawn_today_j(prepared: pd.DataFrame) -> float:
        """The heat drawn since the local midnight (J), from `prepared` (see
        prepare()): signed per quarter hour, so sensor noise averages out, and
        none while the tank heats - a draw then is not told from the heat put
        in."""

        y = prepared["excess_loss_w"].dropna()

        if y.empty:
            return 0.0

        quarter_s = 24 * 3600 / TapForecaster.QUARTERS_PER_DAY

        return max(float(y[y.index >= y.index[-1].normalize()].sum()) * quarter_s, 0.0)

    @staticmethod
    def _within_todays_budget(
        forecast: pd.Series, y: pd.Series, heavy_day_w: float | None
    ) -> pd.Series:
        """The forecast with the rest of today held to a heavy day's draws
        (see BUDGET_DAYS) less what today has drawn; as it is without one."""

        if y.empty or heavy_day_w is None:
            return forecast

        midnight = y.index[-1].normalize()
        left = max(heavy_day_w - y[y.index >= midnight].sum(), 0.0)
        today = forecast.index < midnight + pd.Timedelta(days=1)
        expected = forecast[today].sum()

        if expected > left:
            forecast = forecast.copy()
            forecast[today] *= left / expected

        return forecast

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        identifier = BoilerThermalIdentifier()
        identifier.states = self.states
        identifier.load(self.models_path)

        if identifier.model is None:
            raise RuntimeError(
                "The boiler thermal model must be calibrated (see "
                "BoilerThermalIdentifier) before the tap forecaster can be "
                "used, since it reads the draws from that model's own "
                "excess-heat-loss diagnostic."
            )

        prepared = identifier.prepare(df)
        excess_loss = identifier.excess_loss_w(prepared)

        # Each minute's loss is the heat over the interval that ends there,
        # summed into the quarter hour that interval began in. Taken from
        # quarter-hour means instead, InfluxDB's fill repeated a quarter's mean
        # into the next ones while the sensors did not change, and the tank
        # seemed to lose 1.1 kWh when they did (real data, 17 Oct 2026 13:30:
        # 4468 W after a run, the sensors 0.25 K down); against the minutes,
        # its quarters correlated 0.76, and its median 0.73.
        # A real draw during active heating is out of scope (confounded with
        # Q_in, same as the calibration diagnostics) - treated as a neutral zero
        # signal here rather than a gap.
        dt_seconds = prepared["dt_seconds"].to_numpy()[1:]
        began = pd.to_datetime(excess_loss["time"]) - pd.to_timedelta(
            dt_seconds, unit="s"
        )
        quarter = pd.Timedelta(minutes=15)
        heat_j = pd.Series(
            excess_loss["excess_loss_w"].fillna(0.0).to_numpy() * dt_seconds,
            index=began,
        )

        # Indexed in local time: a household's day follows the local clock.
        prepared = (
            (heat_j.resample(quarter).sum() / quarter.total_seconds())
            .to_frame("excess_loss_w")
            .asfreq(quarter)
            .tz_convert(LOCAL_TIMEZONE)
        )
        prepared["draw_w_per_k"] = prepared["excess_loss_w"] / self._heated_k(
            prepared.index
        )

        return prepared

    def dataset(self, config: Config) -> DatasetDefinition:
        self.states = config.heat_pump.states
        builder = (
            DatasetBuilder()
            .timeseries(
                "T_ambient",
                config.heat_pump.boiler.ambient_temperature,
                interval=BoilerThermalIdentifier.SAMPLE_INTERVAL,
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                interval=BoilerThermalIdentifier.SAMPLE_INTERVAL,
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                interval=BoilerThermalIdentifier.SAMPLE_INTERVAL,
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval=BoilerThermalIdentifier.SAMPLE_INTERVAL,
                aggregation="last",
                fill="previous",
            )
        )

        return builder.build()
