from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.optimization import Optimization
from domain.sensors import SensorReference
from domain.state import BaseloadHabit, State
from domain.time import LOCAL_TIMEZONE
from features.baseload import BaseloadForecaster

# A Wednesday morning, so the history reaches back across the DST change of
# 25 October.
NOW = pd.Timestamp("2026-10-28 06:00", tz="UTC")


def _history(weeks: int = 13) -> pd.DataFrame:
    """100 W, except Wednesday at 10 local: 500 W at 10:00-10:15 in the last
    8 weeks before today, 5000 W the whole hour in the 4 weeks before those,
    and 9999 W today."""

    time = pd.date_range(NOW - pd.Timedelta(weeks=weeks), NOW, freq="15min")
    local = time.tz_convert(LOCAL_TIMEZONE)
    midnight = pd.Timestamp(local[-1]).normalize()
    hour = (local.day_of_week == 2) & (local.hour == 10)
    recent = local >= midnight - pd.DateOffset(weeks=8)
    power = np.full(len(time), 100.0)
    power[hour & (local.minute == 0) & recent] = 500.0
    power[hour & ~recent] = 5000.0
    power[local >= midnight] = 9999.0
    return pd.DataFrame({"time": time, "P_baseload": power})


def _slot(day: str, hour: int, minute: int = 0) -> int:
    weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(day)
    return (weekday * 24 + hour) * 4 + minute // 15


@pytest.fixture(scope="module")
def habit() -> tuple[list[float], list[float], list[float]]:
    return BaseloadForecaster().habit(_history())


def test_the_median_takes_eight_weeks_the_band_twelve_and_never_today(habit):
    """The same quarter hour's median over the 8 weeks before midnight; the
    band from that weekday's hour over 12 weeks - here the older spikes too;
    today's readings in neither."""

    p10, p50, p90 = habit

    assert p50[_slot("wed", 10)] == 500.0
    assert p50[_slot("wed", 10, 15)] == 100.0
    assert p90[_slot("wed", 10, 30)] == 5000.0
    assert p90[_slot("tue", 10)] == 100.0
    assert p10[_slot("wed", 10)] == 100.0
    assert max(p50) < 9999.0 and max(p90) < 9999.0


def test_the_forecast_keeps_the_local_clock_across_a_dst_change():
    """Saturday 17:45 is the same local quarter hour in summer and winter
    time, an hour apart in UTC; the forecast keeps the times it was asked."""

    marked = [float(slot) for slot in range(BaseloadForecaster.QUARTERS_PER_WEEK)]
    times = pd.DatetimeIndex(["2026-10-24 15:45Z", "2026-10-31 16:45Z"])

    forecast = BaseloadForecaster().predict((marked, marked, marked), times)

    assert forecast.index.equals(times)
    assert (forecast["baseload"] == _slot("sat", 17, 45)).all()


def test_the_band_holds_the_median():
    """A quarter hour's median outside its hour's band widens the band to it."""

    size = BaseloadForecaster.QUARTERS_PER_WEEK
    times = pd.DatetimeIndex(["2026-10-28 09:00Z"])

    forecast = BaseloadForecaster().predict(
        ([300.0] * size, [200.0] * size, [150.0] * size), times
    )

    assert forecast.iloc[0].to_dict() == {
        "baseload_p10": 200.0,
        "baseload": 200.0,
        "baseload_p90": 200.0,
    }


def test_the_habit_is_made_once_a_day_and_forecast_every_plan():
    """Twelve weeks load once a day: today's habit is kept in the state, and
    each plan reads its forecast at its own times from it."""

    loads: list[datetime] = []

    def load(dataset, start, end):
        loads.append(end)
        return _history()

    saved: list[BaseloadHabit] = []
    optimization = Optimization(
        loader=SimpleNamespace(load=load),  # type: ignore[arg-type]
        state_manager=SimpleNamespace(  # type: ignore[arg-type]
            update_habit=lambda _, habit: saved.append(habit)
        ),
        config_repository=None,  # type: ignore[arg-type]
        models_path=Path("."),
        home_assistant=None,  # type: ignore[arg-type]
    )
    state = State()
    config = SimpleNamespace(baseload=SensorReference(entity_id="sensor.baseload"))
    now = NOW.to_pydatetime()
    later = datetime(2026, 10, 28, 11, tzinfo=timezone.utc)

    optimization._baseload(state, config, now, [now])  # type: ignore[arg-type]
    state.predictions.baseload_habit = saved[-1]
    # Wednesday 10:00 local is 09:00 UTC.
    forecast = optimization._baseload(  # type: ignore[arg-type]
        state, config, later, [datetime(2026, 10, 28, 9, tzinfo=timezone.utc)]
    )
    optimization._baseload(  # type: ignore[arg-type]
        state, config, now + pd.Timedelta(days=1), [now]
    )

    assert len(loads) == 2
    assert forecast["baseload"].iloc[0] == 500.0
