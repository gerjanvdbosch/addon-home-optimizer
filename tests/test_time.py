import time
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from domain.time import local_day_start, to_local_time


@pytest.fixture
def amsterdam_time(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Amsterdam")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_local_day_start_is_local_midnight_not_utc_midnight(amsterdam_time):
    """The dashboard's day starts at local midnight - in summer that is 22:00
    UTC the evening before, not 00:00 UTC (which would drop two local hours)."""

    now = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)

    start = local_day_start(now)

    assert start.astimezone(UTC) == datetime(2026, 9, 13, 22, 0, tzinfo=UTC)
    assert local_day_start(now, days=2).astimezone(UTC) == datetime(
        2026, 9, 15, 22, 0, tzinfo=UTC
    )


def test_local_day_start_uses_the_offset_valid_at_each_midnight(amsterdam_time):
    """Across the end of summer time (25 October 2026) the two midnights have
    different UTC offsets, so that day really is 25 hours long."""

    now = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)

    start = local_day_start(now)
    next_start = local_day_start(now, days=1)

    assert start.utcoffset() == timedelta(hours=2)
    assert next_start.utcoffset() == timedelta(hours=1)
    assert next_start - start == timedelta(hours=25)


def test_to_local_time_uses_the_offset_valid_at_that_instant(amsterdam_time):
    summer = to_local_time(datetime(2026, 7, 1, 12, 0, tzinfo=UTC))
    winter = to_local_time(datetime(2026, 12, 1, 12, 0, tzinfo=UTC))

    assert summer.utcoffset() == timedelta(hours=2)
    assert winter.utcoffset() == timedelta(hours=1)


def test_to_local_time_accepts_pandas_timestamps(amsterdam_time):
    """Schedule times reach to_local_time() as pandas Timestamps (see
    StateManager.resolve_schedule) - they must convert the same way."""

    local = to_local_time(pd.Timestamp("2026-07-01T12:00:00Z"))

    assert local == datetime(2026, 7, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
