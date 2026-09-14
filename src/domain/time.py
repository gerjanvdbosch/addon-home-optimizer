from datetime import UTC, datetime, time, timedelta

import pandas as pd


def parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def to_local_time(dt: datetime) -> datetime:
    # astimezone() without an argument applies the system's local rules to this
    # instant, so each timestamp gets the UTC offset valid at that moment - a
    # fixed "current" offset would shift everything past a DST change by an hour.
    # Via to_pydatetime(): pandas Timestamps (also passed in here) do not accept
    # astimezone() without a target zone.
    return pd.Timestamp(dt).to_pydatetime().astimezone()


def local_day_start(dt: datetime, days: int = 0) -> datetime:
    """Local midnight of the day `days` after dt's local date, carrying the UTC
    offset valid at that midnight itself (so a day containing a DST change is
    23 or 25 hours long, as it really is)."""

    local_date = to_local_time(dt).date() + timedelta(days=days)

    return datetime.combine(local_date, time.min).astimezone()


def to_local_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(
        datetime.now().astimezone().tzinfo
    )


def to_local_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(index, utc=True).tz_convert(
        datetime.now().astimezone().tzinfo
    )
