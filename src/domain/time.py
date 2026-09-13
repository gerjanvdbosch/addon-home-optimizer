from datetime import UTC, datetime

import pandas as pd


def parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def to_local_time(dt: datetime) -> datetime:
    return dt.astimezone(datetime.now().astimezone().tzinfo)


def to_local_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(
        datetime.now().astimezone().tzinfo
    )


def to_local_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(index, utc=True).tz_convert(
        datetime.now().astimezone().tzinfo
    )
