import numpy as np
import polars as pl
from datetime import timedelta


def cyclical_calendar_encoding(
    df: pl.DataFrame, 
    datetime_col: str, 
    period: str,
    step : str = '1ms'
) -> pl.DataFrame:
    """
    Creates a cyclical encoding over a given calendar period (year, month, week, day) in a given step size (1m, 1h, 1d, 1w, 1mo, 1y, etc).
    
    Args:
        df (pl.DataFrame): Input DataFrame containing the datetime column.
        datetime_col (str): Name of the datetime column to encode.
        period (str): Calendar period to encode over ('year', 'month', 'week', 'day').
        step (str): Step size for truncating the datetime (default: '1ms').

    Returns:
        pl.DataFrame: DataFrame with cyclical encoding columns (cosine and sine for the specified period).
    """
    
    df = df.with_columns(pl.col(datetime_col).cast(pl.Datetime))
    
    if period == "day":
        period_start = pl.col(datetime_col).dt.truncate(every='1d').dt.timestamp(time_unit='ms')
        max_period = timedelta(days=1).total_seconds()
    elif period == "week":
        period_start = pl.col(datetime_col).dt.truncate(every='1w').dt.timestamp(time_unit='ms')
        max_period = timedelta(weeks=1).total_seconds()
    elif period == "month":
        period_start = pl.col(datetime_col).dt.truncate(every='1mo').dt.timestamp(time_unit='ms')
        max_period = ((pl.col(datetime_col).dt.month_end() - pl.col(datetime_col).dt.month_start()).dt.total_days() + 1) * 24*60*60
    elif period == "year":
        period_start = pl.col(datetime_col).dt.truncate(every='1y').dt.timestamp(time_unit='ms')
        max_period = pl.when(pl.col(datetime_col).dt.is_leap_year()).then(timedelta(days=366).total_seconds()).otherwise(timedelta(days=365).total_seconds())
    
    df = df.with_columns(
        timestamp=pl.col(datetime_col).dt.truncate(every=step).dt.timestamp(time_unit='ms') - period_start
    )
    max_period *= 1e3  # convert to ms

    df = df.with_columns([
        (2 * np.pi * (pl.col('timestamp') / max_period)).cos().alias(f"{period}_cos"),
        (2 * np.pi * (pl.col('timestamp') / max_period)).sin().alias(f"{period}_sin")
    ])

    return df



def holiday_embeddings():
    ...