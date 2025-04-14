import polars as pl
import numpy as np
from datetime import datetime, timedelta


def gen_window_targets_by_timedelta(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    input_scaler: object,
    output_scaler: object,
    lookback: timedelta,
    horizon: timedelta,
    time_col: str = "timestamp",
):
    """
    Generate sliding windows of input and target data for a time series dataframe.
    Args:
        df (pl.DataFrame): DataFrame containing the time series data with a 'timestamp' column.
        feature_cols (list[str]): List of feature column names to be used as input.
        target_col (str): Name of the target column to be predicted.
        input_scaler (object): Scaler object for input features.
        output_scaler (object): Scaler object for output target.
        lookback (timedelta): Time duration to look back for input features.
        horizon (timedelta): Time duration to look forward for target prediction.
        time_col (str): Name of the column containing timestamps. Default is 'timestamp'.
    Returns:
        tuple: Tuple containing:
            - input_windows (list): List of input feature windows.
            - output_windows (list): List of target windows.
            - input_indices (list): List of indices for input windows in the original DataFrame.
            - output_indices (list): List of indices for target windows in the original DataFrame.
    """
    input_windows = []
    output_windows = []

    input_indices = []
    output_indices = []

    start_date = df[time_col].dt.replace_time_zone(None).min()
    end_date = df[time_col].dt.replace_time_zone(None).max()
    step_date = start_date
    step_rate = lookback
    df = df.with_row_index()
    while step_date <= (end_date - step_rate):
        step_date += step_rate
        if (step_date - lookback) < start_date:
            continue
        if (step_date + horizon) > end_date:
            continue

        # get the lookback
        lookback_df = df.filter(
            (pl.col(time_col).dt.replace_time_zone(None) > (step_date - lookback))
            & (pl.col(time_col).dt.replace_time_zone(None) <= step_date)
        )
        target_df = df.filter(
            (pl.col(time_col).dt.replace_time_zone(None) >= step_date)
            & (pl.col(time_col).dt.replace_time_zone(None) < (step_date + horizon))
        )
        # handle dataframes with gaps
        if lookback_df.height == 0 or target_df.height == 0:
            continue

        # record the indices in the full buffer where these windows are
        lookback_start_idx = lookback_df.select(pl.min('index')).item()
        lookback_end_idx = lookback_df.select(pl.max('index')).item() + 1
        target_start_idx = target_df.select(pl.min('index')).item()
        target_end_idx = target_df.select(pl.max('index')).item() + 1

        # apply scaling
        input_features = input_scaler.transform(lookback_df[feature_cols].to_numpy())
        output = output_scaler.transform(target_df[target_col].reshape((-1, 1)).to_numpy())

        # create the input/target pair
        input_windows.append(input_features)
        output_windows.append(output)

        input_indices.append(np.array((lookback_start_idx, lookback_end_idx)))
        output_indices.append(np.array((target_start_idx, target_end_idx)))
    
    return (
        input_windows,
        output_windows,
        input_indices,
        output_indices
    )


def gen_window_targets_by_index(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    input_scaler: object,
    output_scaler: object,
    lookback: int,
    horizon: int,
):
    input_windows = []
    output_windows = []

    input_indices = []
    output_indices = []

    centre = 0
    step_rate = lookback
    end_index = df.height - 1
    df = df.with_row_index()
    while centre <= (end_index - step_rate):
        centre += step_rate

        # record the indices in the full buffer where these windows are
        lookback_start_idx = centre - lookback
        lookback_end_idx = centre
        target_start_idx = centre
        target_end_idx = centre + horizon

        lookback_df = df[feature_cols].slice(lookback_start_idx, lookback)
        target_df = df[target_col].slice(target_start_idx, horizon)

        # apply scaling
        input_features = input_scaler.transform(lookback_df.to_numpy())
        output = output_scaler.transform(target_df.reshape((-1, 1)).to_numpy())

        # create the input/target pair
        input_windows.append(input_features)
        output_windows.append(output)

        input_indices.append(np.array((lookback_start_idx, lookback_end_idx)))
        output_indices.append(np.array((target_start_idx, target_end_idx)))
    
    return (
        input_windows,
        output_windows,
        input_indices,
        output_indices
    )



def spliti(iterable, ratio, split_gap=0):
    """
        Split the iterable into two proportionally to ratio. Optionally ensure
        an aboslute gap ahead of the split point to prevent window overlapping.
    
    Args:
        iterable (Iterable): The iterable to split
        ratio (float): The split ratio
        split_gap (int): The index gap between the two splits
    
    Returns:
        (Iterable, Iterable): The split segments
    """
    return (iterable[:int(len(iterable)*ratio) - 1], iterable[int(len(iterable)*ratio) - 1 + split_gap:])
