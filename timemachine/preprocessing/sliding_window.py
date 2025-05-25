import polars as pl
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Set, Tuple
from itertools import islice



def make_window_target_pairs(
        df : pl.DataFrame,
        centers : List[datetime],
        timestamp_col : str,
        lookback : timedelta,
        lookahead : timedelta,
        lookback_cols : Set[str],
        lookahead_cols : Set[str]) -> Tuple[List, List]:
    """
    Generates the lookback and target windows around each center from the
    time series data frame.

    Lookback is exclusive of the center; lookahead is inclusive.

    Args:
        df (pl.DataFrame): The time series dataframe
        centers (List[datetime]): List of center datetimes to base the windows around
        timestamp_col (str): The time stamp column name
        lookback (timedelta): Lookback window from center in time units (exclusive)
        lookahead (timedelta): Lookahead window from center in time units (inclusive)
        lookback_cols (Set[str]): Column names of lookback window variables
        lookahead_cols (Set[str]): Column names of lookahead window variables
    Returns:
        Tuple[list, list]: The lookback and respective lookahead window lists.
    """
    lookback_windows = []
    target_windows = []

    min_dt = df[timestamp_col].min()
    max_dt = df[timestamp_col].max()

    for center_dt in centers:
        # skip if this will attempt access outside the dataset range
        if (center_dt - lookback) < min_dt:
            continue
        if (center_dt + lookahead) > max_dt:
            continue
        
        # lookback exclusive of center
        lookback_df = df.filter(
            (pl.col(timestamp_col).dt.datetime() >= (center_dt - lookback)) &
            (pl.col(timestamp_col).dt.date() < center_dt)
        )
        # lookahead inclusive of center
        lookahead_df = df.filter(
            (pl.col(timestamp_col).dt.datetime() < (center_dt + lookahead)) &
            (pl.col(timestamp_col).dt.date() >= center_dt)
        )

        lookback_windows.append(lookback_df[lookback_cols].to_numpy())
        target_windows.append(lookahead_df[lookahead_cols].to_numpy())

    return lookback_windows, target_windows


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


def windows_to_batch(windows : List[np.ndarray]) -> torch.Tensor:
    """
    Utility to pack windows into a batch tensor.

    Args:
        windows (List[np.ndarray]): The list of windows.
    Returns:
        torch.Tensor
        The batch of windows.
    """
    batch_size = len(windows)
    item_size = windows[0].shape
    batch = torch.zeros(size=(batch_size, *item_size))
    for i in range(batch_size):
        batch[i] = windows[i]
    return batch


def make_sliding_window_batch(
    df : pl.DataFrame,
    centers : List[datetime],
    timestamp_col : str,
    lookback : timedelta,
    horizon : timedelta,
    lookback_cols : Set[str],
    target_cols : Set[str]) -> torch.Tensor:
    """
    Creates a batch tensor of sliding (lookback and target) windows around
    each center.
    """
    X_windows, y_windows = make_window_target_pairs(
        df=df,
        centers=centers,
        timestamp_col=timestamp_col,
        lookback=lookback,
        lookahead=horizon,
        lookback_cols=lookback_cols,
        lookahead_cols=target_cols
    )
    X_batch = windows_to_batch(X_windows)
    y_batch = windows_to_batch(y_windows)
    return X_batch, y_batch


def transform_batch(batch : torch.Tensor, transform):
    """
    Utility for applying a transformation to each batch item.

    Args:
        batch (torch.Tensor): A tensor of (batch_size, seq_len, n_channels)
        transform (func): A function that transforms (seq_len, n_channels)
    """
    batch = batch.numpy()
    for i in range(batch.shape[0]):
        batch[i] = transform(batch[i])
    return torch.tensor(batch)


def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be >= 1')
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class SlidingWindowDataset(torch.utils.data.Dataset):
    """
    """

    def __init__(
        self,
        df : pl.DataFrame,
        timestamp_col : str,
        lookback : timedelta,
        horizon : timedelta,
        stride : timedelta,
        lookback_steps : int,
        horizon_steps : int,
        lookback_cols : Set[str],
        target_cols : Set[str]
    ):
        self.df = df
        self.ts_col = timestamp_col
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps
        self.lookback_cols = lookback_cols
        self.target_cols = target_cols

        self._commit()


    def _commit(self):
        min_dt = self.df[self.ts_col].min()
        max_dt = self.df[self.ts_col].max()

        center_dts = pl.datetime_range(start=min_dt, end=max_dt, interval=self.stride, eager=True)
        self.centers = []

        # thread this
        for dt in center_dts:
            has_lookback = self.df.filter(
                (pl.col(self.ts_col) >= (dt - self.lookback)) &
                (pl.col(self.ts_col) < dt)
            ).height == self.lookback_steps
            has_lookahead = self.df.filter(
                (pl.col(self.ts_col) < (dt + self.horizon)) &
                (pl.col(self.ts_col) >= dt)
            ).height == self.horizon_steps
            if has_lookback and has_lookahead:
                self.centers.append(dt)


    def __len__(self):
        return len(self.centers)
    

    def __getitem__(self, index):
        center_dt = self.centers[index]
            
        # lookback exclusive of center
        lookback_df = self.df.filter(
            (pl.col(self.ts_col) >= (center_dt - self.lookback)) &
            (pl.col(self.ts_col) < center_dt)
        )
        # lookahead inclusive of center
        lookahead_df = self.df.filter(
            (pl.col(self.ts_col) < (center_dt + self.horizon)) &
            (pl.col(self.ts_col) >= center_dt)
        )

        lookback_window = torch.tensor(
            lookback_df[self.lookback_cols].to_numpy(),
            dtype=torch.float32
        )
        target_window = torch.tensor(
            lookahead_df[self.target_cols].to_numpy(),
            dtype=torch.float32
        )

        return lookback_window, target_window
