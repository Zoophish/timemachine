import polars as pl
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Set, Tuple
import sys
import random


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


class DTSlidingWindowDataset(torch.utils.data.Dataset):
    """
    Dataframe requirements
        - Regular timestamp intervals
        - Sorted by timestamp
        - Unique timestamps
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

        print("Generating centers...")
        self.centers = []

        # unfortunately, we must test every single centre to make sure it produces valid contiguous windows
        # this is a slightly cumbersome but necessary process
        timestamps = self.df[self.ts_col].to_numpy()

        stride = int(self.stride / timedelta(minutes=1))
        i = self.lookback_steps
        while i < len(timestamps) - self.horizon_steps:
            lookback_delta = timestamps[i] - timestamps[i - self.lookback_steps]
            horizon_delta = timestamps[i + self.horizon_steps] - timestamps[i]
            # if the horizon skipped ahead, advance the center (i) to the next valid chunk
            # this is to avoid skipping things with the stride
            if horizon_delta > self.horizon:  
                while (timestamps[i] - timestamps[i - self.lookback_steps]) != self.lookback:
                    i += 1
            # center has contiguous windows on either side
            if lookback_delta == self.lookback and horizon_delta == self.horizon:
                self.centers.append(self.df.row(i, named=True)[self.ts_col])

            i += stride

            # print progress
            last_j = -1
            j = 100 * i // len(timestamps)
            if j != last_j:
                sys.stdout.write(f'\r[{"█" * (j//2)}{"░" * (50-j//2)}] {j}%')
                sys.stdout.flush()
        print(f"Finished. ({len(self.centers)}) centers.")

        # thread this
        # for n, dt in enumerate(center_dts):
        #     has_lookback = self.df.filter(
        #         (pl.col(self.ts_col) >= (dt - self.lookback)) &
        #         (pl.col(self.ts_col) < dt)
        #     ).height == self.lookback_steps
        #     has_lookahead = self.df.filter(
        #         (pl.col(self.ts_col) >= dt) &
        #         (pl.col(self.ts_col) < (dt + self.horizon))
        #     ).height == self.horizon_steps
        #     if has_lookback and has_lookahead:
        #         self.centers.append(dt)
        #     i = n // len(center_dts)
        #     sys.stdout.write(f'\r[{"█" * (i//2)}{"░" * (50-i//2)}] {i}%')
        #     sys.stdout.flush()


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


class SlidingWindowDataset(torch.utils.data.Dataset):
    """
    Non-datetime indexed sliding window dataset.
    """
    def __init__(self, time_series, lookback_steps, forecast_steps, sample_mode='random', **kwargs):
        self.lookback_steps = lookback_steps
        self.forecast_steps = forecast_steps
        self.sample_mode = sample_mode
        self.n_samples = kwargs.get('n_samples')
        self.time_series = torch.tensor(time_series, dtype=torch.float32)
        self.indices = self._create_indices()
        self.remainder = 0

    def _create_indices(self):
        indices = []
        match self.sample_mode:
            case 'segments':
                i = self.lookback_steps
                while i <= len(self.time_series) - self.forecast_steps:
                    indices.append(i)
                    i += self.lookback_steps + self.forecast_steps
                self.remainder = len(self.time_series) - 1 - i
            case 'complete-output-coverage':
                i = self.lookback_steps
                while i <= len(self.time_series) - self.forecast_steps:
                    indices.append(i)
                    i += self.forecast_steps
                self.remainder = len(self.time_series) - 1 - i
            case 'random':
                indices = [random.randint(
                    self.lookback_steps,
                    len(self.time_series) - self.forecast_steps
                ) for _ in range(self.n_samples)]
                self.remainder = self.forecast_steps
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        forecast_start_idx = self.indices[idx]
        lookback_start_idx = forecast_start_idx - self.lookback_steps
        x = self.time_series[lookback_start_idx:forecast_start_idx]
        y = self.time_series[forecast_start_idx:forecast_start_idx + self.forecast_steps]
        if x.dim() == y.dim() == 1:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
        return x, y
