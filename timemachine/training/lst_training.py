# ====== Stohastic Segments Training ======
# Organising the time series data in a way to encourage the discovery of better short term and long term dynamics, while
# also encouraging the model to use its cell and hidden states robustly.

#   When training LSTMs, a standard approach is to format the data into pairs of lookback windows and lookahead windows. Lookahead
# windows are often just a single timestep. The effect of this is that the model gets 1 piece of error information for each
# lookback window.
#   But what are we really trying to train the LSTM do do? Ideally, it will use whatever past values we give it as best it can to
# predict the next value. This requires an intimate understanding of the nature of the time series across a range of timescales and
# from arbitrary segments. A robust model should be able to learn:
#           - Short-term temporal dynamics
#           - Long-term temporal dynamics
#           - Interplay between short-term and long-term
#           - Starting from different initial points (learning how to manage initial cell/hidden states)
#           - Avoid memorising the original sequence (overfitting)
#
#   Although the model may be trained on a fixed window length, the model
# can 'unroll' as many times as we desire. If the model models general the temporal dynamics well, it should know how to do smaller
# unrollings as well as larger unrollings
# 
#   Fixed lookback windows could essentially encourage the model to become a lookup table, rather than fostering a temporal understanding of the data.
# Window predictions should be statefully connected - that is the cell/hidden states should be passed on. Otherwise, the model only learns how to use
# memory for a fixed input length, which might not generalise well for long term changes, and you might as well use an MLP or regression tree model that treats all inputs as lags.
# To do this, the model has to be trained using the passed-on states. Moreover, to encourage the model not to use its previous state as a lookup table,
# the model needs to learn how to start with initial states on different inputs.
#
#  Assumptions:
#   - The model architecture can stably learn on long sequences (i.e. no inherant artifacts as a result of using long inputs, e.g. exploding gradients)
#   - The model architecture is, at least in theory, able to universally represent non-linear patterns over any temporal duration, not just short windows
#   - Learning smaller sequences before large ones, or vice versa, is not better for learning, i.e. assume as little as possible over which parts of the data are easier to learn
#   - Noise is the information theory limit whereby temporal fluctuations are completely decoupled from any future value, i.e. a 'perfect temporal non-linear blackbox learner'
#     could not use that information to enhance its prediction.
#
# ---- How it works ----
#
#   1) Choose a lookback window size range and a connected segment proportion range.
#   For each epoch:
#   a) Stochastically draw segments of varing length, for various lookback window sizes.
#   b) For every lookback-n+1 pair sequence, partition it into randomly sized segments.
#   c) Pack these into batches for parallelisation and gradient smoothing.
#
# The method is meant to be stochastic. Each epoch, the majority of the full dataset should be 'covered' with a range of lookback windows and segments.
#   - Lookback windows are the number of unrolls per single prediction
#   - A segment is a continuous sequence of lookback windows (and predictions) which are statefully connected during training
#  
# The key idea is to train the model on varying window lengths and segment lengths and offsets in the original time series.
# This is done stochastically for practical purposes and to reduce overfitting. It should encourange the model to learn
# temporal patterns on all time intervals.
#
# The main technical complications are in creating batches:
#   - Segments must be processed serially between batches, so different segements can be processed in parallel.
#   - An end-of-segment (EOS) mask represents the segments that have terminated in a batch (so the states will be reset in that batch dimension).
#   - Segments are sorted by lookback length to reduce padding variation, which improves performance (and because PyTorch requires it)
#   - The final few batches will be smaller as the final few segments are processed (and PyTorch won't support zero length entries)
#
#
# Notes/Todo:
#   - Try and guide the models during training without introducing bias, for example:
#       - adding truely random noise and including the current timestep noise estimate in the output.
#
# - Curriculum learning - define 'simpler' things to produce, and gradually introduce complexity. Difficult to assess 'simple' on things like financial data.
#   - Being able to determine the simple->hard spectrum is effectively what enables learning as there needs to be a smooth path.
#   - How to make an effective learner for learning or adaptive teacher?
# - Provider richer backprop data - i.e. change the predictions. Separate important variables out like sign, magnitude, variance etc.
#   - Additionally make the model predict more symbols, not for the purposes of making signals, but to provide a richer backprop.
# - Snapshot training. Capture parameters at specific points in training and allow models to traverse different paths. Organise outcomes in tree.

import random
import numpy as np
import polars as pl
from typing import Iterable, Tuple, List, Set


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


def fixed_window_pairs(series : Iterable, lookback : int, lookahead : int = 1) -> Tuple[np.array, np.array]:
    """
    Generates pairs of lookback windows and next timesteps from the series.
    Each pair will be separated by the lookback size so that the lookback windows are effectively contiguous.

    Args:
        series (Iterable): The series data
        lookback (int): The lookback window size
        lookahead (int): The lookahead size (default 1, i.e. next time step)
    
    Returns:
        (np.array, np.array): Tuple of two arrays: (lookback windows, next time steps), where the pairs are at the same indices in each array.
    """
    lookback_sequences = []
    lookahead_sequences = []
    for i in range(0, len(series) - lookback - lookahead, lookback):
        lookback_sequences.append(series[i:i + lookback])
        lookahead_sequences.append(series[i + lookback:i+lookback+lookahead])
    return np.array(lookback_sequences), np.array(lookahead_sequences)


def fixed_window_pairs(
        in_series : Iterable,
        target_series : Iterable,
        lookback : int,
        next : bool = True
    ) -> Tuple[np.array, np.array]:
    """
    Generates pairs of lookback windows from the input data and next timesteps from the target data.
    Each pair will be separated by the lookback size so that the lookback windows are effectively contiguous.

    Args:
        in_series (Iterable): The input series data
        target_series (Iterable): The target series data
        lookback (int): The lookback window size
        next (bool): If True, the lookahead will be the next element in the target sequence, else it will be the corresponding target.
    
    Returns:
        (np.array, np.array): Tuple of two arrays: (lookback windows, next time steps), where the pairs are at the same indices in each array.
    """
    if len(in_series) != len(target_series):
        raise ValueError(f"In series and target series length must match. {len(in_series)} != {len(target_series)}.")
    
    lookahead = 1
    offset = 0 if next else -1
    lookback_sequences = []
    lookahead_sequences = []
    for i in range(0, len(in_series) - lookback - lookahead - offset, lookback):
        lookback_sequences.append(np.array(in_series[i:i + lookback]))
        lookahead_sequences.append(np.array(target_series[i + lookback + offset:i+lookback+lookahead+offset]))
    return np.array(lookback_sequences), np.array(lookahead_sequences)


def multiscale_window_pairs(in_series : Iterable, target_series : Iterable, lookbacks : Set[int], next : bool) -> dict:
    """
    Generates (in window, target t+1) pair sequences for every lookback in the range, or (in window, target t) if next is False.

    Args:
        in_series (Iterable): The input series data
        target_series (Iterable): The target series data
        lookback (Set[int]): The lookback window sizes
    
    Returns:
        Dict[int, Iterable]: The window pair array tuples for each lookback size
    """
    return {
        lb: fixed_window_pairs(in_series, target_series, lb, next)
        for lb in lookbacks
    }


def stochastic_segment(
        in_array : np.array,
        targets_array : np.array,
        l_mean : int,
        l_std : int
    ) -> List[Iterable]:
    """
    Stochastically partition the array into segments, based on the normal distribution.
    NOTE: If this runs many times, will the average segment position be skewed towards the end?

    Args:
        in_array (np.array): The input array
        targets_array (np.array): The target array
        l_mean (int): The mean segment length
        l_std (int): The segment length standard deviation
            
    Returns:
        List[Iterable]: Each segment (in order)
    """
    segments = []
    size = in_array.shape[0]
    i = 0
    while i < size:
        seg_len = int(abs(np.random.normal(l_mean, l_std)))
        segments.append((in_array[i: min(i + seg_len, size)], targets_array[i: min(i + seg_len, size)]))
        i += seg_len
    return segments


def segment_sample_normal(
        in_array : np.array,
        targets_array : np.array,
        l_mean : int,
        l_std : int,
    ) -> Iterable:
    """
    Stochastically draw a random segment from the array once, based on the normal distribution.

    Args:
        in_array (np.array): The input array
        targets_array (np.array): The targets array
        l_mean (int): The mean segment length
        l_std (int): The segment length standard deviation
            
    Returns:
        Iterable: The segment
    """
    size = in_array.shape[0]
    seg_len = max(1, int(abs(np.random.normal(l_mean, l_std))))
    i = int(np.random.random() * (size - seg_len))
    return in_array[min(i, size-1): min(i + seg_len, size)], targets_array[min(i, size-1): min(i + seg_len, size)]


def segment_sample_uniform(
        in_array : np.array,
        targets_array : np.array,
        l_min : int,
        l_max : int,
    ) -> Iterable:
    """
    Stochastically draw a random segment from the array once, based on the uniform distribution.

    Args:
        in_array (np.array): The input array
        targets_array (np.array): The targets array
        l_min (int): The minimum segment length
        l_max (int): The maximum segment length
            
    Returns:
        Iterable: The segment
    """
    size = in_array.shape[0]
    seg_len = int(np.random.uniform(l_min, l_max))
    i = int(np.random.random() * (size - seg_len))
    return in_array[min(i, size-1): min(i + seg_len, size)], targets_array[min(i, size-1): min(i + seg_len, size)]



def gen_batches(
        multiscale_segments,
        target_batch_size : int,
        shuffle_segments : bool
    ) -> Tuple[List[np.array], List[np.array], List[np.array], List[np.array], List[np.array]]:
    """
    Organises the segment items into batches, ideally for use in a single epoch.
    Additionally returns an end of segment (EOS) mask, indicating where segments reset between batches.

    The output batches **must be used in order** or this will not work.

    The sorting critera:
        - Consecutive items within a segment (i.e. window,n+1 pairs) line up between batches
        - Segment items of similar lookback size are grouped to reduce padding

    Note, the lookback size (input timesteps) for each batch will vary depending on the lookback range chosen.

    Sorting indices:
        - The batches and other elements will be returned unsorted.
        - The sorting indices give the sorting required to make the batch items ordered by lookback length (required for pytorch efficiency)
        - The hidden/cell states will thus need sorting by these indices to correctly match up to the previous batch.

    Args:
        multiscale_segments (Dict[int]): Dictionary containing for each lookback size, the segments (sequences of window,n+1 pairs)
        target_batch_size (int): The desired size for each batch. The final batch sizes will likely differ, so you can optionally cut these.
        shuffle_segments (bool): An experimental parameter. Ordering the segments will yield better performance, but I have no idea how this will affect
            generalisation.
            
    Returns:
        Tuple[List[np.array], List[np.array], List[np.array], List[np.array], List[np.array]]: The X batch, y batch, EOS mask, length list and sorting indices for each batch.
    """

    # This does the following:
    #   1) Sorts the flattened segments by lookback size to reduce padding in the time dimension
    #   2) Optionally shuffles the order of segments within each lookback size (without affecting the overall ordering of lookback sizes).
    #       This aims to improve generalisation by deordering the segments. It may significantly hinder batching performance though.
    #   3) Flattens the segments into a list of (lookback_size, segment) tuples
    sorted_lb_sizes = sorted(multiscale_segments.keys())
    flattened_segments = []
    for lb_size in reversed(sorted_lb_sizes):  # sorting largest to smallest lookbacks is critical for making the batches and later packing
        if shuffle_segments:
            segments = random.shuffle(multiscale_segments[lb_size])  # shuffle segment order
        else:
            segments = sorted(multiscale_segments[lb_size], key=lambda x : x[0].shape[0])  # sort by segment length
        for segment in segments:
            flattened_segments.append((lb_size, segment))

    # Now we build batches from the flattened segments:
    # Intuition helpers:
    #   - Each batch is of shape (batch_size, timesteps, features)
    #   - The only thing we can 'parallelise' into batches are independant segments. i.e. pairs from the same segment
    #           should not be in the same batch! Instead, they should be spread across batches.
    #   - Timesteps = maximum lookback size in whatever segments are in that batch

    batch_size = target_batch_size
    n_segments = len(flattened_segments)
    if batch_size > n_segments:
        print(f"Target batch size {batch_size} is too big for given segments. Reducing to {n_segments}")
        batch_size = n_segments

    n_features = flattened_segments[0][1][0].shape[2]
    n_outputs = flattened_segments[0][1][1].shape[2]

    # initialise trackers for active segments
    active_segments = flattened_segments[:batch_size]  # pop the first `batch_size` segments off
    flattened_segments = flattened_segments[batch_size:]
    pointers = [0] * batch_size  # pointer to track position within each segment

    # outputs
    X_batches = []
    y_batches = []
    eos_masks  = []  # end of segment indicators (consecutive True values between batches will be statefully connected)
    packing_lengths = []  # pytorch packing lengths (for omitting the padded regions from calculations)
    sortings = []  # sorting indices as lanes of batch may change to ensure it is always sorted by largest->smallest lookback
    
    while active_segments:  # continue until all segments are exhausted
        max_lookback = max(active_segments, key=lambda x : x[0])[0]  # get the largest lookback in this batch...
        X_batch = np.zeros(shape=(batch_size, max_lookback, n_features))  #... and use this as time axis (unset values will be padding)
        y_batch = np.zeros(shape=(batch_size, 1, n_outputs))
        eos_mask = np.full(shape=(batch_size,), dtype=np.int8, fill_value=0)
        packing_length = np.zeros(shape=(batch_size,), dtype=np.int64) # the length of each axis

        remove_lanes = np.full((batch_size), dtype=bool, fill_value=False)

        # collect the next step for each lane (fills a batch)
        for i in range(len(active_segments)):  # remember segments have two parts for the input and output data
            lb_size, segment = active_segments[i]
            window, next_timestep = segment[0][pointers[i]], segment[1][pointers[i]] # get the lookback,n+1 pair
            X_batch[i, :lb_size,  :] = window
            y_batch[i, :1,  :] = next_timestep
            packing_length[i] = lb_size  # pytorch will need to know how long the time axis is (it may be padded)
            if pointers[i] + 1 >= len(active_segments[i][1][0]):  # check if this is the last item in the segment
                eos_mask[i] = 0  # mark this lane as ending
                if flattened_segments:  # load a new segment, if there is one
                    active_segments[i] = flattened_segments.pop(0)  # because these are sorted, we are guarunteed to get max_lookback or smaller
                    pointers[i] = 0
                else:  # otherwise mark this lane as inactive and shrink the next batch
                    batch_size -= 1  # shrink the next one
                    remove_lanes[i] = True
            else:
                eos_mask[i] = 1  # segment remains connected
                pointers[i] += 1  # advance the pointer

        # get the indices to sort the batch by lookback size largest->smallest
        sorting_indices = np.argsort(-packing_length, kind='stable')  # stable, easier to debug
        # sorting_indices = np.argsort(packing_length)[::-1]  # slightly faster, unstable TODO: check this (should be fine)
        active_segments = [active_segments[i] for i in sorting_indices]
        pointers = [pointers[i] for i in sorting_indices]
        X_batch = X_batch[sorting_indices]
        y_batch = y_batch[sorting_indices]
        eos_mask = eos_mask[sorting_indices]
        packing_length = packing_length[sorting_indices]
        remove_lanes = remove_lanes[sorting_indices]

        # - We shrink the batch size as all segments are exhausted, so we don't have lanes of zeros
        # - the eos mask will map to the next lanes, after shrinkage if there is any
        for i, remove in reversed(tuple(enumerate(remove_lanes))):
            if remove:
                del active_segments[i]
                del pointers[i]

        X_batches.append(X_batch)
        y_batches.append(y_batch)
        eos_masks.append(eos_mask)
        packing_lengths.append(packing_length)
        sortings.append(sorting_indices)

    return X_batches, y_batches, eos_masks, packing_lengths, sortings



class LSTTrainer():
    """
    Brings everything together.
    """

    def __init__(
            self,
            window_sizes : set,
            X_data : np.ndarray,
            y_data : np.ndarray,
            batch_size : int,
            next_predictor : bool = False,
            segment_samples : int = 32,
            segment_strategy : str = 'fixed',
            seed=0 # this will be for ensemble training
        ):
        self.window_sizes = window_sizes
        self.multiscale_windows = multiscale_window_pairs(X_data, y_data, window_sizes, next_predictor)
        self.batch_size = batch_size

    def get_epoch(self):
        # Three questions:
        #   1) How long shoulds segments be?
        #   2) How varied should segment lengths be?
        #   3) How many segments should we sample?
        #
        # 1) Say we have a characteristic length that is the num timesteps which we aim to be able to model well over, char_len
        #    Say we have a null length in which we believe there is no useful temporal information encoded, null_len
        #    This means that the resultant segments total timsteps for each lb window should be roughly more than null_len and roughly not more than char_len
        #
        # 2) The most unbiased way of varing the segment lengths would be uniform sampling, though this might not be the most efficient.
        # 
        # 3) Per epoch, the batches should roughly cover the entire input time series. But we effecitvely cover the time series with
        #    different levels of backprop temporal granularity multiple times within the same epoch.
        #    Do we use more/less samples for each lookback length?
        #
        #   I will start off with a fixed sampling method, but we could potentially make this entire step adaptive to learning rates in the future.
        #
        # Consider doing curriculum learning (not givving the model too much to start with) but make this s unbiased as possible.
        # # REDUCE NOISY GRADIENTS EARLY ON?

        # so we should choose segment samples per lb_size to make the mean seg length char_len
        characteristic_length = 128
        null_length = 8
        n_samples = 32
        multiscale_segments = {}
        for lb_size in self.multiscale_windows:
            multiscale_segments[lb_size] = [
                segment_sample_uniform(self.multiscale_windows[lb_size][0], self.multiscale_windows[lb_size][1], null_length, characteristic_length)
                for _ in range(n_samples)
            ]

        return gen_batches(multiscale_segments, self.batch_size, shuffle_segments=False)

class LazyRandomSegmentTrainer():
    """For very large datasets that need loading procedurally from polars dataframes."""
    def __init__(
            self,
            lb_sizes : set,
            df_partitions : List[pl.DataFrame],
            input_columns: List[str],
            target_columns: List[str],
            batch_size : int,
            samples : int,
            char_length : int,
            null_length : int
        ):
        self.lb_sizes = lb_sizes
        self.batch_size = batch_size
        self.samples = samples,
        self.df_partitions = df_partitions
        self.input_cols = input_columns,
        self.target_cols = target_columns,
        self.char_length = char_length,
        self.null_length = null_length

    def get_epoch(self):
        # sample a load of segments
        multiscale_segments = {}
        for lb_size in self.lb_sizes:
            multiscale_segments[lb_size] = []
            for s in range(self.samples):
                n_df_partitions = len(self.df_partitions)
                partition_idx = int(np.random.uniform(0, n_df_partitions-1))  # sample partition
                partition_df = self.df_partitions[partition_idx]
                # get days in partition
                days = partition_df.select(pl.col("date_id").unique()).collect().to_numpy().flatten()
                date_idx = days[int(np.random.uniform(0, len(days)-1))]
                day_symbols = (
                    partition_df.filter(pl.col("date_id") == date_idx)
                    .select(pl.col("symbol_id").unique())
                ).collect().to_numpy().flatten()
                symbol = day_symbols[int(np.random.uniform(0, len(day_symbols-1)))]
                times = (
                    partition_df.filter((pl.col("date_id") == date_idx) & (pl.col("symbol_id") == symbol))
                    .select(pl.col("time_id").unique())
                ).collect().to_numpy().flatten()

                # NOTE: this currently won't allow for day overlaps
                t0 = times[int(np.random.uniform(0, len(times) - self.char_length))]  # start idx
                t1 = min(int(np.random.uniform(t0 + self.null_length, t0 + self.char_length)), np.max(times))

                filtered_df = partition_df.filter(
                    (pl.col("date_id") == date_idx) &
                    (pl.col("symbol_id") == symbol) &
                    (pl.col("time_id") >= t0) &
                    (pl.col("time_id") <= t1)
                )
                # shape: (time, n_features)
                input_arra = filtered_df.select(self.input_cols).collect().to_numpy()
                # shape: (time, n_responders)
                target_arr = filtered_df.select(self.target_cols).collect().to_numpy()
                
                # Filter and standardise stage
                # features_arr = rf.apply_vectorized_ukf(features_arr)
                # features_arr = process_with_threads(features_arr, rf.apply_vectorized_ukf)
                all_nan_columns = np.isnan(input_arra).all(axis=0)
                input_arra[:, all_nan_columns] = 0
                imputer = KNNImputer(n_neighbors=2)
                input_arra = imputer.fit_transform(input_arra)
                input_arra[:, all_nan_columns] = 0
                for f_idx in range(input_arra.shape[1]):
                    # reduce gaussian noise, fill in blanks
                    # features_arr[:, f_idx] = ke.robust_financial_filter(features_arr[:, f_idx])
                    # standardise
                    input_arra[:, f_idx] = standardise(input_arra[:, f_idx], feature_means[f_idx], feature_stds[f_idx])

                # add the one hot encoding of the symbol id to X_data
                ohv = one_hot_encode(symbol, unique_symbols)
                ohv = ohv.reshape(1, -1)
                ohv = np.repeat(ohv, input_arra.shape[0], axis=0)
                input_arra = np.concatenate((input_arra, ohv), axis=1)

                # now given the lb_size, we generate the respective lb pairs data for thie
                X_data, y_data = fixed_window_pairs(input_arra, target_arr, lb_size, next=False)
                multiscale_segments[lb_size].append((X_data, y_data))
        
        # we let gen_batches convert this into stateful batch data for this epoch
        return self.gen_batches(
                multiscale_segments=multiscale_segments,
                target_batch_size=self.batch_size,
                shuffle_segments=False
            )
    
    def get_test_seg(self, lb_len):
        n_df_partitions = len(self.df_partitions)
        partition_idx = int(np.random.uniform(0, n_df_partitions-1))
        partition_df = partition_dfs[partition_idx]
        days = partition_df.select(pl.col("date_id").unique()).collect().to_numpy().flatten()
        date_idx = days[int(np.random.uniform(0, len(days)-1))]
        day_symbols = (
            partition_df.filter(pl.col("date_id") == date_idx)
            .select(pl.col("symbol_id").unique())
        ).collect().to_numpy().flatten()
        symbol = day_symbols[int(np.random.uniform(0, len(day_symbols-1)))]
        times = (
            partition_df.filter((pl.col("date_id") == date_idx) & (pl.col("symbol_id") == symbol))
            .select(pl.col("time_id").unique())
        ).collect().to_numpy().flatten()

        t0 = times[int(np.random.uniform(0, len(times) - lb_len))]
        t1 = min(t0 + lb_len + 1, np.max(times))

        filtered_df = partition_df.filter(
            (pl.col("date_id") == date_idx) &
            (pl.col("symbol_id") == symbol) &
            (pl.col("time_id") >= t0) &
            (pl.col("time_id") < t1)
        )
        features_arr = filtered_df.select(feature_columns).collect().to_numpy()
        responder_arr = filtered_df.select(responder_columns).collect().to_numpy()
        
        # Filter and standardise stage
        # features_arr = rf.apply_vectorized_ukf(features_arr)
        # features_arr = process_with_threads(features_arr, rf.apply_vectorized_ukf)
        all_nan_columns = np.isnan(features_arr).all(axis=0)
        features_arr[:, all_nan_columns] = 0
        imputer = KNNImputer(n_neighbors=2)
        features_arr = imputer.fit_transform(features_arr)
        features_arr[:, all_nan_columns] = 0
        for f_idx in range(features_arr.shape[1]):
            # reduce gaussian noise, fill in blanks
            # features_arr[:, f_idx] = ke.robust_financial_filter(features_arr[:, f_idx])
            # standardise
            features_arr[:, f_idx] = standardise(features_arr[:, f_idx], feature_means[f_idx], feature_stds[f_idx])

        # add the one hot encoding of the symbol id to X_data
        ohv = one_hot_encode(symbol, unique_symbols)
        ohv = ohv.reshape(1, -1)
        ohv = np.repeat(ohv, features_arr.shape[0], axis=0)
        features_arr = np.concatenate((features_arr, ohv), axis=1)

        X_data, y_data = fixed_window_pairs(features_arr, responder_arr, lb_len, next=False)
        return X_data[0], y_data[0]  # gets the first pair 