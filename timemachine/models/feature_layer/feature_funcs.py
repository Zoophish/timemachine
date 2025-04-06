import torch
import torch.nn as nn


# @torch.compile
def _gauss_kernel(x, g, k = 3.0):
    """
    Noncyclic windowing kernel. Assumes distance x is normalised to (0, 1).
    g is sharpness, k is like a shift factor for low values of g.
    Becomes a hard window at g->inf.
    Negative x can lead to NaNs.

    y = k^{ -kx^{g} }
    """
    return torch.pow(k, -torch.pow(k * x, g))


def rolling_avg(
        x: torch.Tensor,
        center : int,
        window_size: torch.Tensor,
        gamma : torch.Tensor | float = 3.0,
        **kwargs
    ) -> torch.Tensor:
    """
    The simplest example of a rolling feature that uses a continuous window size (and is therefore differentiable
    with respect to the loss function).

    Rolling features are can be optimised because the only use information from the window size number of elements
    proceeding the target sequence. This way, rolling features are always consistent when viewed by the forecasting
    model which should hopefully make the parameters easier to optimise.

    The way this works is to apply a 'sliding window' over x. The sliding is done outide of this function
    in the sequence loop by incrementally shifting the input x series forwards. This function simply computes
    the rolling average value for the provided x and window size (for each batch item).
    
    The sequence loop will ensure that the first element of x will be the first element in the sliding window,
    and we know that the end of the window must be window_size ahead. More sequence is provided than this function
    is strictly meant to see because batched tensors must be rectangular and ragged.
    
    We use a kernel (window function) weighting instead of a hard window to make the window_size continuous and thus
    differentiable with respect to the loss function. To picture it, only one 'side' of the window function is used,
    i.e. it is centered at the end of the window element.
        
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        window_size: Learnable window size parameter
    
    Returns:
        Tensor of shape (batch_size, input_dim) with rolling mean values
    """
    batch_size, seq_len, input_dim = x.shape
    
    if seq_len == 1:
        return x[:, -1, :]
    
    # generate continuous sequence positions
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)

    # positions operate on the time dimension, make broadcasting work
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)
    
    # we want to discard elements past the centre (we'd ideally not pass these in the first place, but the x tensor
    # must be squared off because it contains all the batch items)
    mask = positions <= center    

    # get the distance from the centre
    normalised_dist = torch.abs((center - positions) / window_size)

    # parameters vary on the channel dimension
    window_size = window_size.view(1, 1, -1)
    
    gamma = gamma.view(1, 1, -1)

    # gaussian-style windowing function (very small at normalised_dist = 1)
    weights = _gauss_kernel(normalised_dist, gamma) * mask
    # normalise weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    # compute weighted sum (the average)
    weighted_sum = (x * weights).sum(dim=1)
    
    # output should be (batch_size, 1, input_dim)
    return weighted_sum


def rolling_std(
        x: torch.Tensor,
        center : int,
        window_size: torch.Tensor,
        gamma : torch.Tensor | float = 3,
        **kwargs,
    ) -> torch.Tensor:
    """
    Rolling weighted variance.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        window_size: Learnable window size parameter
    
    Returns:
        Tensor of shape (batch_size, input_dim) with rolling std values
    """
    batch_size, seq_len, input_dim = x.shape
    
    # the mean of this 'window'/kernel
    mean = rolling_avg(x, center, window_size, gamma)
    
    # if sequence length is 1, return zeros
    if seq_len == 1:
        return torch.zeros_like(mean)
    
    # same process as rolling_mean
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)

    # parameters vary on the channel dimension
    window_size = window_size.view(1, 1, -1)

    # make sure not to include elements beyond the end of the window
    mask = positions <= center
    
    # take the abs to avoid negative distances (window function undefined)
    normalised_dist = torch.abs((center - positions) / window_size)

    gamma = gamma.view(1, 1, -1)
    
    weights = _gauss_kernel(normalised_dist, gamma) * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # compute weighted squared deviation
    deviation = x - mean.unsqueeze(1)
    sq_deviation = deviation ** 2
    
    # apply weights to squared deviations
    weighted_variance = (sq_deviation * weights).sum(dim=1)
    
    return weighted_variance


def rolling_skewness(
        x: torch.Tensor,
        center : int,
        window_size: torch.Tensor,
        gamma : torch.Tensor | float = 3,
        **kwargs
    ) -> torch.Tensor:
    """
    Rolling weighted skewness.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        window_size: Learnable window size parameter
    
    Returns:
        Tensor of shape (batch_size, input_dim) with rolling std values
    """
    batch_size, seq_len, input_dim = x.shape
    
    # the mean of this 'window'/kernel
    mean = rolling_avg(x, center, window_size, gamma)
    
    # if sequence length is 1, return zeros
    if seq_len == 1:
        return torch.zeros_like(mean)
    
    # same process as rolling_mean
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)

    # parameters vary on the channel dimension
    window_size = window_size.view(1, 1, -1)

    # make sure not to include elements beyond the end of the window
    mask = positions <= center
    
    normalised_dist = torch.abs((center - positions) / window_size)

    gamma = gamma.view(1, 1, -1)
    
    weights = _gauss_kernel(normalised_dist, gamma) * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # compute weighted squared deviation
    deviation = x - mean.unsqueeze(1)
    
    # weighted sum of (deviation^3)
    numerator = (weights * torch.pow(deviation, 3)).sum(dim=1)
    # squared weighted sum of std
    denominator = (torch.pow(deviation, 2) * weights).sum(dim=1) ** (3/2)
    
    skewness = numerator / denominator
    return skewness


def rolling_kurtosis(
        x: torch.Tensor,
        center : int,
        window_size: torch.Tensor,
        gamma : torch.Tensor | float = 3,
        **kwargs
    ) -> torch.Tensor:
    """
    Rolling weighted kurtosis (tailedness).
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        window_size: Learnable window size parameter
    
    Returns:
        Tensor of shape (batch_size, input_dim) with rolling std values
    """
    batch_size, seq_len, input_dim = x.shape
    
    # the mean of this 'window'/kernel
    mean = rolling_avg(x, center, window_size, gamma)
    
    # if sequence length is 1, return zeros
    if seq_len == 1:
        return torch.zeros_like(mean)
    
    # same process as rolling_mean
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)

    # parameters vary on the channel dimension
    window_size = window_size.view(1, 1, -1)

    # make sure not to include elements beyond the end of the window
    mask = positions <= center
    
    normalised_dist = torch.abs((center - positions) / window_size)

    gamma = gamma.view(1, 1, -1)
    
    weights = _gauss_kernel(normalised_dist, gamma) * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # compute weighted squared deviation
    deviation = x - mean.unsqueeze(1)
    
    # weighted sum of (deviation^4)
    numerator = (weights * torch.pow(deviation, 4)).sum(dim=1)
    # squared weighted sum of std
    denominator = (torch.pow(deviation, 2) * weights).sum(dim=1) ** 2
    
    kurtosis = numerator / denominator
    return kurtosis




def rolling_hurst(
        x: torch.Tensor,
        center : int,
        window_size : torch.Tensor,
        ranges: torch.Tensor | None = None,
        gamma : torch.Tensor | float = 3.0,
        **kwargs
    ) -> torch.Tensor:
    """
    Approximate the Hurst exponent using RS analysis.
        
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        max_window: Maximum window (range) size
        ranges: Ranges to sample.
    
    Returns:
        Tensor of shape (batch_size, input_dim) with rolling mean values
    """
    batch_size, seq_len, input_dim = x.shape

    if ranges is None:
        default_ranges = torch.tensor([16, 32, 64, 128], device=x.device)
        ranges = default_ranges.unsqueeze(1).expand(-1, input_dim)  # (4, input_dim)
    n_ranges = len(ranges)

    # same process as rolling_mean
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)
    
    gamma = gamma.view(1, 1, -1)

    # make sure not to include elements beyond the end of the window
    mask = positions <= center

    m = torch.zeros((batch_size, n_ranges, input_dim))  # means
    Y = torch.zeros((batch_size, n_ranges, seq_len, input_dim))  # deviations
    y = torch.zeros((batch_size, n_ranges, seq_len, input_dim))  # cumulative sum of deviations
    R = torch.zeros((batch_size, n_ranges, input_dim))  # widest difference of range
    s = torch.zeros((batch_size, n_ranges, input_dim))  # standard deviations of range
    RS = torch.zeros((batch_size, n_ranges, input_dim))  # rescaled range

    eps = 1e-8
    gamma = gamma.view(1, 1, -1)

    for i in range(ranges.shape[0]):
        # this range (window size) for all input dims
        range_size = ranges[i, :]
        # make the kernel for this range
        normalised_dist = torch.abs((center - positions) / range_size)
        weights = _gauss_kernel(normalised_dist, gamma) * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
        # NOTE can probably take some of these out the loop
        m[:, i, :]    = rolling_avg(x, center, range_size, gamma)  # get the mean of this range NOTE you could optimise this and std by expanding it here
        Y[:, i, :, :] = (x - m[:, i, :].unsqueeze(1)) * weights  # deviation from mean
        y[:, i, :, :] = torch.cumsum(Y[:, i, :, :], dim=1)  # cumulative sum of deviations
        R[:, i, :]    = torch.max(y[:, i, :, :], dim=1)[0] - torch.min(y[:, i, :, :], dim=1)[0]  # max range
        s[:, i, :]    = rolling_std(x, center, range_size, gamma)  # standard deviation of range
        RS[:, i, :]   = R[:, i, :] / (s[:, i, :] + eps)
    
    log_size = torch.log(ranges)  # shape (batch_size, n_ranges, input_dim)
    log_RS = torch.log(RS.clamp(min=eps))  # shape (batch_size, n_ranges, input_dim)

    B, N, D = log_RS.shape
    log_RS_flat = log_RS.permute(0, 2, 1).reshape(B * D, N)  # (batch_size * input_dim, n_ranges)
    ones = torch.ones((B * D, N), device=x.device)
    log_size_flat = log_size.t().unsqueeze(0).expand(B, D, N).reshape(B * D, N)
    X = torch.stack([ones, log_size_flat], dim=-1)  # (B * D, N, 2)

    solution = torch.linalg.lstsq(X, log_RS_flat)
    beta = solution.solution  # (B * D, 2), [c, H] per problem
    estimated_H = beta[:, 1].reshape(B, D)  # (batch_size, input_dim)

    return estimated_H.unsqueeze(1)



def rolling_dft_magnitude(
    x: torch.Tensor,
    center: int,
    window_size: torch.Tensor,         # Shape (input_dim,) - From learnable_params
    gamma: torch.Tensor,               # Shape (input_dim,) - From learnable_params
    center_period: torch.Tensor,       # Shape (input_dim,) - From learnable_params
    frequency_smoothness: torch.Tensor,# Shape (input_dim,) - From learnable_params
    min_period_threshold: float = 1.0, # Threshold below which period is treated as DC (k=0)
    **kwargs
) -> torch.Tensor:
    """
    Calculates a weighted average of RFFT magnitudes centered around a LEARNABLE
    frequency (derived from center_period) with LEARNABLE smoothness/width,
    all defined per input channel. Designed for out_size=1 per channel.

    Args:
        x: Input tensor (batch_size, seq_len, input_dim).
        center: Current time step index.
        window_size: Learnable window size (input_dim,).
        gamma: Learnable time-domain kernel gamma (input_dim,).
        center_period: Learnable center period of the frequency band (input_dim,).
                       Values constrained by the range in feature_config.
        frequency_smoothness: Learnable smoothness/width (std dev) of the frequency
                              weighting kernel (input_dim,). Constrained by range.
        min_period_threshold: Periods below this value will map to the DC component (k=0).
                              Should be >= smallest period resolvable/meaningful.
        **kwargs: Catches unused parameters.

    Returns:
        Tensor of shape (batch_size, input_dim) containing the weighted average RFFT
        magnitude around the learned center period for each channel.
    """
    batch_size, seq_len, input_dim = x.shape
    device = x.device
    eps = 1e-8 # Epsilon for numerical stability

    # --- 0. Reshape learnable parameters for broadcasting ---
    # Shape (D,) -> (1, 1, D) for time-domain ops
    window_size = window_size.view(1, 1, input_dim)
    gamma = gamma.view(1, 1, input_dim)
    # Shape (D,) -> (1, D) for frequency-domain ops
    center_period = center_period.view(1, input_dim)
    frequency_smoothness = frequency_smoothness.view(1, input_dim)

    # Ensure smoothness is positive (should be guaranteed by config range > 0)
    # assert torch.all(frequency_smoothness > 0), "Smoothness range must be > 0"

    # --- 1. Time-Domain Windowing ---
    fft_window_len = center + 1
    if fft_window_len <= 1:
        return torch.zeros((batch_size, input_dim), device=device)

    positions = torch.arange(fft_window_len, device=device, dtype=torch.float32)
    rel_positions = (center - positions).view(1, -1, 1) # Shape (1, N, 1)
    normalised_dist = torch.abs(rel_positions / (window_size + eps)) # Shape (1, N, D)
    # Apply kernel, expand batch dim, normalize
    weights = _gauss_kernel(normalised_dist, gamma).expand(batch_size, -1, -1) # Shape (B, N, D)
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    # Apply weights to data
    window_data = x[:, :fft_window_len, :] # Shape (B, N, D)
    weighted_window = window_data * weights

    # --- 2. RFFT ---
    rfft_output = torch.fft.rfft(weighted_window, n=fft_window_len, dim=1)
    rfft_magnitudes = torch.abs(rfft_output) # Shape: (B, F, D), F = N//2 + 1
    num_freq_bins = rfft_magnitudes.shape[1]

    # --- 3. Calculate Frequency-Domain Weights ---
    # Target center frequency bin (float), per channel
    # Shape (1, D)
    k_center_float = (fft_window_len / (center_period + eps))

    # Handle mapping near-zero periods (below threshold) to DC component (k=0)
    k_center_float[center_period < min_period_threshold] = 0.0

    # Clamp k_center_float to valid range [0, N//2] just in case?
    # Might prevent gradients from pushing period too low if threshold is poorly set.
    # k_center_float = k_center_float.clamp(min=0, max=num_freq_bins - 1) # Optional clamp

    # Actual frequency bin indices: 0, 1, ..., F-1
    # Shape: (F,)
    k_bins = torch.arange(num_freq_bins, device=device, dtype=torch.float32)

    # Calculate squared distance from center bin for each bin, per channel
    # Broadcasting: k_bins(F,1) - k_center_float(1,D) -> Shape (F, D)
    freq_dist_sq = torch.square(k_bins.view(-1, 1) - k_center_float)

    # Apply Gaussian kernel in frequency domain using learnable smoothness (std dev)
    # Broadcasting: freq_dist_sq(F,D) / ( (...) * frequency_smoothness(1,D)**2 + eps)
    # Shape (F, D)
    # Note: smoothness acts like std deviation here. Variance = smoothness^2.
    variance = torch.square(frequency_smoothness) + eps
    freq_weights = torch.exp(-freq_dist_sq / (2 * variance))

    # Normalize weights across frequency bins for each channel
    # Shape (F, D)
    freq_weights = freq_weights / (freq_weights.sum(dim=0, keepdim=True) + eps)

    # Expand freq_weights for batch dimension: (1, F, D) -> (B, F, D)
    # Use .detach().clone() if expansion causes gradient issues, but unlikely here.
    freq_weights_expanded = freq_weights.unsqueeze(0) # .expand(batch_size, -1, -1) implicitly

    # --- 4. Calculate Weighted Average Magnitude ---
    # Element-wise: rfft_magnitudes(B, F, D) * freq_weights_expanded(B, F, D)
    # Sum along frequency dim (dim=1)
    # Output shape: (B, D)
    weighted_avg_magnitude = (rfft_magnitudes * freq_weights_expanded).sum(dim=1)

    return weighted_avg_magnitude



def differentiable_ewma(x: torch.Tensor, context : torch.tensor, alpha: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Vectorized differentiable implementation of exponential weighted moving average (EWMA/EMA).
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        last: The last value of the ewma sequence.
        alpha: Learnable smoothing parameter (between 0 and 1)
    
    Returns:
        Tensor of shape (batch_size, input_dim) with EWMA values
    """
    batch_size, seq_len, input_dim = x.shape
    
    # If sequence length is 1, just return the last observation
    if seq_len == 1:
        return x[:, 0, :]
    
    # broadcast the channel dimension properly
    alpha = alpha.view(1, 1, -1)
    
    weighted_sum = (1 - alpha) * context + alpha * x[:, -1, :]
    context.copy_(weighted_sum)

    return weighted_sum


def differentiable_ewma_slope(x: torch.Tensor, context : torch.tensor, alpha: torch.Tensor, beta: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """
    Calculates the slope (rate of change) of the EWMA, maintaining differentiability.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        alpha: Learnable smoothing parameters of shape (input_dim,)
        beta: Optional scaling parameter for slope calculation, defaults to None
    
    Returns:
        Tensor of shape (batch_size, input_dim) with EWMA slope values
    """
    batch_size, seq_len, input_dim = x.shape
    # broadcast the channel dimension properly
    alpha = alpha.view(1, 1, -1)
    
    # If sequence length is 1, just return the last observation
    if seq_len == 1:
        return torch.zeros_like(alpha)
    
    ewma = (1 - alpha) * context + alpha * x[:, -1, :]
    context = ewma
    slope = ewma - context

    return slope