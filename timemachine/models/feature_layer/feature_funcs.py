import torch
import torch.nn as nn
from typing import List, Dict, Callable, Tuple, Optional, Union



def differentiable_rolling_mean(x: torch.Tensor, window_size: torch.Tensor, **kwargs) -> torch.Tensor:
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
    and we know that the end of the window must be window_size ahead.
    
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
    # seq_len will be max_window because the x tensor contains the windows for all batch items and the tensor must be ragged
    
    if seq_len == 1:
        return x[:, -1, :]
    
    # generate continuous sequence positions
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)

    # positions operate on the time dimension, make broadcasting work
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)
    
    # center at the last element in the window
    center = window_size - 1

    # we want to discard elements past the centre (we'd ideally not pass these in the first place, but the x tensor
    # must be squared off because it contains all the batch items)
    mask = positions <= (window_size - 1)

    # parameters vary on the channel dimension
    window_size = window_size.view(1, 1, -1)
    
    # Calculate squared distance from center, normalized by window_size
    dist = ((center - positions) / window_size)**2
    
    # gaussian-like weighting (TODO make this an actual bell curve)
    weights = torch.exp(-dist) * mask
    # normalise weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    # compute weighted sum (the average)
    weighted_sum = (x * weights).sum(dim=1)
    
    # output should be (batch_size, 1, input_dim)
    return weighted_sum


def differentiable_rolling_std(x: torch.Tensor, window_size: torch.Tensor, **kwargs) -> torch.Tensor:
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
    mean = differentiable_rolling_mean(x, window_size)
    
    # if sequence length is 1, return zeros
    if seq_len == 1:
        return torch.zeros_like(mean)
    
    # same process as rolling_mean
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)
    
    # Center at the last position
    center = seq_len - 1.0

    # parameters vary on the channel dimension
    window_size = window_size.view(1, 1, -1)

    # make sure not to include elements beyond the end of the window
    mask = positions <= (window_size - 1)
    
    dist = ((center - positions) / window_size)**2
    
    # TODO use a better window here
    weights = torch.exp(-dist) * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # compute weighted squared deviation
    deviation = x - mean.unsqueeze(1)
    sq_deviation = deviation ** 2
    
    # apply weights to squared deviations
    weighted_variance = (sq_deviation * weights).sum(dim=1)
    
    return weighted_variance


def differentiable_ewma(x: torch.Tensor, alpha: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Vectorized differentiable implementation of exponential weighted moving average (EWMA/EMA).
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
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
    
    # Initialize output tensor
    weighted_sum = torch.zeros((batch_size, input_dim), device=x.device)
    
    # We'll compute weighted sum directly to avoid creating a large weights tensor
    cumulative_weight = torch.zeros((batch_size, input_dim), device=x.device)
    
    # Process sequence from newest to oldest
    for i in range(seq_len):
        # Get reversed index (most recent first)
        rev_i = seq_len - 1 - i
        # Current observation
        curr_obs = x[:, rev_i, :]
        
        # For first item (most recent), weight is just alpha
        if i == 0:
            weight = alpha.squeeze(1)  # Shape: (1, input_dim)
        else:
            weight = alpha * ((1 - alpha) ** i)
        
        # Add to weighted sum
        weighted_sum = weighted_sum + curr_obs * weight
        cumulative_weight = cumulative_weight + weight
    
    # Normalize by sum of weights
    return weighted_sum / (cumulative_weight + 1e-8)


def differentiable_ewma_slope(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor = None, **kwargs) -> torch.Tensor:
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
    
    # If sequence length is 1, slope is undefined (return zeros)
    if seq_len <= 1:
        return torch.zeros((batch_size, input_dim), device=x.device)
    
    # First get the EWMA values
    ewma = differentiable_ewma(x, alpha, **kwargs)
    
    # Get the most recent observation
    latest_observation = x[:, -1, :]
    
    # If beta is not provided, derive it from alpha
    if beta is None:
        # Ensure alpha has correct shape (input_dim,)
        if alpha.dim() == 1:
            # Add small epsilon to avoid log(0)
            beta = -torch.log(1 - alpha + 1e-8)
        else:
            # If alpha has different shape, reshape appropriately
            beta = -torch.log(1 - alpha.view(-1) + 1e-8)
    
    # Ensure beta has shape (input_dim,) for proper broadcasting
    if beta.dim() > 1:
        beta = beta.view(-1)
    
    # Calculate slope using the continuous formula
    slope = beta.view(1, -1) * (latest_observation - ewma)
    
    return slope


def differentiable_rolling_rate_of_change(x: torch.Tensor, window_size: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Calculates the rolling rate of change (fractional change) over a window.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        window_size: Learnable window size parameter of shape (input_dim,)
    
    Returns:
        Tensor of shape (batch_size, input_dim) with rate of change values
    """
    batch_size, seq_len, input_dim = x.shape
    
    # If sequence length is 1, return zeros (no rate of change)
    if seq_len == 1:
        return torch.zeros((batch_size, input_dim), device=x.device)
    
    # Get the most recent value
    current_value = x[:, -1, :]
    
    # Reshape window_size for proper broadcasting: (input_dim) -> (1, 1, input_dim)
    window_size = window_size.view(1, 1, -1)
    
    # Create sequence positions
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    positions = positions.view(1, -1, 1)
    
    # Center at the last position
    center = seq_len - 1.0
    
    dist = ((center - positions) / window_size)**2
    weights = torch.exp(-dist)
    
    # Normalize weights per dimension
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    past_value = (x * weights).sum(dim=1)
    
    # Calculate time difference based on window_size
    # This represents the effective time span over which we're measuring change
    time_diff = window_size.squeeze(0).squeeze(0)
    
    # Calculate rate of change: (current - past) / (past * time)
    # Add small epsilon to prevent division by zero
    rate_of_change = (current_value - past_value) / (past_value * time_diff + 1e-8)
    
    return rate_of_change


def differentiable_hurst_exponent(x: torch.Tensor, window_size: torch.Tensor, num_lags: int = 20, min_lag: int = 2, **kwargs) -> torch.Tensor:
    """
    Vectorized differentiable implementation of the Hurst exponent using Rescaled Range (R/S) analysis.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        window_size: Learnable window size parameter of shape (input_dim,)
        num_lags: Number of lag values to use for estimation
        min_lag: Minimum lag value
        
    Returns:
        Tensor of shape (batch_size, input_dim) with Hurst exponent values
    """
    batch_size, seq_len, input_dim = x.shape
    device = x.device
    
    # If sequence length is too small, return 0.5 (random walk)
    if seq_len <= min_lag + 1:
        return 0.5 * torch.ones((batch_size, input_dim), device=device)
    
    # Apply a soft window size constraint
    # We need to ensure our effective window doesn't exceed sequence length
    effective_window = torch.minimum(
        window_size,
        torch.tensor(seq_len - 1, dtype=torch.float32, device=device)
    )
    
    # Use a Gaussian-like weighting for the window
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    positions = positions.view(1, -1, 1).expand(batch_size, -1, input_dim)
    
    # Center at the last position
    center = window_size - 1.0
    effective_window = effective_window.view(1, 1, -1)  # Reshape for broadcasting
    
    mask = positions <= (window_size - 1)

    # Calculate weights based on distance from center and window size
    dist = ((center - positions) / effective_window) ** 2
    weights = torch.exp(-dist) * mask
    
    # Normalize weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Get the weighted price series
    xw = x * weights
    
    # Compute log differences (similar to returns) with proper padding
    # We'll use the differentiable version of diff
    log_x = torch.log(xw + 1e-8)  # Add small epsilon to avoid log(0)
    
    # Compute log returns using a differentiable approach
    log_returns = log_x[:, 1:, :] - log_x[:, :-1, :]
    
    # Define lag values ensuring we don't exceed available data
    max_possible_lag = min(int(effective_window.max().item()), seq_len - 2)
    max_lag = min(max_possible_lag, min_lag + num_lags)
    
    if max_lag <= min_lag:
        # Not enough data points, return 0.5 (random walk)
        return 0.5 * torch.ones((batch_size, input_dim), device=device)
    
    # Create lag values
    lags = torch.arange(min_lag, max_lag, device=device, dtype=torch.float32)
    log_lags = torch.log(lags)
    
    # Initialize tau tensor
    tau_values = torch.zeros((batch_size, len(lags), input_dim), device=device)
    
    # Calculate tau for each lag
    for i, lag in enumerate(lags):
        lag = int(lag.item())  # We need an integer index but this doesn't affect differentiability
        if lag < seq_len - 1:
            # Calculate lagged differences
            lagged_diff = log_returns[:, lag:, :] - log_returns[:, :-lag, :]
            
            # Calculate standard deviation (sqrt of variance)
            # We use a differentiable version that avoids the bias term
            # Note: std = sqrt(mean(x^2) - mean(x)^2)
            diff_mean = torch.mean(lagged_diff, dim=1)
            diff_var = torch.mean(lagged_diff**2, dim=1) - diff_mean**2
            diff_std = torch.sqrt(torch.clamp(diff_var, min=1e-8))  # Clamp to avoid negative values
            
            tau_values[:, i, :] = diff_std
    
    # Take log of tau values
    log_tau = torch.log(tau_values + 1e-8)
    
    # Perform differentiable linear regression to estimate Hurst exponent
    # This replaces np.polyfit with a differentiable version
    
    # Expand log_lags for broadcasting
    log_lags_expanded = log_lags.view(1, -1, 1).expand(batch_size, -1, input_dim)
    
    # Calculate means for the linear regression formula
    n = len(lags)
    sum_x = torch.sum(log_lags_expanded, dim=1)
    sum_y = torch.sum(log_tau, dim=1)
    sum_xy = torch.sum(log_lags_expanded * log_tau, dim=1)
    sum_xx = torch.sum(log_lags_expanded * log_lags_expanded, dim=1)
    
    # Calculate slope using the formula: slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
    denominator = n * sum_xx - sum_x * sum_x
    # Add small epsilon to avoid division by zero
    slope = (n * sum_xy - sum_x * sum_y) / (denominator + 1e-8)
    
    # The Hurst exponent is the slope multiplied by 2
    hurst_exponent = slope * 2.0
    
    # Clamp values to be in the reasonable range [0, 1]
    hurst_exponent = torch.clamp(hurst_exponent, 0.0, 1.0)
    
    return hurst_exponent



# Create default feature configurations
def create_default_feature_configs():
    """Create a set of default feature configurations."""
    return [
        {
            'name': 'rolling_mean',
            'function': differentiable_rolling_mean,
            'learnable_params': {
                'window_size': [1.0, 100.0]
            }
        },
        {
            'name': 'rolling_std',
            'function': differentiable_rolling_std,
            'learnable_params': {
                'window_size': [1.0, 100.0]
            },
            'fixed_params': {
                'eps': 1e-8
            }
        },
        {
            'name': 'ewma',
            'function': differentiable_ewma,
            'learnable_params': {
                'alpha': [0.1, 0.9]
            }
        }
    ]