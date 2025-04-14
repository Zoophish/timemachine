import numpy as np
import pandas as pd
from scipy import signal


def log_relative(series, base_series, eps : float = 1e-4):
    """
    Returns the log of one series to another, i.e. log(series/base_series).
    """
    return np.log(series / (base_series + eps))

def log_returns(series, eps : float = 1e-4):
    """
    Log returns of timeseries, i.e. ln(x[t]/x[t-1]).
    """
    out = np.zeros(shape=(len(series)))
    returns = series[1:] / (series[:-1] + eps)
    out[1:] = np.log(abs(returns) + eps)
    return out

def rolling_future_horizon_abs_change(series, horizon):
    """
    Returns mean absolute change of the future horizon for each position in series.
    """
    assert horizon > 1, "Horizon must be greater than 1."
    out = np.zeros_like(series)
    for i in range(len(series)):
        horizon_end = min(len(series), i + horizon)
        horizon_mean = np.mean(series[i+1:horizon_end])
        out[i] = horizon_mean - out[i]
    return out


def rolling_future_horizon_abs_change_volatility_adjusted(series, horizon):
    """
    Returns mean absolute change divided by the standard deviation of the future horizon for each position in series.
    """
    assert horizon > 1, "Horizon must be greater than 1."
    out = np.zeros_like(series)
    for i in range(len(series)):
        horizon_end = min(len(series), i + horizon + 1)
        horizon_mean = np.mean(series[i+1:horizon_end])
        horizon_std = np.std(series[i+1:horizon_end])
        out[i] = (horizon_mean - series[i]) / (1 + horizon_std)
    return out


def rolling_mean(series, window):
    """
    Calculate the rolling mean of a time series.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Rolling mean of the input series.
    """
    return pd.Series(series).rolling(window=window, min_periods=1).mean().ffill().bfill().to_numpy()

def rolling_std(series, window, shift=0):
    """
    Calculate the rolling standard deviation of a time series.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Rolling standard deviation of the input series.
    """
    pdseries = pd.Series(series).shift(shift)
    return pdseries.rolling(window=window, min_periods=1).std().ffill().bfill().to_numpy()

def rolling_rsi(series, window):
    """
    Calculate the rolling Relative Strength Index (RSI) of a time series.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Rolling RSI of the input series.
    """
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Add small value to avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.ffill().bfill().to_numpy()

def lagged_price_change(series, lag=1):
    """
    Calculate the lagged price change of a time series.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        lag (int): Number of periods to lag.

    Returns:
        np.ndarray: Lagged price change of the input series.
    """
    series = pd.Series(series)
    return series.diff(periods=lag).ffill().bfill().to_numpy()

def rolling_zscore(series, window):
    """
    Calculate the rolling z-score of a time series.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Rolling z-score of the input series.
    """
    series = pd.Series(series)
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)  # Add small value to avoid division by zero
    return zscore.ffill().bfill().to_numpy()

def price_ratio(series1, series2):
    """
    Calculate the price ratio between two time series.

    Args:
        series1 (pd.Series or np.ndarray): First input time series.
        series2 (pd.Series or np.ndarray): Second input time series.

    Returns:
        np.ndarray: Ratio of series1 to series2.
    """
    series1, series2 = pd.Series(series1), pd.Series(series2)
    return (series1 / (series2 + 1e-10)).ffill().bfill().to_numpy()  # Add small value to avoid division by zero


def detrend(series, window):
    """
    Detrend the series by subtracting the rolling mean.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        window (int): The detrend rolling mean window.

    Returns:

    """
    rm = rolling_mean(series, window)
    return series - rm


def shannon_entropy(series, window, bins=10):
    """
    Calculate rolling Shannon entropy of a time series.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
        bins (int): Number of bins for histogram
    
    Returns:
        np.ndarray: Rolling Shannon entropy of the input series
    """
    series = pd.Series(series)
    
    def calc_entropy(x):
        if len(x) == 0 or np.all(x == x[0]):  # Handle zero variance case
            return 0.0
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    entropy = series.rolling(window=window, min_periods=1).apply(calc_entropy)
    return entropy.ffill().bfill().to_numpy()


def rolling_slope(series, window):
    """
    Calculate rolling slope using linear regression.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
    
    Returns:
        np.ndarray: Rolling slope of the input series
    """
    series = pd.Series(series)
    x = np.arange(window)
    
    def calc_slope(y):
        if len(y) < 2:  # Need at least 2 points
            return 0.0
        # Use least squares to fit a line and get slope
        A = np.vstack([x[-len(y):], np.ones(len(y))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    
    slope = series.rolling(window=window, min_periods=1).apply(calc_slope)
    return slope.ffill().bfill().to_numpy()


def rolling_integral(series, window):
    """
    Calculate rolling integral (cumulative sum) over a window.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
    
    Returns:
        np.ndarray: Rolling integral of the input series
    """
    series = pd.Series(series)
    integral = series.rolling(window=window, min_periods=1).sum()
    return integral.ffill().bfill().to_numpy()


def rolling_fft(series, window):
    """
    Calculate rolling FFT power spectrum magnitude.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
    
    Returns:
        np.ndarray: Rolling FFT power magnitude sum
    """
    series = pd.Series(series)
    
    def calc_fft_power(x):
        if len(x) < 2:  # Need at least 2 points
            return 0.0
        # Apply Hanning window and calculate FFT
        windowed = signal.windows.hann(len(x)) * x
        fft_result = np.fft.fft(windowed)
        # Return sum of magnitude spectrum (excluding DC component)
        return np.sum(np.abs(fft_result[1:len(x)//2]))
    
    fft_power = series.rolling(window=window, min_periods=1).apply(calc_fft_power)
    return fft_power.ffill().bfill().to_numpy()


def rolling_skewness(series, window):
    """
    Calculate rolling skewness of a time series.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
    
    Returns:
        np.ndarray: Rolling skewness of the input series
    """
    series = pd.Series(series)
    skew = series.rolling(window=window, min_periods=1).skew()
    return skew.ffill().bfill().to_numpy()


def rolling_hurst(series, window, min_lag=2, max_lag=20):
    """
    Calculate rolling Hurst exponent using R/S analysis.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
        min_lag (int): Minimum lag for R/S calculation
        max_lag (int): Maximum lag for R/S calculation
    
    Returns:
        np.ndarray: Rolling Hurst exponent
    """
    series = pd.Series(series)
    
    def calc_hurst(x):
        if len(x) < max_lag or np.std(x) == 0:
            return 0.5  # Neutral value for insufficient data
        lags = range(min_lag, min(max_lag, len(x)//2))
        rs = []
        for lag in lags:
            diff = x[lag:] - x[:-lag]
            r = np.max(diff) - np.min(diff)
            s = np.std(diff)
            rs.append(r/s if s > 0 else 0)
        if not rs or np.std(np.log(rs)) == 0:
            return 0.5
        return np.polyfit(np.log(lags), np.log(rs), 1)[0]
    
    hurst = series.rolling(window=window, min_periods=1).apply(calc_hurst)
    return hurst.ffill().bfill().to_numpy()


def rolling_autocorrelation(series, window, lag=1):
    """
    Calculate rolling autocorrelation at a specific lag.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
        lag (int): Lag for autocorrelation
    
    Returns:
        np.ndarray: Rolling autocorrelation
    """
    series = pd.Series(series)
    autocorr = series.rolling(window=window, min_periods=lag+1).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0.0
    )
    return autocorr.ffill().bfill().to_numpy()


def rolling_entropy_rate(series, window, lag=1):
    """
    Calculate rolling entropy rate using conditional entropy.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
        lag (int): Lag for conditional entropy
    
    Returns:
        np.ndarray: Rolling entropy rate
    """
    series = pd.Series(series)
    
    def calc_entropy_rate(x):
        if len(x) <= lag:
            return 0.0
        x_past = x[:-lag]
        x_future = x[lag:]
        joint = np.histogram2d(x_past, x_future, bins=10, density=True)[0]
        marginal = np.histogram(x_past, bins=10, density=True)[0]
        joint = joint + 1e-10
        marginal = marginal + 1e-10
        cond_entropy = -np.sum(joint * np.log2(joint / marginal[np.newaxis, :]))
        return cond_entropy
    
    entropy_rate = series.rolling(window=window, min_periods=lag+1).apply(calc_entropy_rate)
    return entropy_rate.ffill().bfill().to_numpy()


def rolling_fractal_dimension(series, window):
    """
    Calculate rolling fractal dimension using box-counting method.
    
    Args:
        series (pd.Series or np.ndarray): Input time series
        window (int): Rolling window size
    
    Returns:
        np.ndarray: Rolling fractal dimension
    """
    series = pd.Series(series)
    
    def calc_fd(x):
        if len(x) < 4:
            return 1.0
        scales = [2, 4, 8]
        counts = []
        for scale in scales:
            bins = np.arange(min(x), max(x) + scale, scale)
            hist, _ = np.histogram(x, bins=bins)
            counts.append(np.sum(hist > 0))
        if np.std(np.log(counts)) == 0:
            return 1.0
        return np.polyfit(np.log(1/np.array(scales)), np.log(counts), 1)[0]
    
    fd = series.rolling(window=window, min_periods=1).apply(calc_fd)
    return fd.ffill().bfill().to_numpy()
