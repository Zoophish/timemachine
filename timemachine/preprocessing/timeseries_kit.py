# a selection of basic features for time series

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol


def log_relative(series, base_series, eps : float = 1e-4):
    """
    Returns the log of one series to another, i.e. log(series/base_series).
    """
    return np.log(series / (base_series + eps))

def log_returns(series):
    """
    Log returns of timeseries, i.e. ln(x[t]/x[t-1]).
    """
    out = np.zeros(shape=(len(series)))
    out[1:] = np.log(series[1:] / series[:-1])
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

def rolling_std(series, window):
    """
    Calculate the rolling standard deviation of a time series.

    Args:
        series (pd.Series or np.ndarray): Input time series.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Rolling standard deviation of the input series.
    """
    return pd.Series(series).rolling(window=window, min_periods=1).std().ffill().bfill().to_numpy()

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


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, state_transition, process_noise, observation_matrix, observation_noise):
        self.state = initial_state.reshape(-1, 1)  # Make column vector
        self.covariance = initial_covariance
        self.F = state_transition
        self.Q = process_noise
        self.H = observation_matrix
        self.R = observation_noise

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q
        return self.state[0, 0]  # Return just the price component

    def update(self, observation):
        if np.isnan(observation):
            return self.state[0, 0]  # Return just the price component
            
        y = observation - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.covariance = self.covariance - np.dot(np.dot(K, self.H), self.covariance)
        return self.state[0, 0]  # Return just the price component

def kalman_filter(data, initial_state, initial_covariance, state_transition, process_noise, observation_matrix, observation_noise):
    """
    Apply the Kalman Filter to a time series.
    
    Parameters:
        data (np.array): The observed time series data.
        initial_state (np.array): Initial state estimate (x0).
        initial_covariance (np.array): Initial covariance matrix (P0).
        state_transition (np.array): State transition matrix (F).
        process_noise (np.array): Process noise covariance matrix (Q).
        observation_matrix (np.array): Observation matrix (H).
        observation_noise (np.array): Observation noise covariance matrix (R).
    
    Returns:
        np.array: Smoothed state estimates.
    """
    kf = KalmanFilter(initial_state, initial_covariance, state_transition, process_noise, observation_matrix, observation_noise)
    estimates = np.zeros((len(data)))
    # for i, observation in enumerate(data):
    #     kf.predict()
    #     estimates[i] = kf.update(observation)

    for i, observation in enumerate(data):
        # Get prediction for this timestep
        predicted_state = kf.predict()
        
        if np.isnan(observation):
            # Use the prediction when measurement is missing
            estimates[i] = predicted_state
        else:
            # Use the updated state when we have a measurement
            estimates[i] = kf.update(observation)
    return np.array(estimates)


def kalman_filter_init(data):
    first_valid = next((x for x in data if not np.isnan(x)), 0)

    # State vector: [data, trend]
    initial_state = np.array([first_valid, 0.0])
    
         # Larger uncertainty in trend estimate
    initial_covariance = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ])
    
    # Price changes by trend, trend persists
    state_transition = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    # Process noise increases with time
    process_noise = np.array([
        [0.01, 0.0],
        [0.0, 0.001]
    ])
    # We only observe the price, not the trend
    observation_matrix = np.array([[1.0, 0.0]])
    
    # Measurement noise
    observation_noise = np.array([[0.1]])

    # initial_covariance = np.array([1.0])
    # state_transition = np.array([[1.0]])
    # process_noise = np.array([[0.01]])
    # observation_matrix = np.array([[1.0]])
    # observation_noise = np.array([[0.1]])
    return kalman_filter(data, initial_state, initial_covariance, state_transition, process_noise, observation_matrix, observation_noise)


def ensemble_kalman_filter(observations, initial_state, initial_covariance, process_noise, observation_noise, ensemble_size):
    """
    Ensemble Kalman Filter for a time series (scales better when number of parameters is large).

    Parameters:
        observation (np.array): observed time series data (shape: [time_steps, observation_dim])
        initial_state (np.array): initial state estimate (shape: [state_dim])
        initial_covariance: (np.array): initial state covariance (shape: [state_dim, state_dim])
        process_noise: (np.array): process noise covariance (shape: [state_dim, state_dim])
        observation_noise:(np.array): observation noise covariance (shape: [observation_dim, observation_dim])
        ensemble_size (int): number of ensemble members

    Returns:
        state_estimates (np.array): filtered state estimates (shape: [time_steps, state_dim])
    """
    state_dim = initial_state.shape[0]
    observation_dim = 1 # observations.shape[1]
    time_steps = observations.shape[0]

    # Initialize ensemble
    ensemble = np.random.multivariate_normal(initial_state, initial_covariance, ensemble_size).T  # shape: [state_dim, ensemble_size]

    # Array to store state estimates
    state_estimates = np.zeros((time_steps, state_dim))

    for t in range(time_steps):
        # Forecast step
        ensemble = np.random.multivariate_normal(np.zeros(state_dim), process_noise, ensemble_size).T + ensemble  # Add process noise
        ensemble_mean = np.mean(ensemble, axis=1)  # Mean of the ensemble

        # Compute ensemble covariance
        ensemble_anomaly = ensemble - ensemble_mean[:, np.newaxis]
        ensemble_covariance = np.dot(ensemble_anomaly, ensemble_anomaly.T) / (ensemble_size - 1)

        # Update step (Kalman gain)
        H = np.eye(observation_dim, state_dim)  # Observation matrix (assuming linear observation model)
        K = np.dot(np.dot(ensemble_covariance, H.T), np.linalg.inv(np.dot(np.dot(H, ensemble_covariance), H.T) + observation_noise))

        # Generate perturbed observations
        perturbed_observations = observations[t] + np.random.multivariate_normal(np.zeros(observation_dim), observation_noise, ensemble_size).T

        # Update ensemble
        ensemble = ensemble + np.dot(K, perturbed_observations - np.dot(H, ensemble))

        # Store the mean of the updated ensemble as the state estimate
        state_estimates[t] = np.mean(ensemble, axis=1)

    return state_estimates


def non_linear_model(state):
    """
    Example of a non-linear model.
    """
    return state + 0.1 * np.sin(state)  # Replace with your non-linear model

def ensemble_kalman_filter_nonlinear(observations, initial_state, initial_covariance, process_noise, observation_noise, ensemble_size):
    state_dim = initial_state.shape[0]
    observation_dim = 1 # observations.shape[1]
    time_steps = observations.shape[0]

    # Initialize ensemble
    ensemble = np.random.multivariate_normal(initial_state, initial_covariance, ensemble_size).T

    # Array to store state estimates
    state_estimates = np.zeros((time_steps, state_dim))

    for t in range(time_steps):
        # Forecast step with non-linear model
        ensemble = np.array([non_linear_model(state) for state in ensemble.T]).T
        ensemble += np.random.multivariate_normal(np.zeros(state_dim), process_noise, ensemble_size).T  # Add process noise

        ensemble_mean = np.mean(ensemble, axis=1)

        # Compute ensemble covariance
        ensemble_anomaly = ensemble - ensemble_mean[:, np.newaxis]
        ensemble_covariance = np.dot(ensemble_anomaly, ensemble_anomaly.T) / (ensemble_size - 1)

        # Update step (Kalman gain)
        H = np.eye(observation_dim, state_dim)  # Observation matrix (assuming linear observation model)
        K = np.dot(np.dot(ensemble_covariance, H.T), np.linalg.inv(np.dot(np.dot(H, ensemble_covariance), H.T) + observation_noise))

        # Generate perturbed observations
        perturbed_observations = observations[t] + np.random.multivariate_normal(np.zeros(observation_dim), observation_noise, ensemble_size).T

        # Update ensemble
        ensemble = ensemble + np.dot(K, perturbed_observations - np.dot(H, ensemble))

        # Store the mean of the updated ensemble as the state estimate
        state_estimates[t] = np.mean(ensemble, axis=1)

    return state_estimates



class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_noise, measurement_noise, bounds):
        """
        Initialize the Particle Filter.
        
        Parameters:
            num_particles (int): Number of particles to use.
            state_dim (int): Dimension of the state vector.
            process_noise (float): Standard deviation of the process noise.
            measurement_noise (float): Standard deviation of the measurement noise.
            bounds (list of tuples): Lower and upper bounds for each state dimension.
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles using Sobol sequence
        self.particles = self.initialize_particles_sobol(num_particles, state_dim, bounds)
        self.weights = np.ones(num_particles) / num_particles  # Uniform weights

    def initialize_particles_sobol(self, num_particles, state_dim, bounds):
        """
        Initialize particles using a Sobol sequence.
        """
        sobol = Sobol(d=state_dim, scramble=True)
        particles = sobol.random(num_particles)
        
        # Scale particles to the desired bounds
        for i in range(state_dim):
            particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        
        return particles

    def predict(self):
        """
        Predict the next state of the particles using the process model.
        """
        # Add process noise to the particles
        self.particles += np.random.randn(self.num_particles, self.state_dim) * self.process_noise

    def update(self, measurement):
        """
        Update the particle weights based on the measurement.
        """
        # Compute the likelihood of each particle given the measurement
        likelihood = np.exp(-0.5 * np.sum((self.particles - measurement) ** 2, axis=1) / self.measurement_noise ** 2)
        
        # Update the weights using importance sampling
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)  # Normalize weights

    def resample(self):
        """
        Resample the particles based on their weights.
        """
        # Systematic resampling
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def filter(self, measurements):
        """
        Run the particle filter on the input measurements.
        """
        filtered_estimates = np.zeros_like(measurements)
        
        for t, measurement in enumerate(measurements):
            # Predict the next state
            self.predict()
            
            # Update weights based on the measurement
            self.update(measurement)
            
            # Resample particles
            self.resample()
            
            # Compute the filtered estimate as the mean of the particles
            filtered_estimates[t] = np.mean(self.particles, axis=0)
        
        return filtered_estimates
    

import numpy as np
from scipy.stats.qmc import Sobol

class EnhancedParticleFilter:
    def __init__(self, num_particles, state_dim, process_noise, measurement_noise, bounds):
        """
        Initialize the Enhanced Particle Filter.
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.bounds = bounds
        
        # Initialize particles using Sobol sequence
        self.particles = self.initialize_particles_sobol(num_particles, state_dim, bounds)
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles_sobol(self, num_particles, state_dim, bounds):
        """
        Initialize particles using a Sobol sequence.
        """
        sobol = Sobol(d=state_dim, scramble=True)
        particles = sobol.random(num_particles)
        
        # Scale particles to the desired bounds
        for i in range(state_dim):
            particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        
        return particles

    def predict(self):
        """
        Predict the next state of the particles using a GBM model.
        """
        drift = 0.01  # Example drift
        volatility = 0.1  # Example volatility
        dt = 1.0  # Time step
        
        # GBM process
        self.particles *= np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.random.randn(self.num_particles, self.state_dim) * np.sqrt(dt))

    def update(self, measurement):
        """
        Update the particle weights based on the measurement.
        """
        # Compute the likelihood using a heavy-tailed distribution (e.g., Student's t)
        degrees_of_freedom = 3  # Example degrees of freedom for Student's t
        residuals = self.particles - measurement
        likelihood = (1 + np.sum(residuals**2, axis=1) / degrees_of_freedom) ** (-0.5 * (degrees_of_freedom + self.state_dim))
        
        # Update the weights
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)  # Normalize weights

    def resample(self):
        """
        Resample the particles based on their weights.
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def filter(self, measurements):
        """
        Run the particle filter on the input measurements.
        """
        filtered_estimates = np.zeros_like(measurements)
        
        for t, measurement in enumerate(measurements):
            # Predict the next state
            self.predict()
            
            # Update weights based on the measurement
            self.update(measurement)
            
            # Resample particles
            self.resample()
            
            # Compute the filtered estimate as the mean of the particles
            filtered_estimates[t] = np.mean(self.particles, axis=0)
        
        return filtered_estimates
