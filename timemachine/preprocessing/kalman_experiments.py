import numpy as np
from scipy.linalg import sqrtm


def adaptive_process_function(x, dt=1.0):
    """
    Adaptive nonlinear process function that combines:
    - Mean reversion (for volatility-like series)
    - Momentum with decay (for trending prices)
    - Stochastic volatility (for varying noise levels)
    
    State vector x = [level, trend, volatility, mean_level]
    """
    level, trend, vol, mean = x
    
    # Mean reversion force
    mean_force = 0.1 * (mean - level)
    
    # Momentum decay
    trend_decay = 0.95
    
    # Volatility mean reversion
    vol_mean = 0.2
    vol_force = 0.1 * (vol_mean - vol)
    
    # Update equations
    new_level = level + trend * dt + mean_force * dt
    new_trend = trend * trend_decay + mean_force * 0.1
    new_vol = vol + vol_force * dt
    new_mean = mean + 0.01 * (level - mean) * dt
    
    return np.array([new_level, new_trend, new_vol, new_mean])

def adaptive_measurement_function(x):
    """We only measure the level"""
    return np.array([x[0]])

class RobustUKF:
    def __init__(self, measurement_scale=1.0):
        """
        Initialize a robust UKF for financial time series.
        
        Args:
            measurement_scale: Typical scale of the measurements
        """
        self.state_dim = 4  # level, trend, volatility, mean
        self.measurement_dim = 1
        
        # UKF parameters
        self.alpha = 0.001  # Small alpha for financial data
        self.beta = 2.0     # Gaussian assumption
        self.kappa = 0.0    # Secondary scaling parameter
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        
        # Calculate weights
        self.weights_m = np.zeros(2 * self.state_dim + 1)
        self.weights_c = np.zeros(2 * self.state_dim + 1)
        
        self.weights_m[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.weights_c[0] = self.lambda_ / (self.state_dim + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.state_dim + 1):
            self.weights_m[i] = 1.0 / (2 * (self.state_dim + self.lambda_))
            self.weights_c[i] = self.weights_m[i]
            
        # Initialize adaptive noise
        self.measurement_scale = measurement_scale
        self.innovation_window = []
        self.max_window = 30
        
    def initialize(self, data):
        """Initialize state and covariance from data"""
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            self.x = np.zeros(4)  # all zero state
            scale = 1.0
        else:
            self.x = np.array([
                clean_data[0],  # level
                0.0,           # trend
                0.1,           # volatility
                clean_data[0]  # mean level
            ])
            scale = np.std(clean_data) if len(clean_data) > 1 else 1.0
        
        # Initial covariance
        scale = np.std(clean_data) if len(clean_data) > 1 else 1.0
        self.P = np.diag([
            scale * 0.1,    # level uncertainty
            scale * 0.01,   # trend uncertainty
            0.1,            # volatility uncertainty
            scale * 0.1     # mean uncertainty
        ])
        
        # Adaptive noise matrices
        self.Q = np.diag([0.01, 0.001, 0.001, 0.001]) * scale
        self.R = np.array([[0.1]]) * scale
        
    def adapt_noise(self, innovation):
        """Adapt noise parameters based on innovations"""
        if np.isnan(innovation):
            return
        self.innovation_window.append(innovation)
        if len(self.innovation_window) > self.max_window:
            self.innovation_window.pop(0)
            
        if len(self.innovation_window) > 1:
            innov_std = np.std(self.innovation_window)
            self.R = np.array([[max(innov_std * 0.5, self.measurement_scale * 0.01)]])
            
            # Increase process noise if innovations are large
            if abs(innovation) > 3 * innov_std:
                self.Q *= 1.5
            else:
                self.Q *= 0.99
                
    def generate_sigma_points(self):
        """Generate sigma points"""
        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = self.x
        
        # Use real square root and ensure matrix is positive definite
        try:
            U = sqrtm((self.state_dim + self.lambda_) * self.P)
            # Force real part if complex values appear
            U = np.real(U)
        except:
            # Fallback: use diagonal square root if matrix decomposition fails
            U = np.diag(np.sqrt(np.diag((self.state_dim + self.lambda_) * self.P)))
        
        for i in range(self.state_dim):
            sigma_points[i + 1] = self.x + U[i]
            sigma_points[self.state_dim + i + 1] = self.x - U[i]
            
        return sigma_points
        
    def predict(self):
        """Prediction step"""
        sigma_points = self.generate_sigma_points()
        self.chi = np.array([adaptive_process_function(sigma) for sigma in sigma_points])
        
        self.x_pred = np.sum(self.weights_m.reshape(-1, 1) * self.chi, axis=0)
        
        self.P_pred = np.zeros_like(self.P)
        for i in range(len(self.chi)):
            diff = (self.chi[i] - self.x_pred).reshape(-1, 1)
            self.P_pred += self.weights_c[i] * diff @ diff.T
        self.P_pred += self.Q
        
    def update(self, measurement):
        """Update step"""
        if np.isnan(measurement):
            self.x = self.x_pred
            self.P = self.P_pred
            return self.x[0]  # Return level
            
        gamma = np.array([adaptive_measurement_function(sigma) for sigma in self.chi])
        
        y_pred = np.sum(self.weights_m.reshape(-1, 1) * gamma, axis=0)
        
        Pyy = np.zeros((self.measurement_dim, self.measurement_dim))
        Pxy = np.zeros((self.state_dim, self.measurement_dim))
        
        for i in range(len(gamma)):
            diff_y = (gamma[i] - y_pred).reshape(-1, 1)
            diff_x = (self.chi[i] - self.x_pred).reshape(-1, 1)
            
            Pyy += self.weights_c[i] * diff_y @ diff_y.T
            Pxy += self.weights_c[i] * diff_x @ diff_y.T
            
        Pyy += self.R
        
        K = Pxy @ np.linalg.inv(Pyy)
        
        innovation = measurement - y_pred
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pyy @ K.T
        
        self.adapt_noise(innovation[0])
        
        return self.x[0]  # Return level

def robust_financial_filter(data):
    """
    Apply robust UKF to financial time series.
    
    Args:
        data: Input measurements (can contain NaN)
        
    Returns:
        Filtered values
    """
    if np.all(np.isnan(data)) or (np.all(data[~np.isnan(data)] == 0)):
        return np.zeros_like(data)
    # Check if all values in data are the same
    non_nan_data = data[~np.isnan(data)]  # Ignore NaN values
    if np.all(non_nan_data == non_nan_data[0]):
        return data  # Return the original data if all values are the same

    # Convert to numpy array
    data = np.array(data, dtype=float)
    
    # Calculate measurement scale
    clean_data = data[~np.isnan(data)]
    measurement_scale = np.std(clean_data) if len(clean_data) > 1 else 1.0
    
    # Initialize filter
    ukf = RobustUKF(measurement_scale)
    ukf.initialize(data)
    
    # Process data
    filtered = np.zeros(len(data))
    for i, measurement in enumerate(data):
        ukf.predict()
        filtered[i] = ukf.update(measurement)
        
    return filtered