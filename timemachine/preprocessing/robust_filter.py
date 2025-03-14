import numpy as np
from scipy.linalg import sqrtm

class FinancialUKF:
    def __init__(self, n_dims, alpha=0.001, beta=2.0, kappa=0.0):
        """
        This is a vectorised unscented Kalman filter.
        The default process function is a combination of mean reversion, momentum with decay and stochastic volatility.
        
        Args:
            n_dims: Number of dimensions for each state variable
            alpha: UKF scaling parameter
            beta: UKF secondary scaling parameter for prior knowledge of state distribution
            kappa: UKF tertiary scaling parameter
        """
        self.n_dims = n_dims
        self.state_dim = 4  # Same state structure as original: level, trend, vol, mean
        self.total_dim = self.state_dim * n_dims
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = self.alpha**2 * (self.total_dim + self.kappa) - self.total_dim
        
        # Calculate weights
        self.weights_m = np.zeros(2 * self.total_dim + 1)
        self.weights_c = np.zeros(2 * self.total_dim + 1)
        
        self.weights_m[0] = self.lambda_ / (self.total_dim + self.lambda_)
        self.weights_c[0] = (self.lambda_ / (self.total_dim + self.lambda_) + 
                           (1 - self.alpha**2 + self.beta))
        
        for i in range(1, 2 * self.total_dim + 1):
            self.weights_m[i] = 1.0 / (2 * (self.total_dim + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def process_function(self, x, dt=1.0):
        """
        Vectorized process function for multiple dimensions.
        Adaptive nonlinear process function that combines:
        - Mean reversion (for volatility-like series)
        - Momentum with decay (for trending prices)
        - Stochastic volatility (for varying noise levels)
        
        Args:
            x: State array of shape (4 * n_dims,) containing [level, trend, vol, mean] for each dimension
        """
        x = x.reshape(self.state_dim, self.n_dims)
        level, trend, vol, mean = x
        
        # Parameters
        mean_reversion_strength = 0.1
        trend_decay = 0.95
        vol_mean = 0.2
        vol_strength = 0.1
        mean_adapt_rate = 0.01
        
        # Update equations (vectorized)
        mean_force = mean_reversion_strength * (mean - level)
        vol_force = vol_strength * (vol_mean - vol)
        
        new_level = level + trend * dt + mean_force * dt
        new_trend = trend * trend_decay + mean_force * 0.1
        new_vol = vol + vol_force * dt
        new_mean = mean + mean_adapt_rate * (level - mean) * dt
        
        return np.vstack([new_level, new_trend, new_vol, new_mean]).reshape(-1)

    def measurement_function(self, x):
        """
        Vectorized measurement function that returns only the levels.
        """
        return x.reshape(self.state_dim, self.n_dims)[0]  # Return levels only

    def initialize(self, initial_measurement):
        """
        Initialize state and covariance matrices.
        
        Args:
            initial_measurement: Array of shape (n_dims,)
        """
        # Initialize state vector
        self.x = np.zeros(self.total_dim)
        levels = initial_measurement
        trends = np.zeros(self.n_dims)
        vols = np.ones(self.n_dims) * 0.1
        means = initial_measurement.copy()
        
        self.x = np.hstack([levels, trends, vols, means])
        
        # Initialize covariance matrix
        scale = np.std(initial_measurement) if len(initial_measurement) > 1 else 1.0
        base_P = np.diag([0.1, 0.01, 0.001, 0.1]) * scale
        self.P = np.kron(np.eye(self.n_dims), base_P)
        
        # Initialize noise matrices
        base_Q = np.diag([0.01, 0.001, 0.001, 0.001]) * scale
        self.Q = np.kron(np.eye(self.n_dims), base_Q)
        self.R = np.eye(self.n_dims) * (0.1 * scale)

    def generate_sigma_points(self):
        """Generate sigma points for the entire state vector"""
        sigma_points = np.zeros((2 * self.total_dim + 1, self.total_dim))
        sigma_points[0] = self.x
        
        # try:
        #     U = sqrtm((self.total_dim + self.lambda_) * self.P)
        #     U = np.real(U)  # Ensure real values
        # except:
        #     # Fallback to diagonal if decomposition fails
        #     U = np.diag(np.sqrt(np.diag((self.total_dim + self.lambda_) * self.P)))

        try:
            U = np.linalg.cholesky((self.total_dim + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # Fallback to diagonal if decomposition fails
            U = np.diag(np.sqrt(np.diag((self.total_dim + self.lambda_) * self.P)))
        
        for i in range(self.total_dim):
            sigma_points[i + 1] = self.x + U[i]
            sigma_points[self.total_dim + i + 1] = self.x - U[i]
            
        return sigma_points

    def predict(self):
        """Prediction step"""
        sigma_points = self.generate_sigma_points()
        self.chi = np.apply_along_axis(self.process_function, 1, sigma_points)
        
        self.x_pred = self.weights_m @ self.chi  # Weighted mean
        
        diff = self.chi - self.x_pred
        self.P_pred = diff.T @ np.diag(self.weights_c) @ diff + self.Q

    def update(self, measurement):
        """
        Update step
        
        Args:
            measurement: Array of shape (n_dims,)
        """
        mask = ~np.isnan(measurement)
        if not np.any(mask):  # No valid measurements
            self.x = self.x_pred
            self.P = self.P_pred
            return self.x.reshape(self.state_dim, self.n_dims)[0]
        
        gamma = np.apply_along_axis(self.measurement_function, 1, self.chi)
        y_pred = self.weights_m @ gamma  # Weighted mean of measurements
        
        diff_y = gamma - y_pred
        diff_x = self.chi - self.x_pred
        
        Pyy = diff_y.T @ np.diag(self.weights_c) @ diff_y + self.R
        Pxy = diff_x.T @ np.diag(self.weights_c) @ diff_y
        
        if not np.all(mask):  # Partial measurements
            Pyy = Pyy[np.ix_(mask, mask)]
            Pxy = Pxy[:, mask]
            innovation = measurement[mask] - y_pred[mask]
        else:
            innovation = measurement - y_pred
        
        K = Pxy @ np.linalg.inv(Pyy)
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pyy @ K.T
        
        return self.x.reshape(self.state_dim, self.n_dims)[0]


def robust_ukf(data):
    """
    This applies an unscented Kalman filter to the data.
    This reduces Gaussian noise and fills in blanks (NaNs).
    It is vectorised, so data must have shape = (num_timesteps, num_dimensions).
    E.g. for a single timeseries, you must add a singleton dimension so shape = (num_timesteps, 1).
    
    Args:
        data: Array of shape (timesteps, n_dims)
    
    Returns:
        Filtered data of same shape.
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (timesteps, n_dims)")
        
    timesteps, n_dims = data.shape
    
    # Initialize filter with first non-NaN measurement for each dimension
    first_valid = np.zeros(n_dims)
    for d in range(n_dims):
        valid_data = data[:, d][~np.isnan(data[:, d])]
        first_valid[d] = valid_data[0] if len(valid_data) > 0 else 0.0
    
    ukf = FinancialUKF(n_dims=n_dims)
    ukf.initialize(first_valid)
    
    filtered = np.zeros_like(data)
    for t in range(timesteps):
        ukf.predict()
        filtered[t] = ukf.update(data[t])
        
    return filtered