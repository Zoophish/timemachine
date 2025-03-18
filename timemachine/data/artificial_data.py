# TODO: Make some linear and non-linear time series that a progressively more 'complicated' to show that the models are at least working.

import numpy as np

def linear(n, slope=1.0, intercept=0.0, noise_level=0.0):
    """Simple linear function: f(x) = mx + b + noise"""
    x = np.arange(n)
    return slope * x + intercept + np.random.normal(0, noise_level, n)

def quadratic(n, a=1.0, b=1.0, c=0.0, noise_level=0.0):
    """Quadratic function: f(x) = ax² + bx + c + noise"""
    x = np.arange(n)
    return a * x**2 + b * x + c + np.random.normal(0, noise_level, n)

def exponential(n, base=2.0, scale=1.0, noise_level=0.0):
    """Exponential growth: f(x) = scale * base^x + noise"""
    x = np.arange(n)
    return scale * base**x + np.random.normal(0, noise_level, n)

def logistic(n, k=1.0, x0=0.0, L=1.0, noise_level=0.0):
    """Logistic function: f(x) = L / (1 + e^(-k(x-x0))) + noise"""
    x = np.arange(n)
    return L / (1 + np.exp(-k * (x - x0))) + np.random.normal(0, noise_level, n)

def sinusoidal(n, amplitude=1.0, frequency=0.1, phase=0.0, noise_level=0.0):
    """Sinusoidal function: f(x) = A * sin(2πfx + φ) + noise"""
    x = np.arange(n)
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + np.random.normal(0, noise_level, n)

def compound_sine(n, amplitudes=[1.0, 0.5, 0.25], frequencies=[0.1, 0.2, 0.4], noise_level=0.0):
    """Compound sinusoidal with multiple frequencies"""
    x = np.arange(n)
    result = np.zeros(n)
    for a, f in zip(amplitudes, frequencies):
        result += a * np.sin(2 * np.pi * f * x)
    return result + np.random.normal(0, noise_level, n)

def damped_oscillator(n, amplitude=1.0, frequency=0.1, decay=0.05, noise_level=0.0):
    """Damped oscillator: f(x) = A * e^(-dx) * sin(2πfx) + noise"""
    x = np.arange(n)
    return amplitude * np.exp(-decay * x) * np.sin(2 * np.pi * frequency * x) + np.random.normal(0, noise_level, n)

def logistic_with_seasonality(n, k=1.0, x0=0.0, L=1.0, amplitude=0.1, frequency=0.1, noise_level=0.0):
    """Logistic growth with seasonal pattern"""
    x = np.arange(n)
    logistic_part = L / (1 + np.exp(-k * (x - x0)))
    seasonal_part = amplitude * np.sin(2 * np.pi * frequency * x)
    return logistic_part + seasonal_part + np.random.normal(0, noise_level, n)

def step_function(n, step_points=[0.25, 0.5, 0.75], values=[1, 2, 3, 4], noise_level=0.0):
    """Piecewise constant function with multiple steps"""
    x = np.arange(n)
    result = np.zeros(n)
    steps = [int(p * n) for p in step_points]
    current_value = values[0]
    for i in range(len(steps)):
        result[x < steps[i]] = values[i]
    result[x >= steps[-1]] = values[-1]
    return result + np.random.normal(0, noise_level, n)

def chaotic_logistic(n, r=3.9, x0=0.5, noise_level=0.0):
    """Chaotic logistic map: x_{n+1} = rx_n(1-x_n) + noise"""
    result = np.zeros(n)
    result[0] = x0
    for i in range(1, n):
        result[i] = r * result[i-1] * (1 - result[i-1])
    return result + np.random.normal(0, noise_level, n)

def nonlinear_autoregressive(n, lookback=3, complexity=2, initial_values=None, noise_level=0.0, scaling=1.0):
    """
    Generates a time series where each value depends non-linearly on previous values.
    
    Parameters:
    n (int): Length of the sequence to generate
    lookback (int): How many previous values to consider
    complexity (int): Degree of polynomial interactions (1=linear, 2=quadratic, 3=cubic, etc.)
    initial_values (array-like): Starting values, defaults to random if None
    noise_level (float): Standard deviation of Gaussian noise to add
    scaling (float): Scaling factor to prevent explosion/vanishing
    
    Returns:
    numpy array of length n
    """
    result = np.zeros(n)
    
    # Initialize starting values
    if initial_values is None:
        result[:lookback] = np.random.random(lookback)
    else:
        result[:lookback] = initial_values[:lookback]
    
    # Generate subsequent values
    for i in range(lookback, n):
        # Get previous values
        prev_values = result[i-lookback:i]
        
        # Create non-linear terms up to specified complexity
        terms = []
        for degree in range(1, complexity + 1):
            # Generate all possible combinations of previous values
            # raised to powers that sum to 'degree'
            for combo in range(lookback):
                terms.append(np.power(prev_values[combo], degree))
            # Add interaction terms for degree > 1
            if degree > 1:
                for j in range(lookback):
                    for k in range(j+1, lookback):
                        terms.append(prev_values[j] * prev_values[k])
        
        # Combine terms with random weights
        if i == lookback:  # Generate weights only once
            weights = np.random.random(len(terms)) - 0.5
        
        # Calculate next value
        next_value = np.sum(weights * terms) 
        result[i] = scaling * np.tanh(next_value) + np.random.normal(0, noise_level)
    
    return result

def brownian_series(n, std=1):
    out = np.zeros((n,), dtype=np.float32)
    r = np.random.normal(out, std)
    return np.cumsum(r)
