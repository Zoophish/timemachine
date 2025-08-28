# a lot of these are written by an LLM

import numpy as np
from scipy.fft import fft, ifft

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

def normal_noise(n, variance):
    return np.random.normal(0, variance)

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


def fractional_brownian_motion_hurst_array(n, hurst_array, scale=1.0):
    """
    Generate fractional Brownian motion (fBm) with time-varying Hurst exponent.
    
    This function creates a time series where the fractal properties and long-range 
    dependence characteristics vary over time according to the provided Hurst exponent array.
    
    Parameters:
    n (int): Length of the time series to generate
    hurst_array (array-like): Array of Hurst exponents, must have length n
                             Values should be between 0 and 1:
                             - H < 0.5: Anti-persistent (mean-reverting)
                             - H = 0.5: Standard Brownian motion (memoryless)
                             - H > 0.5: Persistent (trending)
    scale (float): Scaling factor for the output amplitude
    
    Returns:
    numpy array of length n representing the fBm time series
    
    Notes:
    - Uses the Davies-Harte method for generating fBm segments
    - Smoothly transitions between different Hurst regimes
    - Computationally intensive for very long series due to segment-wise generation
    """
    hurst_array = np.asarray(hurst_array)
    if len(hurst_array) != n:
        raise ValueError(f"Hurst array length {len(hurst_array)} must match series length {n}")
    
    if np.any(hurst_array <= 0) or np.any(hurst_array >= 1):
        raise ValueError("Hurst exponents must be between 0 and 1 (exclusive)")
    
    # Initialize the result array
    result = np.zeros(n)
    
    # For time-varying H, we'll use a windowed approach
    # Generate overlapping segments and blend them
    window_size = min(128, n // 4)  # Adaptive window size
    overlap = window_size // 2
    
    if window_size < 16:
        window_size = n
        overlap = 0
    
    # Generate the series using overlapping windows
    positions = np.arange(0, n - window_size + 1, window_size - overlap)
    if positions[-1] + window_size < n:
        positions = np.append(positions, n - window_size)
    
    weights = np.zeros(n)
    
    for pos in positions:
        end_pos = min(pos + window_size, n)
        segment_length = end_pos - pos
        
        # Use average Hurst exponent for this segment
        avg_hurst = np.mean(hurst_array[pos:end_pos])
        
        # Generate fBm segment using Davies-Harte method
        segment = _generate_fbm_segment(segment_length, avg_hurst)
        
        # Apply window weighting (Hann window for smooth blending)
        if len(positions) > 1:
            window_weights = np.hanning(segment_length)
        else:
            window_weights = np.ones(segment_length)
        
        # Add to result with proper weighting
        result[pos:end_pos] += segment * window_weights
        weights[pos:end_pos] += window_weights
    
    # Normalize by weights to handle overlaps
    weights[weights == 0] = 1  # Avoid division by zero
    result /= weights
    
    # Apply local Hurst-dependent scaling
    for i in range(1, n):
        local_hurst = hurst_array[i]
        # Adjust the increment based on local Hurst exponent
        dt_scaling = np.power(1.0, local_hurst)
        if i > 0:
            increment_scale = 0.1 * (2 * local_hurst - 1)  # Small adjustment based on H
            result[i] += increment_scale * result[i-1]
    
    return result * scale

def _generate_fbm_segment(n, hurst):
    """
    Generate a single fBm segment using the Davies-Harte method.
    
    Parameters:
    n (int): Length of the segment
    hurst (float): Hurst exponent for this segment
    
    Returns:
    numpy array of length n
    """
    # Extend to next power of 2 for efficient FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    
    # Generate the covariance sequence
    k = np.arange(n_fft)
    # Autocovariance function for fBm: R(k) = 0.5 * (|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H))
    R = np.zeros(n_fft)
    
    for i in range(n_fft):
        if i == 0:
            R[i] = 1.0
        else:
            R[i] = 0.5 * (np.abs(i-1)**(2*hurst) - 2*np.abs(i)**(2*hurst) + np.abs(i+1)**(2*hurst))
    
    # Create circulant matrix eigenvalues
    R_extended = np.concatenate([R, R[1:-1][::-1]])
    
    # Get eigenvalues via FFT
    eigenvals = np.real(fft(R_extended))
    
    # Handle numerical issues
    eigenvals = np.maximum(eigenvals, 0)
    
    # Generate complex Gaussian random variables
    Z = np.random.normal(0, 1, len(eigenvals)) + 1j * np.random.normal(0, 1, len(eigenvals))
    Z[0] = np.real(Z[0])  # First element should be real
    Z[len(eigenvals)//2] = np.real(Z[len(eigenvals)//2])  # Middle element should be real
    
    # Ensure conjugate symmetry for real output
    Z[len(eigenvals)//2+1:] = np.conj(Z[1:len(eigenvals)//2][::-1])
    
    # Generate the fBm
    fbm_full = np.real(ifft(np.sqrt(eigenvals) * Z))
    
    return fbm_full[:n]

def multifractal_series(n, hurst_base=0.7, intermittency=0.1, scale=1.0):
    """
    Generate a multifractal time series with varying local scaling properties.
    
    This creates a more complex alternative where the Hurst exponent varies
    based on multiplicative cascades, typical in financial and turbulence data.
    
    Parameters:
    n (int): Length of the time series
    hurst_base (float): Base Hurst exponent around which fluctuations occur
    intermittency (float): Strength of multifractal intermittency (0-1)
    scale (float): Overall amplitude scaling
    
    Returns:
    numpy array of length n
    """
    # Generate multiplicative cascade for varying volatility
    cascade = np.ones(n)
    level = 1
    
    while level < n:
        for i in range(0, n, level * 2):
            end_idx = min(i + level, n)
            # Random multiplicative factor
            factor = np.random.lognormal(0, intermittency)
            cascade[i:end_idx] *= factor
        level *= 2
    
    # Normalize cascade
    cascade /= np.mean(cascade)
    
    # Generate base fBm
    base_fbm = _generate_fbm_segment(n, hurst_base)
    
    # Apply multifractal modulation
    result = base_fbm * np.sqrt(cascade)
    

def challenging_nonlinear_fractal_series(n, complexity_level=3, noise_level=0.05, seed=None):
    """
    Generate an extremely challenging non-linear time series with varying fractal properties
    designed to test the limits of ML models.
    
    This combines:
    1. Multiple regime-switching dynamics
    2. Time-varying fractal behavior (Hurst exponents)
    3. Non-linear autoregressive components
    4. Chaotic dynamics with varying parameters
    5. Multiplicative noise and volatility clustering
    6. Hidden state-dependent transitions
    7. Multi-scale temporal dependencies
    
    Parameters:
    n (int): Length of the time series
    complexity_level (int): 1-5, controls how many challenging features to include
    noise_level (float): Base noise level (gets modulated throughout)
    seed (int): Random seed for reproducibility
    
    Returns:
    dict containing:
        'series': The main challenging time series
        'components': Dict of individual components for analysis
        'regime_indicators': Array indicating which regime is active
        'hurst_evolution': Time-varying Hurst exponents used
        'volatility': Time-varying volatility
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n)
    
    # === Component 1: Time-varying Hurst exponents ===
    # Create complex Hurst evolution with multiple scales
    hurst_slow = 0.5 + 0.3 * np.sin(2 * np.pi * t / (n / 3))  # Long-term trend
    hurst_fast = 0.1 * np.sin(2 * np.pi * t / (n / 20))       # Medium-term oscillations
    hurst_chaotic = 0.05 * np.sin(17 * t / n) * np.cos(31 * t / n)  # Complex interactions
    hurst_jumps = np.zeros(n)
    
    # Add sudden Hurst regime changes
    jump_points = np.random.choice(n, size=max(3, n//200), replace=False)
    jump_points.sort()
    for jp in jump_points:
        jump_size = np.random.uniform(-0.2, 0.2)
        hurst_jumps[jp:] += jump_size
    
    hurst_evolution = np.clip(hurst_slow + hurst_fast + hurst_chaotic + hurst_jumps, 0.1, 0.9)
    
    # === Component 2: Regime-switching base dynamics ===
    # Hidden Markov-style regime switching
    n_regimes = min(4, complexity_level + 1)
    regime_probs = np.random.dirichlet(np.ones(n_regimes))
    transition_matrix = np.random.dirichlet(np.ones(n_regimes), size=n_regimes)
    
    regimes = np.zeros(n, dtype=int)
    current_regime = 0
    
    for i in range(1, n):
        # State-dependent transition probabilities
        transition_strength = 0.95 + 0.04 * np.sin(i / n * 10)  # Time-varying persistence
        if np.random.random() > transition_strength:
            current_regime = np.random.choice(n_regimes, p=transition_matrix[current_regime])
        regimes[i] = current_regime
    
    # === Component 3: Regime-dependent dynamics ===
    base_series = np.zeros(n)
    
    for regime in range(n_regimes):
        mask = regimes == regime
        regime_length = np.sum(mask)
        
        if regime_length == 0:
            continue
            
        if regime == 0:  # Trending regime
            regime_series = np.cumsum(np.random.normal(0.1, 0.5, regime_length))
        elif regime == 1:  # Mean-reverting regime  
            regime_series = np.zeros(regime_length)
            for i in range(1, regime_length):
                regime_series[i] = -0.3 * regime_series[i-1] + np.random.normal(0, 0.3)
        elif regime == 2:  # Chaotic regime
            r = 3.7 + 0.2 * np.random.random()  # Random chaotic parameter
            regime_series = np.zeros(regime_length)
            regime_series[0] = np.random.random()
            for i in range(1, regime_length):
                regime_series[i] = r * regime_series[i-1] * (1 - regime_series[i-1])
            regime_series = (regime_series - 0.5) * 4  # Scale and center
        else:  # Oscillatory regime with varying frequency
            freq = 0.1 + 0.2 * np.random.random()
            phase = 2 * np.pi * np.random.random()
            regime_t = np.arange(regime_length)
            regime_series = np.sin(2 * np.pi * freq * regime_t + phase)
            # Add frequency modulation
            freq_mod = 1 + 0.5 * np.sin(regime_t / regime_length * 8)
            regime_series *= freq_mod
        
        base_series[mask] = regime_series
    
    # === Component 4: Non-linear autoregressive overlay ===
    if complexity_level >= 2:
        nar_series = np.zeros(n)
        nar_series[:3] = np.random.normal(0, 0.1, 3)
        
        for i in range(3, n):
            # Time-varying non-linear coefficients
            coef1 = 0.5 * np.sin(i / n * 6)
            coef2 = 0.3 * np.cos(i / n * 4) 
            coef3 = 0.1 * np.sin(i / n * 12)
            
            prev_vals = nar_series[i-3:i]
            nonlinear_term = (coef1 * prev_vals[0] * np.tanh(prev_vals[1]) + 
                            coef2 * prev_vals[1]**2 * np.sign(prev_vals[2]) +
                            coef3 * prev_vals[0] * prev_vals[2])
            
            nar_series[i] = 0.7 * nonlinear_term + np.random.normal(0, 0.1)
        
        base_series += 0.5 * nar_series
    
    # === Component 5: Fractal noise with time-varying properties ===
    fractal_component = fractional_brownian_motion_hurst_array(n, hurst_evolution, scale=1.0)
    
    # === Component 6: Multiplicative volatility clustering ===
    if complexity_level >= 3:
        volatility = np.ones(n)
        vol_persistence = 0.9
        vol_innovations = np.random.normal(0, 0.1, n)
        
        for i in range(1, n):
            # GARCH-like volatility with regime dependence
            regime_vol_factor = 1.0 + 0.5 * regimes[i] / n_regimes
            volatility[i] = (vol_persistence * volatility[i-1] + 
                           (1-vol_persistence) * (np.abs(vol_innovations[i]) + 0.1) * regime_vol_factor)
        
        volatility = np.sqrt(volatility)  # Convert to standard deviation
    else:
        volatility = np.ones(n)
    
    # === Component 7: Multi-scale jumps ===
    if complexity_level >= 4:
        jump_component = np.zeros(n)
        # Large jumps (rare)
        large_jump_times = np.random.poisson(n/100, 1)[0]
        large_jump_indices = np.random.choice(n, size=min(large_jump_times, n//10), replace=False)
        jump_component[large_jump_indices] += np.random.normal(0, 2, len(large_jump_indices))
        
        # Medium jumps (less rare)
        med_jump_times = np.random.poisson(n/50, 1)[0]
        med_jump_indices = np.random.choice(n, size=min(med_jump_times, n//5), replace=False)
        jump_component[med_jump_indices] += np.random.normal(0, 0.5, len(med_jump_indices))
        
        base_series += jump_component
    
    # === Component 8: Long-range dependent noise ===
    if complexity_level >= 5:
        # Additional fBm with different Hurst
        hurst_noise = 0.3 + 0.4 * np.sin(t / n * 8)
        lrd_noise = fractional_brownian_motion_hurst_array(n, hurst_noise, scale=0.3)
        base_series += lrd_noise
    
    # === Final assembly ===
    # Combine all components with time-varying weights
    weight_evolution = 1 + 0.3 * np.sin(t / n * 7) * np.cos(t / n * 13)
    
    final_series = (base_series * weight_evolution + 
                   0.4 * fractal_component + 
                   noise_level * np.random.normal(0, volatility, n))
    
    # Add subtle non-stationarity
    trend_component = 0.1 * t / n * np.sin(t / n * 3)
    final_series += trend_component
    
    # Package results
    components = {
        'base_dynamics': base_series,
        'fractal_noise': fractal_component,
        'trend': trend_component,
        'volatility_factor': volatility,
    }
    
    if complexity_level >= 2:
        components['nonlinear_ar'] = nar_series
    if complexity_level >= 4:
        components['jumps'] = jump_component
    if complexity_level >= 5:
        components['lrd_noise'] = lrd_noise
    
    return {
        'series': final_series,
        'components': components,
        'regime_indicators': regimes,
        'hurst_evolution': hurst_evolution,
        'volatility': volatility
    }