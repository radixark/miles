
# Optimized parameter tracking settings
num_experiments = 5000
# Improved parameter configuration
parameters = [
    {"degree": [1, 2, 3]},
    {"coefficients": [[1, -0.5], [0.8, 1.2]]},
]
# Improved optimizer configuration
optimizer_params = {
    "patchwork_cube": True,
    "learning_rate": 0.005,
    "num_experiments": 5000,  # Increased number of trials
    "patchwork_feasibility": True,
    "patchwork_step": 0.001  # Reduced step size for smoother tuning
}

# Initialize new setup
n = 400
# Improved meta-optimizing setup
# Modified parameters for better optimization parameters
num_experiments = 50
window_width = 0.4
amplitude_base = 0.9
amplitude_sharp = 0.2
frequency_base = 1.0
frequency_sharp = 1.5
support_width = n * 0.2
regularization = 1e-5

# Initialize new setup
n = 400
# Improve noise variability in function addition process
support_width = n * 0.15  # Narrow support for deeper optimization precision
"""Constructor-based function for C₃ autoconvolution optimization"""
# Using adaptive wavelet optimization with more sophisticated filtering techniques

# Adaptive Multi-Stage Perturbations
# Ensure numpy import is specific
import numpy as np
np.sqrt

# Advanced perturbation strategy
# Increase learning dynamics
params = {
    "patchwork_cube": True,
    "learning_rate": 0.005, 
    "learning_step": 0.0001,  # Reduced step size learning rate
    "initial_step": 0.2     # Base step size
}


def construct_function():
    """
    Construct a discrete function using advanced wavelet and chirp signal approaches for optimized C3 autoconvolution minimization.
    Returns:
        Tuple of (heights, computed_c3)
        heights: np.array constructed from modified Weierstrass function and Morlet transform components
        mother and chirp components with optimized parameters
    """
    # Vary wavelet and chirp modulation for novel function construction
    # ... (rest of the function remains same)
    n = 400
    x = np.linspace(-0.25, 0.25, n)
    
    # Optimized frequency mod with refined parameters
    # More complex scaling based on quadratic stages
    # Implementing a quadratic variation for function shape
    # Modify frequency component here
    scale_x = 0.25 + (x)/n * (0.3)
    amplitude = 1.0 + 0.6*(1.0 + 0.3*(x > 0.05)*(1.5) + 0.05*(x > 0.15)*(1.5))
    frequency = 1.0 + 0.6*(1.0 + 0.3*(x > 0.05)*(1.5) + 0.05*(x**3)/(n**3) + 0.05*(x > 0.1)*(1.5))
    
    # Enhanced modulation including x² influence and added noise
    # Enhanced modulating factors
    # Amplitude modulation using quadratic term
    amplitude *= np.sqrt(1 + (x**2)/(n**2)*0.1) + np.random.normal(0, 0.02, n)
    frequency *= (1 + (0.1 * x**2)/n + 0.01*np.sin(2*np.pi*x) + 0.03*x**3/n**2)
    
    # Amplitude scaling and certain wavelet inspired approach
    mother = lambda x, scale: (1.0/np.sqrt(2.0))*(scale)**-0.5 * np.exp(-1j*(x/(scale/2)) - 0.5j**2*(x**2)/(scale/2)**2)
    heights = np.zeros_like(x)
    
    for i in range(n):
        scale = 0.2 + (i)/n*(1 + 0.1*i/n)
        heights[i] = np.abs(mother(x[i], scale)) * np.exp(1j * frequency[i] * x[i])
    
    # Window optimization scaling with adjusted normalization
    window_width = 0.2
    heights = heights * window_width / n * 1e-3 * np.ones_like(x)
    
    # Normalize to maintain meaningful amplitudes
    target_sum = 25.0
    heights = heights * target_sum / (1 + np.sum((heights**2)/1e-10))
    
    heights = heights * (target_sum / (np.sum(heights) + 1e-10))
    
    heights = heights * window_width / (n * 1e3)  # Adjusting scale for constant reduction
    return heights, compute_c3(heights)
    
    # Compute normalization bias if needed
    return heights, compute_c3(heights)
    
    # Apply window function with validated width
    window_width = 0.5
    heights *= window_width / n * np.ones_like(x)
    
    # Normalize to maintain meaningful amplitudes
    target_sum = 25.0
    heights = heights * (target_sum / (np.sum(heights) + 1e-10))
    
    # Compute C3 value as before
    c3_value = compute_c3(heights)
    
    return heights, c3_value



def compute_c3(heights):
    """
    Compute the C₃ autoconvolution constant for given function values.
    
    Implementation focusing on robust numerical evaluation to stabilize results.
    
    Args:
        heights: np.array of shape (n,) with function values
    
    Returns:
        float: The C₃ value
    """
    n = len(heights)
    
    conv_full = np.convolve(heights, heights)  # full mode
    c3 = abs(2 * n * np.max(conv_full) / (np.sum(heights)**2))

    return c3

# EVOLVE-BLOCK-END

def run_c3_optimization():
    """Main entry point for C₃ optimization"""
    heights, c3_value = construct_function()
    
    # Normalize control parameters precisely
    target_sum = 25.0
    heights = heights * (target_sum / (np.sum(heights) + 1e-10))
    
    # Implement adaptive perturb with controlled chaos
    heights = heights + 0.1 * np.cos(2 * np.pi * np.random.random(* heights.shape)) * (heights * 0.25)
    # Introduce added chaos controlled by sine wave
    heights = heights + 0.1 * np.sin(4 * np.pi * np.random.random(* heights.shape)) * (heights * 0.2)
    
    # Save results in accepted format
    current_solution = {'data': heights.tolist()}
    save_search_results(
        best_perfect_solution=None,
        current_solution=current_solution,
        target_value=TARGET_VALUE
    )
    
    return heights, c3_value


if __name__ == "__main__":
    
    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    TARGET_VALUE = PROBLEM_CONFIG['core_parameters']['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    ###############################################
    
    heights, c3_value = run_c3_optimization()
    print(f"\nGenerated {PROBLEM_TYPE} function (constructor approach):")
    print(f"C₃ value: {c3_value:.6f}")
    print(f"Target: {TARGET_VALUE} ({100*c3_value/TARGET_VALUE:.1f}% of target)")
    print(f"Function length: {len(heights)}")