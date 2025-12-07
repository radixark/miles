
# EVOLVE-BLOCK-START
"""Constructor-based function for C₃ autoconvolution optimization"""
import numpy as np


def construct_function():
    """
    Construct a discrete function on [-1/4, 1/4] that attempts to minimize
    C₃ = 2·n_steps·max(|conv(f,f)|) / (∑f)²
    
    Returns:
        Tuple of (heights, c3_value)
        heights: np.array of shape (n,) with function values
        c3_value: The computed C₃ value
    """
    # Choose length - this is a design variable that evolution can explore
    # Reference range is 300-500, but we can try different values
    n = 400  # Starting point - evolution will explore other lengths
    
    # Initialize function values
    x = np.linspace(-0.25, 0.25, n)
    
    # Strategy: Start with a smooth bell-shaped profile
    # This is a simple starting point - evolution will improve this
    
    # Create a Gaussian-like envelope
    center = 0.0
    width = 0.1
    heights = np.exp(-((x - center) ** 2) / (2 * width ** 2))
    
    # Add some oscillatory components to explore phase cancellation
    freq1 = 8.0
    freq2 = 16.0
    heights += 0.3 * np.cos(2 * np.pi * freq1 * x)
    heights += 0.15 * np.sin(2 * np.pi * freq2 * x)
    
    # Normalize to have reasonable sum (around 20-30)
    target_sum = 25.0
    heights = heights * (target_sum / np.sum(heights))
    
    # Compute C₃ value
    c3_value = compute_c3(heights)
    
    return heights, c3_value



def compute_c3(heights):
    """
    Compute the C₃ autoconvolution constant for given function values.
    
    C₃ = 2·n_steps·max(|conv(f,f)|) / (∑f)²

    Alphaevolve use abs(2 * n * np.max(conv_full) / (np.sum(heights)**2))
    
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