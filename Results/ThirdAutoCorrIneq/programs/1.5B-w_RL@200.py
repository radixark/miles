
# New improvement to function generator adding controlled variability with refined boundary handling
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
    n = 300
    
    num_iterations = 5000
    max_step_initial = 0.3
    max_step_final = 0.04
    x = np.linspace(-0.15, 0.15, n)
    support_width = 0.06  # Narrower window for focus optimization
    
    # Initialize function values
    center = 0.0
    wavelet_width = 0.15
    
    # Base mother wavelet with precise support
    # Base window function with more controlled modification
    heights = np.exp(-((x)**2)/(2*(0.2)**2)) 

    # Modified Gaussian component with random amplitude variations
    # Adjust Frequency Parameters and Variations
    freq1 = 10.0
    A = 0.15
    B= 0.015
    heights += 0.3 * np.sin(2 * np.pi * 13*x) + 0.2*(0.15*np.exp(-(-x**2)/ (0.1)**2 )) + 0.1*np.exp(2 * np.pi* 9.8*x) + 0.15*np.sin(2 * np.pi * 11*x)
    heights += A * np.exp( - (2*np.pi*10*x)**2 ) + B * 0.5 + 0.01*B * (0.01 + 0.005*np.random.randn(len(x)))
    
    # Modulation add term with square wave variations
    term2 = 0.15 * np.cos( 2*np.pi*(20*x) + 0.2*np.random.randn(len(x))*0.5 )
    heights += term2
    
    # Modified Mexican hat wavelet with controlled parameters
    heights += 0.3 * ( (abs(x) <= 0.015) * 2 - 0.5 ) * 1.3 * 2  + 0.2*(abs(x>0.01) * (0.1))  # Tighter window
    heights += 0.3*(abs(x>0.0075))*(0.2)* 1.2 + 0.15*np.cos(2*np.pi*9.5*x + np.random.randn(len(x)))+ 0.08*np.sin(2*np.pi*12*x + 0.1*np.random.randn(x.size))
    
    # Add frequency components with controlled variations
    freq1 = 10.0
    heights += 0.3 * np.cos(2 * np.pi * freq1*x + np.random.randn(1))
    freq2 = 15.0
    heights += 0.15 * np.sin(2 * np.pi * freq2*x)
    
    # Redistribution of heights
    heights /= np.sum(heights) + 0.01
    target_sum = 28.0
    
    # Improved Handling introducing random variations
    heights = heights * (target_sum / np.sum(heights)) + 0.2 * (0.4*(np.exp(-(x**2/0.15)**2)) + 0.6*(np.exp(-( (x-0.2)**2)/0.15**2)) + 0.1*(np.sin(2*np.pi* x + 0.1*np.random.randn(x.size)) + 0.05*(np.random.randn(x.size) * 0.1)))
    
    # Adaptive normalization with base scaling
    target_sum = 26.0
    heights = heights * (target_sum / np.sum(heights)) + 0.2 * (0.4*(np.exp(-(x**2/0.15)**2)) + 0.6*(np.exp(-( (x-0.2)**2)/0.15**2)))
    
    # Normalize heights post adaptive steps
    target_sum = 26.0
    heights = heights / np.sum(heights) * target_sum
    
    return heights, compute_c3(heights)



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
    
    conv_full = np.convolve(heights, heights, mode='full')
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