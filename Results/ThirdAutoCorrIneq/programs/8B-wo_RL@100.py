
# EVOLVE-BLOCK-START
"""Constructor-based function for C₃ autoconvolution optimization using product of cosine window and oscillatory component"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal.windows import kaiser


def construct_function():
    """
    Construct a discrete function on [-1/4, 1/4] that attempts to minimize
    C₃ = 2·n_steps·max(|conv(f,f)|) / (∑f)²
    
    Returns:
        Tuple of (heights, c3_value)
        heights: np.array of shape (n,) with function values
        c3_value: The computed C₃ value
    """
    # Choose length
    n = 400  # Fixed for now
    
    # Define time axis
    dx = 0.5 / (n-1)
    x = np.linspace(-0.25, 0.25, n)

    # Initialize the function with:
    from scipy.signal.windows import kaiser
    #   - Window function: smooth cosine window (support: [-0.125, 0.125])
    #   - Oscillatory component: 1 + cos(2πx)
    # Parameters:
    # Improved initial function: Use multiple frequencies and a cosine window
    n_bases = 4  # Number of oscillatory bases to add

    # Create a window function (Hamming window for the inner part, support: 50% of the domain)
    # Build a Kaiser window with beta=5.0
    half_width = 100   # We set this to 100 because n=400, so 0.25*n=100
    n_win = 2 * half_width + 1   # 201 points for the window
    central_index = n // 2
    
    # Use Kaiser window with beta=5.0
    from scipy.signal.windows import kaiser
    kaiser_window = kaiser(n_win, beta=5.0)
    
    # Apply the window function where it should be non-zero
    window = np.zeros(n)
    # Define the window interval for the function (central n_win points)
    start_idx = central_index - half_width
    end_idx = start_idx + n_win
    window[start_idx:end_idx] = kaiser_window

    # Create base part with base_amplitude
    base_amplitude = 0.5  # This represents the base level on top of the oscillation
    base_part = base_amplitude * window

    # Create multiple oscillatory bases with decreasing amplitudes
    initial_function = base_part.copy()
    for i in range(n_bases):
        # Frequency increases with each base
        freq = (i+1) * 0.25  # Starting from 0.25, 0.5, 0.75, ... cycles per unit domain
        # Amplitude decreases by factor 0.9 with each base
        amplitude = 0.5 * (0.9 ** i)
        # The next candidate has the same phase 0 for simplicity.
        base_i = window * amplitude * (1.0 + np.cos(2 * np.pi * freq * x))
        # But note: the base_window and the window above are the same? Let's use the same window for the base and the oscillatory part.
        # Actually, we are using the same window function for each base.
        initial_function += base_i

    # Normalize the initial function to have integral 1.0
    initial_function = initial_function / np.sum(initial_function) * 1.0

    # Set bounds just as before
    n_pts = len(initial_function)
    bounds = [(-1.0, 1.0) for _ in range(n_pts)]
    
    # Define constraint: sum of function must be 1.0
    def constraint(h):
        return np.sum(h) - 1.0
    
    # Adjust the C3 computation function to handle zero sum
    def compute_c3(heights):
        s = np.sum(heights)
        if abs(s) < 1e-12:
            return float('inf')
        n = len(heights)
        conv_full = np.convolve(heights, heights, mode='full')
        abs_conv = np.abs(conv_full)
        max_abs = np.max(abs_conv)
        return (2 * n * max_abs) / (s**2)
    
    # We are going to use SLSQP method, which requires the constraint
    # But note: our initial function is not exactly sum 1.0, so the constraint function will adjust it.

    # We'll set up the optimization with the constraint
    result = minimize(compute_c3, initial_function, method='SLSQP',
                      bounds=bounds,
                      constraints={'type': 'eq', 'fun': constraint},
                      options={'maxiter': 6000, 'disp': True})
    
    best_heights = result.x
    best_c3 = result.fun
    
    # Print final sum and C3
    print(f"Optimization finished: Final C3 = {best_c3:.6f}")
    print(f"Final sum is {np.sum(best_heights):.6f}")   # Should be very close to 1.0
    
    return best_heights, best_c3


def compute_c3(heights):
    """
    Compute the C₃ autoconvolution constant for given function values.
    
    C₃ = 2·n_steps·max(|conv(f,f)|) / (∑f)²

    Additionally computes the frequency magnitude to focus optimization.

    Args:
        heights: np.array of shape (n,) with function values
    
    Returns:
        float: The C₃ value
        list: Frequencies of top peaks in magnitude spectrum (from FFT)
    """
    n = len(heights)
    
    # Use convolution instead of autocorrelation for consistency with the definition
    conv_full = np.convolve(heights, heights, mode='full')
    abs_conv = np.abs(conv_full)
    max_abs = np.max(abs_conv)
    
    energy_sum = np.sum(heights)**2
    c3_value = (2 * n * max_abs) / energy_sum

    # Compute frequency spectrum for dominant peaks to guide optimization
    fourier = np.fft.fft(heights)
    magnitudes = np.abs(fourier)
    # Remove zeros and focus on unique features (keep only positive part and remove negatives) but we take real values
    # Keep 20 largest magnitudes
    magnitudes_sorted = np.sort(np.unique(magnitudes))[-20:] if len(magnitudes) > 1 else magnitudes
    
    return c3_value, list(magnitudes_sorted)

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