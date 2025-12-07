
# EVOLVE-BLOCK-START
"""Gradient-based function optimization for C₃ autoconvolution"""

from scipy.optimize import minimize
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict


def compute_autocorrelation(f: np.ndarray) -> float:
    """
    Compute the autocorrelation at lag 3 (C3) using optimized computation.
    
    Args:
        f: 1D array of function values
        
    Returns:
        float: The C3 value
    """
    n = len(f)
    # Optimize by using smaller temporary arrays and vectorized operations
    # Compute convolution with higher numerical stability
    conv_full = np.convolve(f, f, mode='full')
    # C3 is defined as the maximum absolute value of the autocorrelation at lag 3?
    # But the problem says minimize C3, so we take the maximum absolute value.
    conv_full_abs = np.abs(conv_full)
    max_conv = np.max(conv_full_abs)
    total_sum = np.sum(f)
    
    # Avoid division by zero while handling cases where total_sum might be zero
    if total_sum == 0 or max_conv == 0:
        return float('inf')
        
    # Using analytical derivation: the C3 constant is defined as the maximum absolute value of the auto-correlation at lag 3 divided by (average power)^2?
    # Actually, in the context of this problem, the score is normalized by the square of the integral (total_sum) and we multiply by 2n.
    return 2 * n * max_conv / (total_sum ** 2)


def construct_function() -> Tuple[np.ndarray, float]:
    """
    Construct an optimized discrete function on [-1/4, 1/4] to minimize C₃.
    
    Uses a cosine-based initialization with compact support and 
    optimizes parameters A/(B,C) for the function form f(x) = A*(1 + B*cos(C*x))
    over a centered window.
    
    Returns:
        Tuple of (heights, c3_value)
        heights: np.array of shape (n,) with function values
        c3_value: The computed C₃ value
    """
    # Configuration parameters
    n = 400  # Function length
    target_sum = 25.0  # Target function integral
    
    # Create domain
    x = np.linspace(-0.25, 0.25, n)
    # Set the window to be the central 50% of the domain
    # The domain length is 0.5 (from -0.25 to 0.25)
    window_fraction = 0.5  # 50% of the domain
    window_length = 0.5 * window_fraction  # total window length in domain values
    window_start = -window_length / 2  # Start of the window (left boundary)
    window_end = window_length / 2     # End of the window (right boundary)
    
    # Define the initial values for the windowed cosine function
    A_init = 1.0
    B_init = 0.5
    C_init = 1.0
    
    # Create initial function: zero everywhere at first
    best_heights = np.zeros(n)
    
    # Create the initial function: inside the window, use A*(1+B*cos(2*pi*C*x)), else 0.
    in_window = (x >= window_start) & (x <= window_end)
    best_heights[in_window] = A_init * (1 + B_init * np.cos(2 * np.pi * C_init * x[in_window]))
    
    # Normalize the function to have the target integral (although the autocorrelation is scale-invariant, we keep the integral fixed for consistency)
    target_sum_value = target_sum
    total_sum = np.sum(best_heights)
    if total_sum != 0:
        scaling_factor = target_sum_value / total_sum
        best_heights = best_heights * scaling_factor
    
    best_c3 = compute_autocorrelation(best_heights)
    
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    # We are not recording parameters for the Mexican Hat, but we can keep the same structure
    best_params = (1, 1, 1)  # dummy, will be overwritten by optimization
    
    # Try different optimization methods
    for method in methods:
        try:
            print(f"Using {method} method...")
            options = {'maxiter': 10000, 'disp': False, 'ftol': 1e-9}
            
            # We'll use analytical gradient if possible... (but requires more computation)
            result = minimize(compute_autocorrelation, 
                              best_heights, 
                              method=method,
                              jac=None,
                              options=options)
            
            # Check if optimization improved the result
            if result.fun < best_c3:
                best_c3 = result.fun
                best_heights = result.x.copy()
                # We do not need to re-normalize because the autocorrelation is scale invariant.
                # But ensure the saved function is not NaN or Inf.
                # If after optimization the sum is zero, then the function is invalid.
                if np.sum(best_heights) == 0 or np.isnan(np.sum(best_heights)):
                    # Cannot normalize, but we use the result
                    pass
                else:
                    # Only re-normalize if we are very far from the current best?
                    # But the objective is scale invariant, so re-normalization is not critical.
                    # We leave it here for safety.
                    best_heights = best_heights * (target_sum / np.sum(best_heights))
        
        except Exception as e:
            print(f"Optimization with {method} failed: {str(e)}")
            continue
    
    # Compute final C3 value
    c3_value = best_c3
    print(f"Optimization completed. C3 value: {c3_value}")
    
    # Try different optimization methods
    for method in methods:
        try:
            print(f"Using {method} method...")
            options = {'maxiter': 10000, 'disp': False, 'ftol': 1e-9}
            
            # We'll use analytical gradient if possible... (but requires more computation)
            result = minimize(compute_autocorrelation, 
                              best_heights, 
                              method=method,
                              jac=None,
                              options=options)
            
            # Check if optimization improved the result
            if result.fun < best_c3:
                best_c3 = result.fun
                best_heights = result.x.copy()
                # We do not re-normalize the function because the C3 is scale invariant.
                # Also, to avoid numerical issues, if best_heights becomes zero, we leave it.
                if np.sum(best_heights) == 0 or np.isnan(np.sum(best_heights)):
                    # Leave it as is
                    pass
                else:
                    # You can optionally re-normalize if you want, but C3 is independent of the scale
                    # For safety, we leave it here but note that re-normalization is not nessessary.
                    # best_heights = best_heights * (target_sum / np.sum(best_heights))
                    pass
        
        except Exception as e:
            print(f"Optimization with {method} failed: {str(e)}")
            continue
    
    # Compute final C₃ value
    c3_value = best_c3
    print(f"Optimization completed. C3 value: {c3_value}")
    
    return best_heights, c3_value

def run_c3_optimization() -> None:
    """Main entry point for C₃ optimization"""
    heights, c3_value = construct_function()
    
    # Save results in accepted format
    current_solution = {'data': heights.tolist()}
    save_search_results(
        best_perfect_solution=None,
        current_solution=current_solution,
        target_value=TARGET_VALUE
    )
    
    # Check if we need to do local optimization first
    print("Doing local optimization...")
    # We run the optimization once with a focus on avoiding local minima by using a high iteration count
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    best_heights = heights.copy()
    best_c3 = c3_value

    for method in methods:
        try:
            print(f"Using {method} method...")
            options = {'maxiter': 10000, 'disp': False, 'ftol': 1e-9}
            # Minimize the C3 function
            result = minimize(compute_autocorrelation, best_heights, method=method, 
                              jac=None, options=options)
            if result.fun < best_c3:
                best_c3 = result.fun
                best_heights = result.x.copy()
                print(f"Improved C3 to: {best_c3:.6f}")
        except Exception as e:
            print(f"Optimization with {method} failed: {str(e)}")
    
    # Update the current best
    heights = best_heights
    c3_value = best_c3

    # Implement a simulated annealing escape mechanism and sign flipping to break local optima
    if c3_value > 1.46 and np.random.rand() < 0.3:  # 30% chance to escape
        print("\nDetected potential local minimum, doing simulated annealing...")
        from scipy.optimize import differential_evolution
        
        # Define a temperature schedule
        def temp_schedule(current_temp):
            return current_temp * 0.999  # Exponential decay

        # Set initial temperature
        current_temp = 1.0
        initial_temp = 10.0

        # Perturbation function: flip signs in a random segment or random walk
        def perturb(f):
            n = len(f)
            segment_length = int(n * 0.1)
            start_idx = np.random.randint(0, n - segment_length + 1)
            if np.random.rand() < 0.5:
                # Sign flipping in a segment
                f_perturbed = f.copy()
                f_perturbed[start_idx:start_idx+segment_length] = -f_perturbed[start_idx:start_idx+segment_length]
                return f_perturbed
            else:
                # Random walk perturbation: flip signs at each point with 0.5 probability
                mask = np.random.rand(n) < 0.5
                return f.copy() * (1 - 2 * mask.astype(float)).reshape((len(f),))
        
        # Create a deep copy of the current best to avoid altering original during annealing
        best_heights = heights.copy()
        best_energy = c3_value

        # Annealing process
        for i in range(200):  # 200 iterations for annealing
            new_heights = perturb(best_heights)
            new_energy = compute_autocorrelation(new_heights)
            delta_energy = new_energy - best_energy

            # Simulated annealing: accept worse solutions with exponential probability
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / current_temp):
                best_energy = new_energy
                best_heights = new_heights

            current_temp = temp_schedule(current_temp)

            # Periodically, restart with a new segment of sign flips (every 5 steps)
            if i % 5 == 0 and np.random.rand() < 0.2:
                best_heights = perturb(best_heights)  # This is a chance to do a big change

        # Update if better found
        if best_energy < c3_value:
            print(f"Improved C3 after escape: {best_energy:.6f}")
            c3_value = best_energy
            heights = best_heights.copy()
    
    return heights, c3_value


if __name__ == "__main__":
    
    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    TARGET_VALUE = PROBLEM_CONFIG['core_parameters']['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    ###############################################
    
    # Part 1: Print problem configuration
    print(f"Solving {PROBLEM_TYPE} with target C₃: {TARGET_VALUE}")
    print("Using constructor-based initialization")
    
    # Part 2: Optimize the function
    import time
    start_time = time.time()
    heights, c3_value = run_c3_optimization()
    end_time = time.time()
    optimization_time = end_time - start_time
    
    print(f"\nGenerated {PROBLEM_TYPE} function (optimized):")
    print(f"C₃ value: {c3_value:.6f}")
    print(f"Target: {TARGET_VALUE} ({100*c3_value/TARGET_VALUE:.2f}% of target)")
    print(f"Function length: {len(heights)}")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    
    # Part 3: Save solutions
    # Save the current perfect solution (the best one)
    current_solution = {
        "data": heights.tolist(),
        "c3_value": float(c3_value),
        "function_type": "optimized"
    }
    
    # Note: We are not given the best_perfect_solution, just saving the current one
    save_search_results(
        best_perfect_solution=None,
        current_solution=current_solution,
        target_value=TARGET_VALUE
    )