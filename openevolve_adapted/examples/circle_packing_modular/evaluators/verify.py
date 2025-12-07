"""
Circle Packing Validation Functions
"""

import numpy as np
from typing import List, Tuple, Union, Any
# Import standardized error constants
from openevolve.modular_utils.error_constants import ErrorCodes
from openevolve.modular_utils.evaluation_controller import get_current_problem_config

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    """
    Validate circle packing solution format and constraints
    
    Args:
        solution_data: Expected to be (centers, radii) tuple

    Returns:
        (is_valid, error_message)
    """
    try:
        # Get expected n_circles from core_parameters
        n_circles = PROBLEM_CONFIG['core_parameters']['n_circles']
        
        # Validate solution format
        if not isinstance(solution_data, (tuple, list)) or len(solution_data) != 2:
            return False, "Solution must be (centers, radii) tuple"
        
        centers, radii = solution_data
        
        # Convert to numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)
        
        # Validate shapes
        if centers.shape != (n_circles, 2):
            return False, f"Centers must be ({n_circles}, 2) array, got {centers.shape}"

        if radii.shape != (n_circles,):
            return False, f"Radii must be ({n_circles},) array, got {radii.shape}"

        
        # No need to check sum consistency - we recompute it ourselves
        
        # Check for NaN or infinite values
        if np.any(np.isnan(centers)) or np.any(np.isinf(centers)):
            return False, "Centers contain NaN or infinite values"
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            return False, "Radii contain NaN or infinite values"

        # Check if circles are within unit square
        for i, (center, radius) in enumerate(zip(centers, radii)):
            x, y = center
            if x - radius < -1e-6 or x + radius > 1 + 1e-6:
                return False, f"Circle {i} x-boundary violation: center=({x:.6f}, {y:.6f}), radius={radius:.6f}"
            if y - radius < -1e-6 or y + radius > 1 + 1e-6:
                return False, f"Circle {i} y-boundary violation: center=({x:.6f}, {y:.6f}), radius={radius:.6f}"
        
        # Check for overlaps
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-6:
                    return False, f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
        
        return True, "Valid circle packing"
        
    except Exception as e:
        print(f"ERROR: validation failed: {str(e)}")
        return False, f"Validation error: {str(e)}"


def compute_objective_value(solution_data: Any) -> float:
    """
    Recompute the true objective value from validated solution data
    This prevents cheating by ensuring we calculate the real sum_radii
    
    Args:
        solution_data: (centers, radii) tuple

    Returns:
        True objective value (sum of radii) if valid, 0.0 otherwise
    """
    try:
        if not isinstance(solution_data, (tuple, list)) or len(solution_data) != 2:
            return 0.0
        
        centers, radii = solution_data
        
        # Convert to numpy arrays
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)
        
        # First verify the solution is valid
        is_valid, _ = validate_solution(solution_data)
        if not is_valid:
            return 0.0  # Invalid solutions get 0 objective
        
        # Recompute the true sum of radii (anti-cheat)
        true_sum = float(np.sum(radii))
        
        return true_sum
        
    except Exception as e:
        print(f"Error computing objective: {e}")
        return 0.0


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Solution data file path")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        solution_data = json.load(f)
    
    is_valid, error_msg = validate_solution(solution_data)
    if is_valid:
        objective_value = compute_objective_value(solution_data)
        print(f"{objective_value}")
    else:
        print(f"INVALID: {error_msg}")