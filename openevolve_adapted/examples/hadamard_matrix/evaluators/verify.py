"""
Hadamard Matrix Verification Functions
"""

from typing import List, Tuple, Any
import numpy as np
from openevolve.modular_utils.evaluation_controller import get_current_problem_config

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def det_bareiss(A: List[List[int]]) -> int:
    """
    Bareiss algorithm for exact integer determinant calculation.
    """
    n = len(A)
    if n == 0:
        return 1
    M = [row.copy() for row in A]
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return M[-1][-1]


def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    """
    Validate solution format and basic constraints
    
    Args:
        solution_data: Hadamard matrix as numpy array or list

    Returns:
        (is_valid, error_message)
    """
    # Convert to numpy array if it's a list
    if isinstance(solution_data, list):
        try:
            matrix = np.array(solution_data)
        except Exception as e:
            print(f"Error converting solution to numpy array: {e}")
            return False, "Could not convert solution to numpy array"
    elif isinstance(solution_data, np.ndarray):
        matrix = solution_data
    else:
        return False, "Solution must be a numpy array or list"
    
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False, "Matrix must be square"
    
    n = matrix.shape[0]
    
    # Check if entries are ±1
    if not np.all(np.isin(matrix, [-1, 1])):
        return False, "Matrix entries must be +1 or -1"
    
    # Check if size matches expected problem size
    expected_n = PROBLEM_CONFIG['core_parameters'].get('matrix_size')
    if expected_n and n != expected_n:
        return False, f"Matrix size {n} does not match expected size {expected_n}"
    
    return True, ""


def compute_objective_value(solution_data: Any) -> float:
    """
    Compute objective value from solution
    
    THIS FUNCTION MUST RECOMPUTE THE OBJECTIVE FROM solution_data
    DO NOT trust any pre-computed objective values to prevent hacking
    
    Args:
        solution_data: Hadamard matrix as numpy array or list

    Returns:
        Objective value to maximize (determinant ratio to theoretical maximum)
    """
    # Convert to numpy array if needed
    if isinstance(solution_data, list):
        try:
            matrix = np.array(solution_data)
        except Exception as e:
            print(f"Error converting solution to numpy array: {e}")
            return 0.0
    elif isinstance(solution_data, np.ndarray):
        matrix = solution_data
    else:
        return 0.0
    
    # Validate basic constraints
    is_valid, _ = validate_solution(matrix)
    if not is_valid:
        return 0.0
    
    n = matrix.shape[0]
    
    try:
        # Use exact integer determinant calculation
        int_matrix = matrix.astype(int).tolist()
        det_exact = det_bareiss(int_matrix)
        abs_det = abs(det_exact)
        
        # Get theoretical maximum from config
        theoretical_max = PROBLEM_CONFIG['core_parameters'].get('theoretical_max')
        if theoretical_max is None:
            # Default theoretical maximum for Hadamard matrix is n^(n/2)
            theoretical_max = n**(n/2)
        
        # Return pure determinant ratio as objective value
        if theoretical_max > 0:
            ratio = abs_det / theoretical_max
        else:
            ratio = 0.0
        
        return float(ratio)
        
    except Exception as e:
        print(f"Error computing objective value: {e}")
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