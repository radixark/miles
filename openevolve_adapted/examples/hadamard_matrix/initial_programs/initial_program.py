# EVOLVE-BLOCK-START
import numpy as np
import sys

def generate_hadamard_matrix(n: int) -> np.ndarray:
    """
    Generate a Hadamard matrix of size n using optimization methods.
    
    Args:
        n: Matrix size
        
    Returns:
        n x n matrix with entries +1 or -1
    """
    import numpy as np
    import random
    import time
    
    def det_bareiss(A):
        """Bareiss algorithm for exact integer determinant calculation."""
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

    def hill_climb_with_annealing(A, max_iters=2000, seed=None, temp0=0.5):
        """Hill climbing with simulated annealing for Hadamard matrix optimization."""
        rng = random.Random(seed)
        n = len(A)
        current_matrix = [row.copy() for row in A]
        det_curr = det_bareiss(current_matrix)
        best_det = det_curr
        best_matrix = [row.copy() for row in current_matrix]

        for t in range(1, max_iters + 1):
            # Random flip
            i = rng.randrange(n)
            j = rng.randrange(n)
            old_val = current_matrix[i][j]
            current_matrix[i][j] = -old_val
            
            # Calculate new determinant
            d_new = det_bareiss(current_matrix)
            
            # Accept or reject based on simulated annealing
            accept = False
            if abs(d_new) >= abs(det_curr):
                accept = True
            else:
                # Annealing temperature schedule
                T = temp0 / (1.0 + t * 0.001)
                if T > 0 and rng.random() < np.exp((abs(d_new) - abs(det_curr)) / max(1.0, T * abs(det_curr))):
                    accept = True
            
            if accept:
                det_curr = d_new
                if abs(det_curr) > abs(best_det):
                    best_det = det_curr
                    best_matrix = [row.copy() for row in current_matrix]
                    if t % 100 == 0:
                        print(f"Hill climb iter {t}: New best |det| = {abs(best_det)}")
            else:
                current_matrix[i][j] = old_val  # Revert
                
            # Early termination for very good solutions
            theoretical_max = THEORETICAL_MAX
            if abs(best_det) > 0.8 * theoretical_max:
                print(f"Found high-quality solution at iteration {t}")
                break

        return np.array(best_matrix), best_det

    def create_structured_start(n):
        """Create a structured starting matrix for optimization."""
        # For N=29, use the exact same approach as original
        if n == 29:
            # Create structured starting point based on quadratic residues mod 29
            matrix = []
            for i in range(n):
                row = []
                for j in range(n):
                    # Use pattern based on (i-j)^2 mod 29 - exactly like original
                    diff = (i - j) % n
                    quadratic_residues = {1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}  # QR mod 29
                    if diff in quadratic_residues:
                        row.append(1)
                    else:
                        row.append(-1)
                matrix.append(row)
            return matrix
        
        # For other sizes, try general quadratic residue approach
        elif n > 2 and n % 4 == 1:
            try:
                # Calculate quadratic residues mod n (not n-1!)
                quadratic_residues = set()
                for i in range(1, n):
                    quadratic_residues.add((i * i) % n)
                
                matrix = []
                for i in range(n):
                    row = []
                    for j in range(n):
                        # Use pattern based on difference
                        diff = (i - j) % n
                        if diff in quadratic_residues:
                            row.append(1)
                        else:
                            row.append(-1)
                    matrix.append(row)
                return matrix
            except Exception as e:
                print(f"Error creating structured start: {e}")
        
        # Random starting point for other cases
        return np.random.choice([-1, 1], size=(n, n)).tolist()

    print(f"Generating Hadamard matrix for N={n}")
    
    best_matrix = None
    best_score = -float('inf')
    
    start_time = time.time()
    
    # Try multiple optimization runs with different starting points
    num_runs = min(5, max(1, 1000 // n))  # Adjust runs based on matrix size
    
    for run in range(num_runs):
        print(f"Optimization run {run + 1}/{num_runs}")
        
        # Create starting matrix
        if run == 0:
            start_matrix = create_structured_start(n)
        else:
            start_matrix = np.random.choice([-1, 1], size=(n, n)).tolist()
        
        # Optimize using hill climbing with annealing
        iters_per_run = max(1000, 2000 // num_runs)
        optimized_matrix, det_val = hill_climb_with_annealing(
            start_matrix, 
            max_iters=iters_per_run,
            seed=run * 42,
            temp0=0.5
        )
        
        # Calculate score (determinant ratio)
        abs_det = abs(det_val)
        score = abs_det / THEORETICAL_MAX if THEORETICAL_MAX > 0 else 0
        
        print(f"Run {run + 1}: Score = {score:.6f}, |det| = {abs_det:.0f}")
        
        if score > best_score:
            best_score = score
            best_matrix = optimized_matrix.copy()
            print(f"New best solution! Score: {best_score:.6f}")
        
        # Continue optimization if time allows
        elapsed = time.time() - start_time
        if elapsed > 240:  # Stop after 4 minutes
            break
    
    print(f"Final best score: {best_score:.6f}")
    return best_matrix

# EVOLVE-BLOCK-END

def initial_program():
    """
    Main entry point called by OpenEvolve for Hadamard Matrix problem.
    This function must remain unchanged.
    """
    matrix = generate_hadamard_matrix(MATRIX_SIZE)
    
    # Save search results using modular file_io
    if matrix is not None:
        # Convert numpy array to list for JSON serialization
        matrix_list = matrix.tolist() if hasattr(matrix, 'tolist') else matrix
        current_solution = {'data': matrix_list}
    else:
        current_solution = {'data': None}
        
    best_perfect = None  # No perfect solutions expected for most sizes
    
    # Save results with problem-specific metadata
    try:
        save_search_results(
            best_perfect,
            current_solution,
            matrix_size=MATRIX_SIZE,
            theoretical_max=THEORETICAL_MAX
        )
    except Exception as e:
        print(f"Failed to save search results: {e}")
    
    return matrix


if __name__ == "__main__":
    # Import modular functions
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config


    # Get parameters from config
    PROBLEM_CONFIG = get_current_problem_config()
    MATRIX_SIZE = PROBLEM_CONFIG['core_parameters']['matrix_size']
    THEORETICAL_MAX = PROBLEM_CONFIG['core_parameters']['theoretical_max']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']

    result = initial_program()
    print(f"Generated {PROBLEM_TYPE} solution for matrix size {MATRIX_SIZE}")
    
    # Output matrix in +/- format for verification
    print("MATRIX_START")
    if result is not None:
        for i in range(MATRIX_SIZE):
            row_str = ''.join('+' if x == 1 else '-' for x in result[i])
            print(row_str)