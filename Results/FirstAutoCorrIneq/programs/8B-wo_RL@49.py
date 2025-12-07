# EVOLVE-BLOCK-START
import numpy as np
import time
from openevolve.modular_utils.evaluation_controller import get_current_problem_config
from openevolve.modular_utils.file_io_controller import save_search_results




def search_for_best_sequence() -> list[float]:
  """Function to search for the best coefficient sequence."""
  # Fixed length for all runs (200)
  # We are using sequences of length 200
  n_sequences = 200
  # Start with uniform distribution: [1/200] * 200
  best_sequence = [1.0/200] * 200
  curr_sequence = best_sequence.copy()
  best_score = np.inf
  start_time = time.time()
  last_save = start_time
  initial_t = 0.1
  t = initial_t

  restarts_done = 0
  # Calculate initial_time_per_restart: at least 0.1 seconds, but not more than max_time/num_restarts
  # We let the user set MAX_TIME, and we calculate the initial time per restart and the number of restarts
  if MAX_TIME <= 0:
      num_restarts = 0
      max_time_per_restart = 0
  else:
      num_restarts = 10   # Increased from 5 to 10
      max_time_per_restart = max(10, (MAX_TIME * 0.9) // num_restarts)  # Use 90% of the total time
      print(f"Working with {num_restarts} restarts, each with at most {max_time_per_restart} seconds budget")

  restart_times_per_search = []  # Track time spent in each restart

  while restarts_done < num_restarts and time.time() - start_time < MAX_TIME:
    # Start with an all-ones vector for numerical stability
    # Then adjust with a small random vector to avoid stagnation
    if restarts_done == 0:
        n = 200
        # Central step function: set indices 50 to 149 (inclusive) to 1.0
        curr_sequence = [0.0] * n
        start_index = 50
        for i in range(start_index, start_index+100):
            if i < n:  # Avoid index out of bounds
                curr_sequence[i] = 1.0
    else:
        # For subsequent restarts, we either take a uniform random sequence or a perturbation from the best_sequence
        n = 200  # For consistency
        if best_sequence is None:
            # Then we use a uniform random sequence
            curr_sequence = [np.random.uniform(0, 1)] * n
            print("Using uniform random restart")
        else:
            # With 50% probability use uniform random, with 50% use a perturbation of best_sequence
            if np.random.rand() < 0.5:
                curr_sequence = [np.random.uniform(0, 1)] * n
                print("Using uniform random restart on subsequent")
            else:
                curr_sequence = best_sequence.copy()
                # Perturbation the best_sequence by adding a random vector with mean 0 and standard deviation 0.1
                perturbation = [np.random.rand() * 0.5 for _ in range(n)]
                # We want the perturbation to be larger to explore more
                curr_sequence = [max(0.0, x + p) for x, p in zip(curr_sequence, perturbation)]
                print("Using perturbed best_sequence restart")

    restart_start = time.time()
    # Avoid infinite loop by ensuring we make progress
    orig_score = evaluate_sequence(curr_sequence)
    max_iter = 1000  # Maximum iterations per restart
    for _ in range(max_iter):
        if time.time() - restart_start > MAX_TIME:
            break
        h_function = get_good_direction_to_move_into(curr_sequence)
        if h_function is None:
            # If LP fails, perturb randomly
            seq_index = np.random.randint(len(curr_sequence))
            curr_sequence[seq_index] = (curr_sequence[seq_index] + np.random.rand() * 0.1) % 1.0
        else:
            curr_sequence = h_function

        curr_score = evaluate_sequence(curr_sequence)
        if curr_score < orig_score and time.time() - restart_start < MAX_TIME:
            orig_score = curr_score

        # Update global best
        if curr_score < best_score:
            best_score = curr_score
            best_sequence = curr_sequence

        # Save at most once per restart
        if restarts_done == 0 and start_time + 20 < time.time():
            # Only save for restart 0 every 20 seconds
            save_checkpoint(best_sequence, best_score, time.time() - start_time)
            last_save = time.time()

    # Track memory for restart timing
    restart_times_per_search.append(time.time() - restart_start)

    restarts_done += 1

  print(f"Completed {restarts_done} restarts in {time.time() - start_time:.1f} seconds")
  # Balance any leftover time across restarts (roughly) but complete the last restart quickly
  
  # Return the best sequence found
  return best_sequence



# Next we implement the simple LP computation from [2]
# In this case we simply asked Gemini to convert the recipe in the paper to code

from scipy import optimize
linprog = optimize.linprog


def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
  """Returns the direction to move into the sequence."""
  n = len(sequence)
  sum_sequence = np.sum(sequence)
  normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
  rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
  g_fun = solve_convolution_lp(normalized_sequence, rhs)
  if g_fun is None:
    return None
  sum_sequence = np.sum(g_fun)
  normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
  t = 0.01
  new_sequence = [
      (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
  ]
  return new_sequence


def solve_convolution_lp(f_sequence, rhs):
  """Solves the convolution LP for a given sequence and RHS."""
  n = len(f_sequence)
  c = -np.ones(n)
  a_ub = []
  b_ub = []
  for k in range(2 * n - 1):
    row = np.zeros(n)
    for i in range(n):
      j = k - i
      if 0 <= j < n:
        row[j] = f_sequence[i]
    a_ub.append(row)
    b_ub.append(rhs)

  # Non-negativity constraints: b_i >= 0
  a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
  b_ub_nonneg = np.zeros(n)  # Zero vector

  a_ub = np.vstack([a_ub, a_ub_nonneg])
  b_ub = np.hstack([b_ub, b_ub_nonneg])

  result = linprog(c, A_ub=a_ub, b_ub=b_ub)

  if result.success:
    g_sequence = result.x
    return g_sequence
  else:
    print('LP optimization failed.')
    return None



def evaluate_sequence(sequence: list[float]) -> float:
  """
  Evaluates a sequence of coefficients with enhanced security checks.
  Returns np.inf if the input is invalid.
  """
  # --- Security Checks ---

  # Verify that the input is a list
  if not isinstance(sequence, list):
    return np.inf

  # Reject empty lists
  if not sequence:
    return np.inf

  # Check each element in the list for validity
  for x in sequence:
    # Reject boolean types (as they are a subclass of int) and
    # any other non-integer/non-float types (like strings or complex numbers).
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return np.inf

    # Reject Not-a-Number (NaN) and infinity values.
    if np.isnan(x) or np.isinf(x):
        return np.inf

  # Convert all elements to float for consistency
  sequence = [float(x) for x in sequence]

  # Protect against negative numbers
  sequence = [max(0, x) for x in sequence]

  # Protect against numbers that are too large
  sequence = [min(1000.0, x) for x in sequence]

  n = len(sequence)
  b_sequence = np.convolve(sequence, sequence)
  max_b = max(b_sequence)
  sum_a = np.sum(sequence)

  # Protect against the case where the sum is too close to zero
  if sum_a < 0.01:
    return np.inf

  return float(2 * n * max_b / (sum_a**2))

# EVOLVE-BLOCK-END

def save_checkpoint(sequence: list[float], score: float, elapsed: float):
  """Save current best sequence to file (overwrites previous save)."""
  try:
    current_solution = {'data': sequence}
    save_search_results(
        best_perfect_solution=None,
        current_solution=current_solution,
        target_value=TARGET_VALUE
    )
    print(f"[CHECKPOINT] t={elapsed:.1f}s, C1={score:.6f}")
  except Exception as e:
    print(f"[WARNING] Failed to save: {e}")


def run_c1_optimization():
    """Main entry point for C1 optimization"""
    heights = search_for_best_sequence()
    c1_value = evaluate_sequence(heights)

    print(f'Generated function C1 value: {c1_value:.6f}, data type: {type(heights)}, length: {len(heights)}')

    # Final save (overwrites the last checkpoint)
    save_checkpoint(heights, c1_value, MAX_TIME)

    return heights, c1_value


if __name__ == "__main__":
    
    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    TARGET_VALUE = PROBLEM_CONFIG['core_parameters']['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    MAX_TIME = PROBLEM_CONFIG['time_limit']  * 0.9  # use 90% of the allowed time
    print(f"MAX_TIME from config: {MAX_TIME} seconds")
    ###############################################
    
    heights, c1_value = run_c1_optimization()
    print(f"\nGenerated {PROBLEM_TYPE} function (constructor approach):")
    print(f"C1 value: {c1_value:.6f}")
    print(f"Target: {TARGET_VALUE} ({100*c1_value/TARGET_VALUE:.1f}% of target)")
    print(f"Function length: {len(heights)}")