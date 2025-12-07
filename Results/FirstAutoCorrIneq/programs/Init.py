# EVOLVE-BLOCK-START
import numpy as np
import time
from openevolve.modular_utils.evaluation_controller import get_current_problem_config
from openevolve.modular_utils.file_io_controller import save_search_results




def search_for_best_sequence() -> list[float]:
  """Function to search for the best coefficient sequence."""
  best_sequence = [np.random.random()] * np.random.randint(100,1000)
  curr_sequence = best_sequence.copy()
  best_score = np.inf
  start_time = time.time()
  last_save = start_time

  while time.time() - start_time < MAX_TIME:
    h_function = get_good_direction_to_move_into(curr_sequence)
    if h_function is None:
      curr_sequence[1] = (curr_sequence[1] + np.random.rand()) % 1
    else:
      curr_sequence = h_function

    curr_score = evaluate_sequence(curr_sequence)
    if curr_score < best_score:
      best_score = curr_score
      best_sequence = curr_sequence

    # Save every 20 seconds
    if time.time() - last_save >= 20:
      save_checkpoint(best_sequence, best_score, time.time() - start_time)
      last_save = time.time()

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