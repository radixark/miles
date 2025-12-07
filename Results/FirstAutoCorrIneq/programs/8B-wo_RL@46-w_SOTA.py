# EVOLVE-BLOCK-START

# feel free to change the entire algorithm here to improve performance
import numpy as np
import time
from openevolve.modular_utils.evaluation_controller import get_current_problem_config
from openevolve.modular_utils.file_io_controller import save_search_results


# NOTE!! Here we give you the previous SOTA result for reference. You can continue improving it.
# previous sota result: get C3 1.5032, length 1319, np.array height_sequence_1
from ref.sota_alphaevolve2 import height_sequence_1
print(f"length of previous SOTA function: {len(height_sequence_1)}") # should be 1319, the same data type as heights below
# you don't have to remove this sota import code, you also don't have to use it. Just as reference.



import numpy as np

def candidate_random(sequence: list[float], jitter_rate: float = 0.1) -> list[float]:
  """Generates a random perturbation of the given sequence by changing 3 random indices."""
  n_seq = len(sequence)
  # Choose 3 distinct indices to change
  indices = np.random.choice(n_seq, 3, replace=False)
  new_seq = sequence.copy()
  for idx in indices:
    d = np.random.randn() * jitter_rate   # Use normal distribution for perturbation
    new_val = max(0, new_seq[idx] + d)
    # Bound the value
    if new_val > 1000:
        new_val = 1000.0
    new_seq[idx] = new_val
  return new_seq

def search_for_best_sequence() -> list[float]:
  """Function to search for the best coefficient sequence using Simulated Annealing."""
  best_sequence = height_sequence_1.copy()
  curr_sequence = best_sequence.copy()
  best_score = np.inf

  # Evaluate initial solution
  if curr_sequence is not None and len(curr_sequence) > 0:
      curr_score = evaluate_sequence(curr_sequence)
      best_score = curr_score
  else:
      # This should ideally not happen, but for safety
      return []

  start_time = time.time()
  last_save = start_time

  # Set initial parameters
  MAX_NO_IMPROVEMENT = 1000
  initial_temperature = 1000.0
  cooling_rate = 0.95
  jitter_rate = 0.1  # Controls the magnitude of random perturbation

  # If initial sequence is the best, update best_sequence
  if best_score == np.inf:
    best_sequence = curr_sequence.copy()
    best_score = curr_score

  # Main loop
  num_steps_without_improvement = 0
  max_steps = int(MAX_TIME)
  # We'll run for a fixed number of steps or until time runs out
  # Calculate time per step estimate (one evaluation per step)
  time_per_step = 0.01  # Estimated time per function evaluation in seconds
  max_steps = 10000  # Limited by time

  temperature = initial_temperature

  while time.time() - start_time < MAX_TIME:
    # Decide to use a directed move (LP-based) or random move
    # With higher probability for LP-based moves (0.9) and random with 0.1
    use_lp = np.random.random() < 0.5

    if use_lp:
      h_direction = get_good_direction_to_move_into(curr_sequence)
      if h_direction is None:
        # If LP fails, use random move
        new_sequence = candidate_random(curr_sequence, jitter_rate)
      else:
        # Define the function for f(t)
        def f(t):
          new_seq = [(1-t)*curr_sequence[i] + t*h_direction[i] for i in range(len(curr_sequence))]
          return evaluate_sequence(new_seq)

        # Perform golden-section search with 100 evaluations
        low = 0.0
        high = 1.0
        n_evals = 30
        t_low = low + (high - low) * 0.333
        t_high = low + (high - low) * 0.666

        f_low = f(t_low)
        f_high = f(t_high)

        for _ in range(n_evals):
          if f_low < f_high:
            high = t_high
            t_high = t_low
            t_low = low + (high - low) * 0.333
            f_high = f_low
            f_low = f(t_low)
          else:
            low = t_low
            t_low = t_high
            t_high = low + (high - low) * 0.666
            f_low = f_high
            f_high = f(t_high)

        if f_low < f_high:
          t_opt = t_low
        else:
          t_opt = t_high

        new_sequence = [(1-t_opt)*curr_sequence[i] + t_opt*h_direction[i] for i in range(len(curr_sequence))]
    else:
      new_sequence = candidate_random(curr_sequence, jitter_rate)

    # Evaluate the new sequence
    new_score = evaluate_sequence(new_sequence)

    # Define the change in score (new_score is better than curr_score if new_score < curr_score)
    delta_score = new_score - curr_score  # new_score is better if delta_score < 0

    # Simulated Annealing acceptance criterion
    if new_score < curr_score:
      # If better, always accept
      accept = True
    else:
      # Allow worse solutions with decreasing probability
      # Metropolis criterion: probability is exp(-delta_score / temperature)
      p_accept = np.exp(-delta_score / temperature) if temperature > 0 else 0.5
      accept = (np.random.rand() < p_accept)  # Uniform random number < p_accept

    # If accepted, update current sequence
    if accept:
      curr_sequence = new_sequence
      curr_score = new_score

      # If better than best, update best
      if curr_score < best_score:
        best_score = curr_score
        best_sequence = curr_sequence

      num_steps_without_improvement = 0
    else:
      num_steps_without_improvement += 1

    # We always cool down by the rate, regardless of improvement.
    temperature *= cooling_rate

    # We will not condition the temperature on improvement.

    # Check time and save periodically
    current_time = time.time() - start_time
    if current_time >= 990:  # Save one more time to be safe
      # Save checkpoint every 20 seconds or when near timeout
      if current_time - last_save >= 20 or current_time >= MAX_TIME - 10:
        save_checkpoint(best_sequence, best_score, current_time)
        last_save = current_time

  return best_sequence


from scipy import optimize
linprog = optimize.linprog


def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
  """Returns the normalized sequence from the LP solution, not including the step."""
  n = len(sequence)
  sum_sequence = np.sum(sequence)
  normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
  rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
  g_fun = solve_convolution_lp(normalized_sequence, rhs)
  if g_fun is None:
    return None
  # We return the normalized_g_fun, which is the direction to move into.
  sum_g = np.sum(g_fun)
  normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
  return normalized_g_fun


def solve_convolution_lp(f_sequence, rhs):
  """Solves the convolution LP for a given sequence and RHS."""
  n = len(f_sequence)
  c = -np.ones(n)
  a_ub = []
  b_ub = []
  # Reduce the number of constraints by using only relevant lags.
  current_conv = np.convolve(f_sequence, f_sequence, mode='full')
  max_conv = np.max(current_conv) if 2*n-1 > 0 else 0
  # Use 70% of the maximum as the threshold for including a constraint.
  threshold = 0.7 * max_conv
  print(f"using constraints for which convolve[k] > {threshold}")

  a_ub_reduced = []
  b_ub_reduced = []
  for k in range(2 * n - 1):
    if current_conv[k] > threshold:
      row = np.zeros(n)
      for i in range(n):
        j = k - i
        if 0 <= j < n:
          row[j] = f_sequence[i]
      a_ub_reduced.append(row)
      b_ub_reduced.append(rhs)

  if len(a_ub_reduced) == 0:
    a_ub_reduced = [np.ones(n)]
    b_ub_reduced = [rhs]
  a_ub = np.vstack([a_ub_reduced])
  b_ub = np.array(b_ub_reduced)
  print(f"Number of constraints: {len(a_ub_reduced)}")

  # Non-negativity constraints: b_i >= 0
  a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
  b_ub_nonneg = np.zeros(n)  # Zero vector

  a_ub = np.vstack([a_ub, a_ub_nonneg])
  b_ub = np.hstack([b_ub, b_ub_nonneg])

  # Try to use a faster method for large scale LP
  result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')

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
    print(f"C1 value: {c1_value:.10f}")
    print(f"Target: {TARGET_VALUE} ({100*c1_value/TARGET_VALUE:.1f}% of target)")
    print(f"Function length: {len(heights)}")