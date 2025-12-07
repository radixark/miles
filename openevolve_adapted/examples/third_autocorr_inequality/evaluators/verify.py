from typing import Tuple, Any, Optional
import numpy as np

from openevolve.modular_utils.evaluation_controller import get_current_problem_config

PROBLEM_CONFIG = get_current_problem_config()


ACCEPTED_KEYS = ("heights", "step_heights", "height_sequence_3", "height_sequence_2", "height_sequence")

DEFAULT_ERROR_RETURN = 1e8
MAX_CHECK_VALUE = 1e10
MIN_CHECK_VALUE = 1e-10

def _coerce_to_1d_array(solution_data: Any) -> Tuple[Optional[np.ndarray], str]:
    """
    Extract a 1D float array of step heights from accepted formats.
    Supports top-level keys and nested under 'current_solution'.
    """
    arr = None

    if isinstance(solution_data, dict):
        # 1) try top-level accepted keys
        for k in ACCEPTED_KEYS:
            if k in solution_data:
                arr = solution_data[k]
                break

        # 2) if not found, try nested under 'current_solution'
        if arr is None and isinstance(solution_data.get("current_solution"), dict):
            cs = solution_data["current_solution"]
            for k in ACCEPTED_KEYS + ("data",):
                if k in cs:
                    arr = cs[k]
                    break
        if arr is None:
            return None, (
                "Dictionary input must contain one of keys: "
                f"{', '.join(ACCEPTED_KEYS)} (optionally nested under 'current_solution')"
            )
    else:
        # direct array-like
        arr = solution_data

    # Coerce to numpy array and validate
    try:
        arr = np.asarray(arr, dtype=float)
    except Exception as e:
        return None, f"Could not convert input to float array: {e}"

    if arr.ndim != 1 or arr.size == 0:
        return None, f"Heights must be a non-empty 1D array, got shape {arr.shape}"

    return arr, ""


def _expected_n_steps() -> Optional[int]:
    try:
        return int(PROBLEM_CONFIG["core_parameters"]["n_steps"])
    except Exception:
        return None  # allow any length if not specified

def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    try:
        heights, err = _coerce_to_1d_array(solution_data)
        if heights is None:
            return False, err

        # Length check if YAML specifies it
        n_expected = _expected_n_steps()
        if n_expected is not None and heights.size != n_expected:
            return False, (
                f"Length mismatch: expected {n_expected} steps, got {heights.size}"
            )

        # Finite check
        if not np.all(np.isfinite(heights)):
            return False, "Heights contain NaN or infinite values"

        # Check for extreme values that could cause numerical overflow
        max_height_abs = float(np.max(np.abs(heights)))
        if max_height_abs > MAX_CHECK_VALUE:
            return False, f"Heights contain extreme values (|h| > {MAX_CHECK_VALUE})"

        # Sum cannot be zero (otherwise integral^2 is zero)
        total = float(np.sum(heights))
        if abs(total) < MIN_CHECK_VALUE:
            return False, "Sum of heights is zero; invalid for C3 objective"

        # Check that sum squared is not too small
        if total ** 2 < MIN_CHECK_VALUE:
            return False, f"Sum squared is too small (< {MIN_CHECK_VALUE}); invalid for C3 objective"

        return True, "Valid step heights for C3"

    except Exception as e:
        return False, f"Validation error: {e}"


def compute_objective_value(solution_data: Any) -> float:
    try:
        is_valid, _ = validate_solution(solution_data)
        if not is_valid:
            return DEFAULT_ERROR_RETURN

        heights, _ = _coerce_to_1d_array(solution_data)
        n = heights.size

        conv_full = np.convolve(heights, heights)  # full mode

        # Guard against extreme convolution values from numerical instability
        max_conv_abs = float(np.max(np.abs(conv_full)))
        if max_conv_abs > MAX_CHECK_VALUE:
            return DEFAULT_ERROR_RETURN

        sum_heights = float(np.sum(heights))
        sum_squared = sum_heights ** 2

        # Compute C3 upper bound
        c3_upper = abs(2 * n * max_conv_abs / sum_squared)

        # Final check for invalid results
        if not np.isfinite(c3_upper):
            return DEFAULT_ERROR_RETURN

        return c3_upper

    except Exception as e:
        print(f"Error computing C3 objective: {e}")
        return DEFAULT_ERROR_RETURN


if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON solution data")
    args = parser.parse_args()

    # Accept either a JSON array or a JSON object with 'heights'/etc.
    try:
        with open(args.input, "r") as f:
            solution_data = json.load(f)
    except Exception as e:
        print(f"INVALID: failed to load JSON: {e}")
        sys.exit(0)

    is_valid, msg = validate_solution(solution_data)
    if not is_valid:
        print(f"INVALID: {msg}")
        sys.exit(0)

    value = compute_objective_value(solution_data)
    # Print the scalar objective to stdout (OpenEvolve convention)
    print(f"{value}")

# OPENEVOLVE_CONFIG_PATH=/home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/examples/third_autocorr_inequality/configs/config_third_autocorr_inequality_oe.yaml PYTHONPATH=/home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/ python /home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/examples/third_autocorr_inequality/evaluators/verify.py --input /home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/third_autocorrelation_inequality_search_data/alphaevolve.json
# OPENEVOLVE_CONFIG_PATH=/home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/examples/third_autocorr_inequality/configs/config_third_autocorr_inequality_oe.yaml PYTHONPATH=/home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/ python /home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/examples/third_autocorr_inequality/evaluators/verify.py --input /home/v-yipwang/Evolve/RL4Evolve/openevolve_adapted/third_autocorrelation_inequality_search_data/old_bound.json