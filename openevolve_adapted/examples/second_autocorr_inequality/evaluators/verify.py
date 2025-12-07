from typing import Tuple, Any, Optional
import numpy as np

from openevolve.modular_utils.evaluation_controller import get_current_problem_config

PROBLEM_CONFIG = get_current_problem_config()


ACCEPTED_KEYS = ("heights", "step_heights", "height_sequence_2", "height_sequence")

DEFAULT_ERROR_RETURN = 0.0
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
    """
    C2 validation requirements:
      - 1D non-empty array with finite values
      - Non-negative: f >= 0 (with small numerical tolerance)
      - If YAML specifies core_parameters.n_steps, length must match
    """
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

        if np.any(heights < - MIN_CHECK_VALUE):
            return False, "Heights must be non-negative for the C2 task (f >= 0)"

        # Check for extreme values that could cause numerical overflow
        max_height_abs = float(np.max(np.abs(heights)))
        if max_height_abs > MAX_CHECK_VALUE:
            return False, f"Heights contain extreme values (|h| > {MAX_CHECK_VALUE})"

        return True, "Valid step heights for C2"

    except Exception as e:
        return False, f"Validation error: {e}"


def compute_objective_value(solution_data: Any) -> float:
    """
    Compute C2 lower bound R(f) (to be maximized):
      1) Full autoconvolution conv = f * f
      2) Piecewise-linear squared integral (zero-padded at endpoints) for ||f*f||_2^2
      3) ||f*f||_1 = sum(|conv|)/(N+1)
      4) ||f*f||_inf = max(|conv|)
      5) R(f) = L2^2 / (L1 * Linf)
    """
    try:
        is_valid, _ = validate_solution(solution_data)
        if not is_valid:
            return DEFAULT_ERROR_RETURN

        heights, _ = _coerce_to_1d_array(solution_data)

        # 1) Full autoconvolution
        conv = np.convolve(heights, heights, mode="full")
        num_points = conv.size
        if num_points == 0:
            return DEFAULT_ERROR_RETURN

        # Guard against extreme convolution values from numerical instability
        max_conv_abs = float(np.max(np.abs(conv)))
        if max_conv_abs > MAX_CHECK_VALUE:
            return DEFAULT_ERROR_RETURN

        # 2) L2 norm squared via piecewise-linear integral with zero padding at ends
        x_points = np.linspace(-0.5, 0.5, num_points + 2)
        x_intervals = np.diff(x_points)
        y_points = np.concatenate(([0.0], conv, [0.0]))

        l2_norm_squared = 0.0
        # Integral formula: h/3 * (y1^2 + y1*y2 + y2^2) for piecewise linear
        for i in range(num_points + 1):
            y1 = y_points[i]
            y2 = y_points[i + 1]
            h  = x_intervals[i]
            l2_norm_squared += (h / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

        # Guard against non-finite l2 norm
        if not np.isfinite(l2_norm_squared) or l2_norm_squared < MIN_CHECK_VALUE * MIN_CHECK_VALUE:
            return DEFAULT_ERROR_RETURN

        # 3) L1 norm (averaged definition)
        norm_1 = float(np.sum(np.abs(conv))) / float(num_points + 1)

        # 4) Linf norm
        norm_inf = float(np.max(np.abs(conv)))

        if norm_1 <= MIN_CHECK_VALUE or norm_inf <= MIN_CHECK_VALUE:
            return DEFAULT_ERROR_RETURN

        # 5) C2 lower bound: R(f)
        c2_upper = float(l2_norm_squared) / (norm_1 * norm_inf)

        # Final check for invalid results
        if not np.isfinite(c2_upper):
            return DEFAULT_ERROR_RETURN

        return c2_upper

    except Exception as e:
        print(f"Error computing C2 objective: {e}")
        return DEFAULT_ERROR_RETURN


if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON solution data")
    args = parser.parse_args()

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
    # OpenEvolve convention: output pure numeric value (R(f) to maximize)
    print(f"{value}")
