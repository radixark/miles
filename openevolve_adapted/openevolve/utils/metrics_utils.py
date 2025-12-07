"""
Safe calculation utilities for metrics containing mixed types
"""

from typing import Any, Dict, Optional
from openevolve.modular_utils.score_transform import transform_score_for_rl


def safe_numeric_average(metrics: Dict[str, Any]) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Average of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    # === START: Yiping debug edits ===
    # If combined_score exists, use it directly for consistent scoring
    if "combined_score" in metrics:
        combined_score = metrics["combined_score"]
        if isinstance(combined_score, (int, float)):
            try:
                float_val = float(combined_score)
                if not (float_val != float_val):  # Check for NaN
                    return float_val
            except (ValueError, TypeError, OverflowError):
                exit(1)  # Fall back to original logic
    # === END: Yiping debug edits ===

    # Original logic (preserved for fallback when no combined_score)
    numeric_values = []
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)


def safe_numeric_sum(metrics: Dict[str, Any]) -> float:
    """
    Calculate the sum of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Sum of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    return numeric_sum


def create_evaluation_metrics(
    combined_score: float,
    validity: float,  # 0.0 for error, 1.0 for success
    runtime_seconds: float = 0.0,
    eval_time: float = 0.0,
    exit_code: int = 0,
    timeout_occurred: bool = False,
    score_transform_config = None,
    **kwargs
) -> Dict[str, Any]:
    """Create standardized evaluation metrics with rl_normalized_reward handling"""

    metrics = {
        "combined_score": combined_score,
        "validity": validity,
        "eval_time": eval_time,
        "program_runtime": runtime_seconds,
        "program_timeout": float(timeout_occurred),
        "program_exit_code": exit_code,
        **kwargs
    }

    # Smart handling: add rl_normalized_reward if not in kwargs and score_transform_config exists
    if 'rl_normalized_reward' not in metrics and score_transform_config is not None:
        rl_normalized_reward = transform_score_for_rl(combined_score, score_transform_config)
        metrics['rl_normalized_reward'] = rl_normalized_reward

    return metrics
