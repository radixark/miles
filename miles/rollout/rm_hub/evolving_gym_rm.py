import math
from typing import Dict, Any, Optional

from openevolve.modular_utils.error_constants import FORMAT_ERRORS, EXECUTION_ERRORS, ALL_ERROR_CODES

# Globally registered gym (same process as rollout manager)
_GYM = None


def _get_combined_score(metrics: Dict[str, Any], default_low_score: float) -> float:
    """Get combined_score from metrics with fallback."""
    return metrics.get("combined_score", default_low_score)


def _get_rl_normalized_reward(metrics: Dict[str, Any], default_low_score: float) -> float:
    """Get rl_normalized_reward from metrics, fallback to combined_score/default_low_score for error/timeout cases."""
    reward_val = metrics.get("rl_normalized_reward")
    if reward_val is None:
        # combined = _get_combined_score(metrics, default_low_score)
        # assert isinstance(combined, (int, float)) and combined <= 0, \
        #     f"rl_normalized_reward missing and combined_score invalid: {metrics}"
        # reward_val = combined
        reward_val = default_low_score
        
    return reward_val


def _get_parent_combined_score(parent_program, default_low_score: float) -> float:
    """Safely get parent program's combined_score."""
    if parent_program is None:
        return default_low_score

    if hasattr(parent_program, 'metrics') and parent_program.metrics:
        return parent_program.metrics.get("combined_score", default_low_score)
    else:
        assert False, f"parent_program.metrics is None or empty, id={getattr(parent_program, 'id', None)}"



def _process_reward(metrics: Dict[str, Any], parent_program, reward_process_type: str, default_low_score: float) -> float:
    """Process reward value based on reward processing type.

    Args:
        metrics: Child program evaluation metrics
        parent_program: Parent program object
        reward_process_type: Processing mode string
        default_low_score: Default fallback score

    Returns:
        Processed reward value
    """

    if reward_process_type == "original_reward":
        # Return raw combined_score without any transformation
        return _get_combined_score(metrics, default_low_score)

    elif reward_process_type == "improve_reward":
        # Binary reward: 1.0 if child > parent, else 0.0
        child_combined_score = _get_combined_score(metrics, default_low_score)

        if child_combined_score in ALL_ERROR_CODES:
            return child_combined_score

        parent_combined_score = _get_parent_combined_score(parent_program, default_low_score)

        eps = 1e-6
        if child_combined_score > parent_combined_score + eps:
            return 1.0
        else:
            return 0.0

    else:
        # All other types use rl_normalized_reward as base
        reward_val = _get_rl_normalized_reward(metrics, default_low_score)

        if reward_process_type == "rl_normalized_reward":
            return reward_val
        elif reward_process_type == "format_reward":
            return reward_val if reward_val in FORMAT_ERRORS else 1.0
        elif reward_process_type == "validation_reward":
            return reward_val if reward_val in ALL_ERROR_CODES else 1.0
        else:
            # Fallback for unknown types
            return reward_val


def _sanitize_reward(reward_val: float, default_low_score: float, clip_min: float = -10.0, clip_max: float = 10.0) -> float:
    """Sanitize reward value: handle NaN/Inf and clip to safe range."""
    if math.isnan(reward_val) or math.isinf(reward_val):
        print(f"[WARNING] Invalid reward {reward_val}, using {default_low_score}")
        return default_low_score

    clipped = max(clip_min, min(clip_max, reward_val))
    if abs(reward_val - clipped) > 1e-6:
        print(f"[WARNING] Reward clipped: {reward_val:.2e} -> {clipped:.2f}")
    return clipped


def set_gym(gym):
    global _GYM
    _GYM = gym


async def evolving_gym_rm(args, sample) -> Dict[str, Any]:
    """
    Score LLM output using gym and return reward dictionary.
    Convention:
      - Return dict contains at least args.reward_key with scalar reward (e.g., combined_score)
      - Include metrics (complete evaluator stats) and child_id as auxiliary fields
      - Cache child_program for visualization/analysis
    """
    default_low_score = -1.0
    if _GYM is None:
        return {getattr(args, "reward_key", "reward"): default_low_score, "error": "gym_not_initialized"}

    metadata = sample.metadata or {}
    if isinstance(metadata, str):
        # Safety check: some upstream code may serialize to JSON string, don't handle here, use dict default
        assert False, "metadata should not be str"
        metadata = {}

    parent_program = metadata.get("parent_program", None)
    if not parent_program:
        return {getattr(args, "reward_key", "reward"): default_low_score, "error": "missing_parent_program"}

    try:
        # Async evaluation (with semaphore + timeout internally)
        result = await _GYM.response_scorer(sample.response or "", parent_program)

    except Exception as e:
        print(f"exception in evolving_gym_rm: {e}")
        return {getattr(args, "reward_key", "reward"): default_low_score, "error": f"exception:{str(e)[:200]}"}

    if result is None or not getattr(result, "child_metrics", None):
        print(f"result is none or child_metrics is none")
        return {getattr(args, "reward_key", "reward"): default_low_score}

    metrics = result.child_metrics or {}
    reward_val = _process_reward(metrics, parent_program, _GYM.reward_process_type, default_low_score)
    reward_val = _sanitize_reward(reward_val, default_low_score)

    # print(f"[DEBUG evolving_gym_rm] reward_val={reward_val}, type={type(reward_val)}, all metric: {metrics}")

    # Cache child program to temp storage (don't update main database yet, preserve sample independence within round)
    try:
        if getattr(result, "child_program", None) is not None:
            _GYM.database.add_temp(result.child_program, iteration=len(_GYM.database.programs))
            # print(f"Added to temp cache. Cache size: {_GYM.database.get_temp_cache_size()}")
    except Exception as e:
        print(f"Failed to add to temp cache: {e}")

    out = {
        getattr(args, "reward_key", "reward"): reward_val,
        "metrics": metrics,
        "child_id": getattr(getattr(result, "child_program", None), "id", None),
    }
    if getattr(result, "iteration_time", None) is not None:
        out["iteration_time"] = result.iteration_time
    if getattr(result, "artifacts", None):
        out["artifacts"] = result.artifacts
    if getattr(result, "runtime_environment_path", None):
        out["runtime_environment_path"] = result.runtime_environment_path


    return out
