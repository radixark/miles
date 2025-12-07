"""
Score transformation utilities for rescaling scores to [0,1] range for RL training

For minimize problems: uses sign flip to convert to maximize problem.
Alternative reciprocal transformation commented for reference.
"""

import math
from typing import Dict, Any

EPS = 1e-8


class ScoreTransformConfig:
    """Configuration for score transformation"""

    def __init__(self,
                 score_range_min: float = 0.0,
                 score_range_max: float = 1.0,
                 alpha: float = 1.0,
                 optimize_mode: str = "maximize",
                 positive_multiplier: float = 1.0):
        self.score_range_min = score_range_min  # A
        self.score_range_max = score_range_max  # B
        self.alpha = alpha
        self.optimize_mode = optimize_mode.lower()
        self.positive_multiplier = positive_multiplier

        # Validation
        if self.score_range_max <= self.score_range_min:
            raise ValueError("score_range_max must be greater than score_range_min")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.optimize_mode not in ["maximize", "minimize"]:
            raise ValueError("optimize_mode must be 'maximize' or 'minimize'")
        if self.positive_multiplier <= 0:
            raise ValueError("positive_multiplier must be positive")


def transform_score_for_rl(original_score: float, config: ScoreTransformConfig) -> float:
    """
    Transform score from [A,B] to [0,1] with power scaling for RL training

    For minimize problems, applies sign flip to convert to maximize problem.
    For maximize problems, applies direct transformation.

    Args:
        original_score: Original score value
        config: Transformation configuration

    Returns:
        Transformed score in [0, positive_multiplier] range,
        or original_score if negative (error codes)
    """
    # Don't transform negative scores (error codes)
    if original_score < 0:
        return original_score

    # Handle minimize mode: flip sign to convert to maximize problem
    if config.optimize_mode == "minimize":
        # Sign flip approach: min(x) problem -> max(-x) problem
        working_score = -original_score
        range_min = -config.score_range_max
        range_max = -config.score_range_min

    else:
        # Maximize mode: direct transformation
        working_score = original_score
        range_min = config.score_range_min
        range_max = config.score_range_max

    # Clamp to valid range
    clamped_score = max(range_min, min(range_max, working_score))

    # Linear transformation to [0, 1]
    linear_score = (clamped_score - range_min) / (range_max - range_min)

    # Apply power transformation: f(x) = x^alpha
    transformed_score = linear_score ** config.alpha

    # Apply positive multiplier
    transformed_score = transformed_score * config.positive_multiplier

    return transformed_score


def create_score_transform_config_from_dict(config_dict: Dict[str, Any]) -> ScoreTransformConfig:
    """Create ScoreTransformConfig from dictionary (usually from YAML)"""
    score_transform = config_dict.get('score_transform', {})

    return ScoreTransformConfig(
        score_range_min=score_transform.get('score_range_min', 0.0),
        score_range_max=score_transform.get('score_range_max', 1.0),
        alpha=score_transform.get('alpha', 1.0),
        optimize_mode=score_transform.get('optimize_mode', 'maximize'),
        positive_multiplier=score_transform.get('positive_multiplier', 5.0)
    )