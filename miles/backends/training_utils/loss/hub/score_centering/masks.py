"""Masking rules for score-centering log-probabilities.

Candidate slots outside ``head_mask`` are padding: their log-probability is a
finite placeholder and their probability is zero, so padding never contributes
to the loss. NaN on inactive (zero-advantage) tokens becomes zero mass.
``-inf`` represents zero probability; scored terms mask non-finite values
before multiplication.
"""

import torch


def _token_mask(active: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-token mask [tokens] over values [tokens, ...]."""
    return active.reshape(active.shape + (1,) * (values.ndim - active.ndim))


def drop_inactive_nan(values: torch.Tensor, active: torch.Tensor, fill: float) -> torch.Tensor:
    """Replace NaN on inactive tokens with ``fill``; preserve values on active tokens."""
    return torch.where(~_token_mask(active, values) & torch.isnan(values), fill, values)


def sanitize_head_log_probs(log_probs: torch.Tensor, head_mask: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Head log-probabilities with inactive NaN mapped to ``-inf`` and padding to the placeholder 0."""
    return torch.where(head_mask, drop_inactive_nan(log_probs, active, -torch.inf), 0.0)


def head_probs(log_probs: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
    """Probabilities of the head candidates, zero at padding."""
    return log_probs.exp().masked_fill(~head_mask, 0.0)


def scored_log_probs(log_probs: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    """Log-probabilities that enter the loss, zeroed wherever the coefficient is zero.

    Masking before the multiply keeps ``0 * -inf`` from turning the loss into NaN.
    """
    finite = torch.isfinite(log_probs)
    scored = coefficient != 0
    return torch.where(scored & finite, log_probs, 0.0)
