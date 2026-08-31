"""Loss spec validation: official loss_fn names and inputs to a plain (name, config) pair, no computation."""

from __future__ import annotations

from typing import Any

SUPPORTED_LOSS_FNS = {
    "cross_entropy": ("target_tokens",),
    "importance_sampling": ("target_tokens", "logprobs", "advantages"),
    "ppo": ("target_tokens", "logprobs", "advantages"),
}
UNSUPPORTED_LOSS_FNS = {"cispo", "dro"}


class UnsupportedLossError(ValueError):
    """Maps to a non-retryable 422 on the wire."""


def validate_loss_spec(loss_fn: str, loss_fn_config: dict[str, Any] | None) -> tuple[str, dict[str, float]]:
    """Return the plain (loss_fn, config) pair or raise UnsupportedLossError."""
    if loss_fn in UNSUPPORTED_LOSS_FNS:
        raise UnsupportedLossError(f"loss_fn {loss_fn!r} has no miles kernel yet")
    if loss_fn not in SUPPORTED_LOSS_FNS:
        raise UnsupportedLossError(f"unknown loss_fn {loss_fn!r}")
    config = dict(loss_fn_config or {})
    for key, value in config.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise UnsupportedLossError(f"loss_fn_config[{key!r}] must be a number, got {value!r}")
    return loss_fn, config


def validate_loss_inputs(loss_fn: str, row_index: int, loss_fn_inputs: dict[str, Any]) -> None:
    """Check one Datum carries every input column its loss requires."""
    missing = [name for name in SUPPORTED_LOSS_FNS[loss_fn] if name not in loss_fn_inputs]
    if missing:
        raise UnsupportedLossError(f"datum {row_index} is missing loss_fn_inputs {missing} for {loss_fn!r}")
