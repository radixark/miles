"""Loss functions used by the Tinker-compatible Megatron executor."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from miles.ray.tinker.protocol import TinkerError

_RL_INPUTS = frozenset({"target_tokens", "logprobs", "advantages"})
_LOSS_CONFIG_KEYS = {
    "cross_entropy": frozenset(),
    "importance_sampling": frozenset(),
    "ppo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
    "cispo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
    "dro": frozenset({"beta"}),
}


def tensor_from_payload(
    payload: Mapping[str, object],
    *,
    device: torch.device | int | str,
) -> torch.Tensor:
    """Materialize a normalized wire tensor on the training device."""
    dtype_name = payload["dtype"]
    if dtype_name == "int64":
        dtype = torch.int64
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        raise TinkerError(f"unsupported TensorData dtype {dtype_name!r}", category="user")
    tensor = torch.tensor(payload["data"], dtype=dtype, device=device)
    return tensor.reshape(payload["shape"])


def validate_and_get_targets(
    inputs: Mapping[str, torch.Tensor],
    *,
    model_input_length: int,
    loss_fn: str,
) -> torch.Tensor:
    """Validate datum fields and return target token IDs."""
    expected = {"target_tokens", "weights"} if loss_fn == "cross_entropy" else set(_RL_INPUTS)
    missing = expected - set(inputs)
    extra = set(inputs) - expected
    if missing:
        raise TinkerError(f"{loss_fn} requires loss_fn_inputs {sorted(missing)}", category="user")
    if extra:
        raise TinkerError(f"{loss_fn} does not accept loss_fn_inputs {sorted(extra)}", category="user")

    targets = inputs["target_tokens"]
    if targets.dtype != torch.int64 or targets.ndim not in (1, 2):
        raise TinkerError("target_tokens must be an int64 tensor with shape (N,) or (N, K)", category="user")
    if targets.ndim == 2 and targets.shape[1] == 0:
        raise TinkerError("target_tokens top-K dimension must be positive", category="user")
    if targets.shape[0] != model_input_length:
        raise TinkerError(
            f"target_tokens first dimension {targets.shape[0]} must equal model_input length {model_input_length}",
            category="user",
        )
    if loss_fn != "cross_entropy" and targets.ndim != 1:
        raise TinkerError(f"{loss_fn} requires one target token per position", category="user")
    return targets


def compute_tinker_loss(
    target_logprobs: torch.Tensor,
    inputs: Mapping[str, torch.Tensor],
    *,
    loss_fn: str,
    loss_fn_config: Mapping[str, float],
) -> torch.Tensor:
    """Compute the exact sum-reduced public Tinker loss."""
    if not bool(torch.isfinite(target_logprobs).all()):
        raise TinkerError("model produced non-finite target log probabilities", category="server")
    if loss_fn not in _LOSS_CONFIG_KEYS:
        raise TinkerError(f"unsupported loss function {loss_fn!r}", category="user")
    unexpected_config = set(loss_fn_config) - set(_LOSS_CONFIG_KEYS[loss_fn])
    if unexpected_config:
        raise TinkerError(
            f"{loss_fn} does not accept loss_fn_config keys {sorted(unexpected_config)}",
            category="user",
        )
    if any(not math.isfinite(float(value)) for value in loss_fn_config.values()):
        raise TinkerError(f"{loss_fn} loss_fn_config values must be finite", category="user")

    if loss_fn == "cross_entropy":
        weights = _matching_float_tensor(inputs["weights"], target_logprobs, "weights")
        return -(target_logprobs * weights).sum()

    sampling_logprobs = _matching_float_tensor(inputs["logprobs"], target_logprobs, "logprobs")
    advantages = _matching_float_tensor(inputs["advantages"], target_logprobs, "advantages")
    log_ratio = target_logprobs - sampling_logprobs

    if loss_fn == "importance_sampling":
        return -(log_ratio.exp() * advantages).sum()

    if loss_fn == "ppo":
        low, high = _clip_thresholds(loss_fn_config, default_low=0.8, default_high=1.2)
        ratio = log_ratio.exp()
        objective = torch.minimum(ratio * advantages, ratio.clamp(low, high) * advantages)
        return -objective.sum()

    if loss_fn == "cispo":
        low, high = _clip_thresholds(loss_fn_config, default_low=0.0, default_high=4.0)
        coefficient = log_ratio.exp().clamp(low, high).detach()
        return -(coefficient * target_logprobs * advantages).sum()

    beta = float(loss_fn_config.get("beta", 0.1))
    if beta < 0:
        raise TinkerError("DRO beta must be non-negative", category="user")
    objective = target_logprobs * advantages - 0.5 * beta * log_ratio.square()
    return -objective.sum()


def tensor_data(tensor: torch.Tensor) -> dict[str, object]:
    """Serialize one float output tensor into the SDK's TensorData shape."""
    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    return {
        "data": value.reshape(-1).tolist(),
        "dtype": "float32",
        "shape": list(value.shape),
    }


def _matching_float_tensor(
    value: torch.Tensor,
    target: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not value.is_floating_point():
        raise TinkerError(f"{name} must be a float32 tensor", category="user")
    if not bool(torch.isfinite(value).all()):
        raise TinkerError(f"{name} must contain only finite values", category="user")
    if value.shape != target.shape:
        raise TinkerError(
            f"{name} shape {list(value.shape)} must match target_tokens shape {list(target.shape)}",
            category="user",
        )
    return value.to(dtype=torch.float32)


def _clip_thresholds(
    config: Mapping[str, float],
    *,
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    low = float(config.get("clip_low_threshold", default_low))
    high = float(config.get("clip_high_threshold", default_high))
    if low < 0 or high < low:
        raise TinkerError(
            f"clip thresholds must satisfy 0 <= low <= high, got low={low}, high={high}",
            category="user",
        )
    return low, high
