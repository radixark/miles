"""Stateless stabilization helpers for the Predictive Routing Replay (PR²)
rollout-side router patch.

Every function in this module takes raw tensors / scalars and returns raw
tensors / scalars (plus a metrics dict for the larger helper). None of them
touch any global state, the controller, or the bias-predictor module — that
integration lives in `predictive_router_replay.py`.

Public API (re-exported by `predictive_router_replay.py` for backward
compat):
    * PREDICTIVE_LAYER_SCALE_SCHEDULES
    * compute_predictive_bias_ratio
    * compute_topk_boundary_margin
    * compute_predictive_layer_scale
    * stabilize_predictive_delta_logits

See `docs/predictive-routing-replay.md` §4.2 for the algorithmic semantics
of each routine.
"""
from __future__ import annotations

import math

import torch

PREDICTIVE_LAYER_SCALE_SCHEDULES = ("none", "linear_decay", "sqrt_decay", "cosine_decay")


def compute_predictive_bias_ratio(predicted_delta_logits: torch.Tensor, reference_logits: torch.Tensor) -> float:
    denom = torch.abs(reference_logits).mean() + 1e-10
    return (torch.abs(predicted_delta_logits).mean() / denom).item()


def compute_topk_boundary_margin(reference_logits: torch.Tensor, topk: int) -> torch.Tensor:
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if reference_logits.dim() < 2:
        raise ValueError(
            f"reference_logits must have at least 2 dimensions [tokens, experts], got {reference_logits.shape}"
        )

    num_experts = reference_logits.shape[-1]
    if topk >= num_experts:
        return torch.full(
            reference_logits.shape[:-1],
            float("inf"),
            device=reference_logits.device,
            dtype=reference_logits.dtype,
        )

    topk_plus_one = torch.topk(reference_logits, k=topk + 1, dim=-1).values
    kth_value = topk_plus_one[..., topk - 1]
    next_value = topk_plus_one[..., topk]
    return torch.clamp(kth_value - next_value, min=0.0)


def compute_predictive_layer_scale(
    *,
    layer_idx: int,
    num_layers: int,
    schedule: str,
    min_scale: float,
) -> float:
    if schedule == "none" or num_layers <= 1:
        return 1.0
    if schedule not in PREDICTIVE_LAYER_SCALE_SCHEDULES:
        raise ValueError(
            f"Unsupported predictive layer scale schedule: {schedule}. "
            f"Expected one of {PREDICTIVE_LAYER_SCALE_SCHEDULES}."
        )

    depth_fraction = min(max(float(layer_idx) / float(num_layers - 1), 0.0), 1.0)
    if schedule == "linear_decay":
        decay_fraction = depth_fraction
    elif schedule == "sqrt_decay":
        decay_fraction = depth_fraction**0.5
    elif schedule == "cosine_decay":
        decay_fraction = 0.5 * (1.0 - math.cos(math.pi * depth_fraction))
    else:
        raise ValueError(f"Unsupported predictive layer scale schedule: {schedule}")
    return 1.0 - decay_fraction * (1.0 - float(min_scale))


def stabilize_predictive_delta_logits(
    *,
    predicted_delta_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    layer_scale_schedule: str,
    layer_scale_min: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    stabilized_delta_logits = predicted_delta_logits
    layer_gate_scale = compute_predictive_layer_scale(
        layer_idx=layer_idx,
        num_layers=num_layers,
        schedule=layer_scale_schedule,
        min_scale=layer_scale_min,
    )
    if layer_gate_scale != 1.0:
        stabilized_delta_logits = stabilized_delta_logits * layer_gate_scale

    raw_mean_abs = torch.abs(predicted_delta_logits.detach()).mean()
    if float(raw_mean_abs.item()) > 1e-10:
        stabilizer_scale = float((torch.abs(stabilized_delta_logits.detach()).mean() / raw_mean_abs).item())
    else:
        stabilizer_scale = 1.0
    metrics = {
        "predictive_raw_bias_to_logits_ratio": compute_predictive_bias_ratio(
            predicted_delta_logits.detach(),
            reference_logits.detach(),
        ),
        "predictive_stabilized_bias_to_logits_ratio": compute_predictive_bias_ratio(
            stabilized_delta_logits.detach(),
            reference_logits.detach(),
        ),
        "predictive_layer_gate_scale": float(layer_gate_scale),
        "predictive_stabilizer_scale": float(stabilizer_scale),
    }
    return stabilized_delta_logits, metrics
