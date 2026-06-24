"""Stateless stabilization helpers for the Predictive Routing Replay (PR²)
rollout-side router patch.

Every function in this module takes raw tensors / scalars and returns raw
tensors / scalars (plus a metrics dict for the two larger helpers). None of
them touch any global state, the controller, or the bias-predictor module —
that integration lives in `predictive_router_replay.py`.

Public API (re-exported by `predictive_router_replay.py` for backward
compat):
    * PREDICTIVE_LAYER_SCALE_SCHEDULES
    * compute_predictive_bias_ratio
    * compute_topk_boundary_margin
    * compute_topk_set_change_mask
    * compute_predictive_layer_scale
    * resolve_predictive_topk_margin_ratio
    * stabilize_predictive_delta_logits
    * apply_predictive_flip_fallback

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


def compute_topk_set_change_mask(
    *,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"reference_logits shape {reference_logits.shape} != candidate_logits shape {candidate_logits.shape}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    _, reference_topk = torch.topk(reference_logits, k=topk, dim=-1)
    _, candidate_topk = torch.topk(candidate_logits, k=topk, dim=-1)
    matches = (reference_topk.unsqueeze(-1) == candidate_topk.unsqueeze(-2)).any(dim=-1)
    return ~matches.all(dim=-1)


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


def resolve_predictive_topk_margin_ratio(
    *,
    base_ratio: float | None,
    final_ratio: float | None = None,
    anneal_start_rollout: int | None = None,
    anneal_end_rollout: int | None = None,
    current_rollout_id: int | None = None,
) -> tuple[float | None, float | None]:
    if base_ratio is None:
        return None, None

    resolved_base_ratio = float(base_ratio)
    if final_ratio is None or anneal_end_rollout is None or current_rollout_id is None:
        return resolved_base_ratio, None

    start_rollout = 0 if anneal_start_rollout is None else int(anneal_start_rollout)
    end_rollout = int(anneal_end_rollout)
    if end_rollout <= start_rollout:
        raise ValueError(
            "anneal_end_rollout must be greater than anneal_start_rollout when using "
            "predictive top-k margin-ratio annealing."
        )

    if current_rollout_id <= start_rollout:
        return resolved_base_ratio, 0.0
    if current_rollout_id >= end_rollout:
        return float(final_ratio), 1.0

    progress = float(current_rollout_id - start_rollout) / float(end_rollout - start_rollout)
    resolved_ratio = resolved_base_ratio + (float(final_ratio) - resolved_base_ratio) * progress
    return resolved_ratio, progress


def stabilize_predictive_delta_logits(
    *,
    predicted_delta_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    layer_scale_schedule: str,
    layer_scale_min: float,
    max_delta_to_old_ratio: float | None,
    topk: int | None = None,
    max_delta_to_topk_margin_ratio: float | None = None,
    max_delta_to_topk_margin_ratio_final: float | None = None,
    topk_margin_ratio_anneal_start_rollout: int | None = None,
    topk_margin_ratio_anneal_end_rollout: int | None = None,
    current_rollout_id: int | None = None,
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

    ratio_clip_scale = 1.0
    if max_delta_to_old_ratio is not None:
        reference_mean_abs = torch.abs(reference_logits.detach()).mean()
        if float(reference_mean_abs.item()) > 1e-10:
            delta_mean_abs = torch.abs(stabilized_delta_logits.detach()).mean()
            max_allowed_mean_abs = reference_mean_abs * float(max_delta_to_old_ratio)
            if float(delta_mean_abs.item()) > float(max_allowed_mean_abs.item()):
                ratio_clip_scale = float((max_allowed_mean_abs / (delta_mean_abs + 1e-10)).item())
                stabilized_delta_logits = stabilized_delta_logits * ratio_clip_scale

    margin_clip_scale_mean = 1.0
    margin_clip_scale_min = 1.0
    topk_boundary_margin_mean = float("inf")
    effective_topk_margin_ratio, topk_margin_ratio_anneal_progress = resolve_predictive_topk_margin_ratio(
        base_ratio=max_delta_to_topk_margin_ratio,
        final_ratio=max_delta_to_topk_margin_ratio_final,
        anneal_start_rollout=topk_margin_ratio_anneal_start_rollout,
        anneal_end_rollout=topk_margin_ratio_anneal_end_rollout,
        current_rollout_id=current_rollout_id,
    )
    if effective_topk_margin_ratio is not None:
        if topk is None:
            raise ValueError("topk must be provided when max_delta_to_topk_margin_ratio is set.")
        boundary_margin = compute_topk_boundary_margin(reference_logits.detach(), topk=topk)
        topk_boundary_margin_mean = float(boundary_margin.mean().item())
        max_abs_delta = torch.abs(stabilized_delta_logits.detach()).amax(dim=-1)
        max_allowed_abs_delta = boundary_margin * (float(effective_topk_margin_ratio) * 0.5)
        margin_clip_scale = torch.ones_like(max_abs_delta)
        clip_mask = max_abs_delta > (max_allowed_abs_delta + 1e-10)
        margin_clip_scale = torch.where(
            clip_mask,
            max_allowed_abs_delta / (max_abs_delta + 1e-10),
            margin_clip_scale,
        )
        margin_clip_scale = torch.clamp(margin_clip_scale, min=0.0, max=1.0)
        stabilized_delta_logits = stabilized_delta_logits * margin_clip_scale.unsqueeze(-1)
        margin_clip_scale_mean = float(margin_clip_scale.mean().item())
        margin_clip_scale_min = float(margin_clip_scale.min().item())

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
        "predictive_ratio_clip_scale": float(ratio_clip_scale),
        "predictive_margin_clip_scale_mean": float(margin_clip_scale_mean),
        "predictive_margin_clip_scale_min": float(margin_clip_scale_min),
        "predictive_topk_boundary_margin_mean": float(topk_boundary_margin_mean),
        "predictive_stabilizer_scale": float(stabilizer_scale),
    }
    if effective_topk_margin_ratio is not None:
        metrics["predictive_effective_topk_margin_ratio"] = float(effective_topk_margin_ratio)
    if topk_margin_ratio_anneal_progress is not None:
        metrics["predictive_topk_margin_ratio_anneal_progress"] = float(topk_margin_ratio_anneal_progress)
    return stabilized_delta_logits, metrics


def apply_predictive_flip_fallback(
    *,
    reference_logits: torch.Tensor,
    adjusted_logits: torch.Tensor,
    topk: int,
    min_post_topk_margin_for_flip: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if reference_logits.shape != adjusted_logits.shape:
        raise ValueError(
            f"reference_logits shape {reference_logits.shape} != adjusted_logits shape {adjusted_logits.shape}"
        )

    route_change_mask = compute_topk_set_change_mask(
        reference_logits=reference_logits.detach(),
        candidate_logits=adjusted_logits.detach(),
        topk=topk,
    )
    post_boundary_margin = compute_topk_boundary_margin(adjusted_logits.detach(), topk=topk)
    changed_fraction = float(route_change_mask.float().mean().item())
    changed_boundary_margin_mean = float("inf")
    if bool(route_change_mask.any().item()):
        changed_boundary_margin_mean = float(post_boundary_margin[route_change_mask].mean().item())

    fallback_mask = torch.zeros_like(route_change_mask, dtype=torch.bool)
    if min_post_topk_margin_for_flip is not None:
        fallback_mask = route_change_mask & (post_boundary_margin < float(min_post_topk_margin_for_flip))

    effective_logits = adjusted_logits
    if bool(fallback_mask.any().item()):
        expanded_fallback_mask = fallback_mask
        while expanded_fallback_mask.ndim < adjusted_logits.ndim:
            expanded_fallback_mask = expanded_fallback_mask.unsqueeze(-1)
        effective_logits = torch.where(expanded_fallback_mask, reference_logits, adjusted_logits)

    confident_flip_mask = route_change_mask & ~fallback_mask
    execution_weights = (~fallback_mask).to(dtype=torch.float32)
    applied_delta_logits = effective_logits - reference_logits
    metrics = {
        "predictive_route_change_fraction": changed_fraction,
        "predictive_flip_fallback_fraction": float(fallback_mask.float().mean().item()),
        "predictive_confident_flip_fraction": float(confident_flip_mask.float().mean().item()),
        "predictive_post_topk_boundary_margin_mean": float(post_boundary_margin.mean().item()),
        "predictive_post_topk_boundary_margin_changed_mean": float(changed_boundary_margin_mean),
        "predictive_applied_bias_to_logits_ratio": compute_predictive_bias_ratio(
            applied_delta_logits.detach(),
            reference_logits.detach(),
        ),
    }
    return effective_logits, applied_delta_logits, execution_weights, metrics
