"""Loss + sample-weighting helpers for Predictive Routing Replay (PR²).

All functions are pure tensor → tensor; no global state, no controller
access. They're called from the training-side compute pass inside
`PredictiveReplayController` (in `predictive_router_replay.py`).

Public API (re-exported by `predictive_router_replay.py` for backward
compat):
    * compute_predictive_loss
    * build_topk_boundary_loss_weights
    * build_synthetic_predictive_loss

See `docs/predictive-routing-replay.md` §4.1 (paper baseline) and §4.2
(Miles weighting extensions) for the algorithmic intent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from miles.backends.megatron_utils.predictive_router_stabilization import compute_topk_boundary_margin


def compute_predictive_loss(
    *,
    old_logits: torch.Tensor,
    current_logits: torch.Tensor,
    predicted_delta_logits: torch.Tensor,
    loss_type: str,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if old_logits.shape != current_logits.shape:
        raise ValueError(f"old_logits shape {old_logits.shape} != current_logits shape {current_logits.shape}")
    if old_logits.shape != predicted_delta_logits.shape:
        raise ValueError(
            f"old_logits shape {old_logits.shape} != predicted_delta_logits shape {predicted_delta_logits.shape}"
        )

    logits_diff = current_logits - old_logits

    def _weighted_mean(per_token_loss: torch.Tensor) -> torch.Tensor:
        if sample_weights is None:
            return per_token_loss.mean()
        weights = sample_weights.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
        weight_sum = weights.sum()
        if float(weight_sum.item()) <= 0.0:
            return per_token_loss.sum() * 0.0
        return (per_token_loss * weights).sum() / weight_sum

    if loss_type == "kl":
        pred_log_probs = torch.log_softmax(predicted_delta_logits, dim=-1)
        target_probs = torch.softmax(logits_diff, dim=-1)
        per_token_loss = F.kl_div(pred_log_probs, target_probs.detach(), reduction="none").sum(dim=-1)
        return _weighted_mean(per_token_loss)

    if loss_type == "kl-post":
        pred_log_probs = torch.log_softmax(old_logits + predicted_delta_logits, dim=-1)
        target_probs = torch.softmax(current_logits, dim=-1)
        per_token_loss = F.kl_div(pred_log_probs, target_probs.detach(), reduction="none").sum(dim=-1)
        return _weighted_mean(per_token_loss)

    raise ValueError(f"Unsupported predictive loss type: {loss_type}")


def build_topk_boundary_loss_weights(
    *,
    old_logits: torch.Tensor,
    topk: int,
    max_boundary_loss_weight: float | None,
    min_boundary_margin: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    boundary_margin = compute_topk_boundary_margin(old_logits.detach(), topk=topk)
    safe_boundary_margin = torch.clamp(boundary_margin, min=float(min_boundary_margin))
    inverse_margin = 1.0 / safe_boundary_margin
    normalized_weights = inverse_margin / (inverse_margin.mean() + 1e-10)
    metrics = {
        "predictive_boundary_margin_mean": float(boundary_margin.mean().item()),
        "predictive_boundary_margin_min": float(boundary_margin.min().item()),
    }
    if max_boundary_loss_weight is None:
        metrics["predictive_boundary_loss_weight_mean"] = 1.0
        metrics["predictive_boundary_loss_weight_max"] = 1.0
        metrics["predictive_boundary_loss_weight_gt1_fraction"] = 0.0
        return None, metrics

    weights = torch.clamp(normalized_weights, min=0.0, max=float(max_boundary_loss_weight)).to(dtype=torch.float32)
    metrics["predictive_boundary_loss_weight_mean"] = float(weights.mean().item())
    metrics["predictive_boundary_loss_weight_max"] = float(weights.max().item())
    metrics["predictive_boundary_loss_weight_gt1_fraction"] = float((weights > 1.0).float().mean().item())
    return weights, metrics


def build_synthetic_predictive_loss(
    *,
    bias_predictor: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    synthetic_delta_logits = bias_predictor(input_tensor.detach())
    return (synthetic_delta_logits * 0.0).sum()
