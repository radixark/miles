"""Tinker protocol losses.

Raw sums over response tokens: normalization is the client's business via
weights/advantages. Under multi-LoRA the loss scaling path (global batch size
1) turns megatron's microbatch/DP averaging into an exact sum, matching the
protocol's "gradient = sum over datums" semantics.

The SDK's forward_backward_custom sends client-side gradients as
``weights = -dL/dlogprob`` with the cross_entropy loss, so cross_entropy is
also the custom-loss backend.

Config keys and defaults follow the Tinker docs.
"""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.utils.types import RolloutBatch

PPO_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
CISPO_DEFAULTS = {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
DRO_DEFAULTS = {"beta": 0.05}

# Per-datum outputs collected across microbatches for protocol reassembly;
# active only between start/drain (the slot executor's forward_backward).
_per_datum_outputs: list[dict] | None = None


def start_per_datum_outputs() -> None:
    global _per_datum_outputs
    _per_datum_outputs = []


def drain_per_datum_outputs() -> list[dict]:
    global _per_datum_outputs
    outputs, _per_datum_outputs = _per_datum_outputs, None
    return outputs or []


def _record_per_datum(batch: RolloutBatch, log_probs: list[torch.Tensor], per_sample_loss: list[torch.Tensor]) -> None:
    if _per_datum_outputs is None:
        return
    sample_indices = batch.get("sample_indices")
    assert sample_indices is not None, "per-datum outputs need sample_indices in the batch"
    for index, log_prob, loss in zip(sample_indices, log_probs, per_sample_loss, strict=True):
        _per_datum_outputs.append(
            {"sample_index": index, "logprobs": log_prob.detach().cpu(), "loss": loss.detach().cpu()}
        )


def _target_logprobs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> list[torch.Tensor]:
    outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    return outputs["log_probs"]


def _like(values, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def compute_per_datum_losses(
    log_probs: list[torch.Tensor], *, batch: RolloutBatch, loss_fn: str
) -> list[torch.Tensor]:
    """Evaluate the same token-sum objective for forward and forward-backward."""
    if loss_fn == "cross_entropy":
        return [-(_like(weights, lp) * lp).sum() for lp, weights in zip(log_probs, batch["loss_weights"], strict=True)]
    if loss_fn not in ("importance_sampling", "ppo", "cispo", "dro"):
        raise ValueError(f"unknown Tinker loss: {loss_fn}")

    config = batch.get("loss_fn_config") or {}
    clip_defaults = CISPO_DEFAULTS if loss_fn == "cispo" else PPO_DEFAULTS
    clip_low = config.get("clip_low_threshold", clip_defaults["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", clip_defaults["clip_high_threshold"])
    beta = config.get("beta", DRO_DEFAULTS["beta"])
    per_sample = []
    for lp, sampling, adv in zip(log_probs, batch["rollout_log_probs"], batch["advantages"], strict=True):
        advantages = _like(adv, lp)
        divergence = lp - _like(sampling, lp)
        if loss_fn == "dro":
            objective = lp * advantages - 0.5 * beta * divergence**2
        else:
            ratio = torch.exp(divergence)
            if loss_fn == "importance_sampling":
                objective = ratio * advantages
            elif loss_fn == "ppo":
                objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
            else:
                coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
                objective = coefficient * lp * advantages
        per_sample.append(-objective.sum())
    return per_sample


def _finish(
    batch: RolloutBatch,
    logits: torch.Tensor,
    log_probs: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if per_sample_loss:
        loss = torch.stack(per_sample_loss).sum()
    else:
        # a microbatch with no supervised tokens still needs the graph alive; fp32 sum avoids fp16 inf -> nan
        loss = logits.sum(dtype=torch.float32) * 0
    _record_per_datum(batch, log_probs, per_sample_loss)
    return loss, {"loss": loss.clone().detach()}


def _loss_function(
    args: Namespace, batch: RolloutBatch, logits: torch.Tensor, *, loss_fn: str
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = compute_per_datum_losses(log_probs, batch=batch, loss_fn=loss_fn)
    return _finish(batch, logits, log_probs, per_sample)


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _loss_function(args, batch, logits, loss_fn="cross_entropy")


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _loss_function(args, batch, logits, loss_fn="importance_sampling")


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _loss_function(args, batch, logits, loss_fn="ppo")


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _loss_function(args, batch, logits, loss_fn="cispo")


def dro_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _loss_function(args, batch, logits, loss_fn="dro")


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
