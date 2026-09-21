"""Sum Tinker losses over datums without server-side normalization.

Client loss inputs carry normalization; the trainer accumulates raw sums.
The SDK represents custom-loss gradients as `weights = -dL/dlogprob` with `cross_entropy`.
"""

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import (
    all_gather_with_cp,
    get_local_response_loss_masks,
    slice_log_prob_with_cp,
)
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

PPO_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
CISPO_DEFAULTS = {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
DRO_DEFAULTS = {"beta": 0.05}


def _target_logprobs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> list[torch.Tensor]:
    # Tinker targets are explicit labels: splice them over the response region of the gather sequence
    label_tokens = [
        torch.cat([tokens[: len(tokens) - len(targets)], _as_tensor_like(targets, tokens)])
        for tokens, targets in zip(batch["unconcat_tokens"], batch["target_tokens"], strict=True)
    ]
    outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=label_tokens,
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    return outputs["log_probs"]


def _as_tensor_like(values, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _per_datum_token_sharding_with_cp(args: Namespace, batch: RolloutBatch, values: list) -> list:
    """Shard each datum's per-response-token loss inputs to this CP rank without changing the batch."""
    max_seq_lens = batch.get("max_seq_lens")
    return [
        slice_log_prob_with_cp(
            value, total, response, args.qkv_format, max_seq_lens[i] if max_seq_lens is not None else None
        )
        for i, (value, total, response) in enumerate(
            zip(values, batch["total_lengths"], batch["response_lengths"], strict=True)
        )
    ]


@torch.no_grad()
def _gather_per_datum_outputs_with_cp(
    args: Namespace,
    batch: RolloutBatch,
    log_probs: list[torch.Tensor],
    per_datum_losses: list[torch.Tensor],
) -> list[dict]:
    """Gather each datum's detached response-token scores and total loss onto every CP rank."""
    if not per_datum_losses:
        return []
    losses = torch.stack(per_datum_losses)
    cp = get_parallel_state().cp
    if cp.size > 1:
        dist.all_reduce(losses, group=cp.group)
    losses = losses.cpu().unbind()
    max_seq_lens = batch.get("max_seq_lens")
    return [
        {
            "sample_index": index,
            "logprobs": all_gather_with_cp(
                log_prob.detach(),
                total,
                response,
                args.qkv_format,
                max_seq_lens[i] if max_seq_lens is not None else None,
            ).cpu(),
            "loss": sample_loss,
        }
        for i, (index, log_prob, total, response, sample_loss) in enumerate(
            zip(
                batch["sample_indices"],
                log_probs,
                batch["total_lengths"],
                batch["response_lengths"],
                losses,
                strict=True,
            )
        )
    ]


def _response_masks(args: Namespace, batch: RolloutBatch, log_probs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per-datum loss masks; a DP-padding datum is all zeros and must not reach the objective."""
    masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    return [_as_tensor_like(mask, log_prob) for mask, log_prob in zip(masks, log_probs, strict=True)]


def _sum_loss_and_outputs(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    log_probs: list[torch.Tensor],
    per_datum_losses: list[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    if any(log_prob.numel() for log_prob in log_probs):
        loss = torch.stack(per_datum_losses).sum()
    else:
        # Ranks without labels must still run backward. An empty view avoids reading the vocabulary tensor.
        loss = logits[..., :0].sum(dtype=torch.float32)
    per_datum = _gather_per_datum_outputs_with_cp(args, batch, log_probs, per_datum_losses)
    return loss, {"loss": loss.detach(), "per_datum": per_datum}


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = [
        -(_as_tensor_like(weights, log_prob) * log_prob * mask).sum()
        for log_prob, weights, mask in zip(
            log_probs,
            _per_datum_token_sharding_with_cp(args, batch, batch["loss_weights"]),
            _response_masks(args, batch, log_probs),
            strict=True,
        )
    ]
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _per_datum_token_sharding_with_cp(args, batch, batch["advantages"]),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        per_datum_losses.append(-(ratio * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", PPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", PPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _per_datum_token_sharding_with_cp(args, batch, batch["advantages"]),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        advantages = _as_tensor_like(advantage, log_prob)
        objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", CISPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", CISPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _per_datum_token_sharding_with_cp(args, batch, batch["advantages"]),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
        per_datum_losses.append(-(coefficient * log_prob * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def dro_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    beta = config.get("beta", DRO_DEFAULTS["beta"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _per_datum_token_sharding_with_cp(args, batch, batch["advantages"]),
        _response_masks(args, batch, log_probs),
        strict=True,
    ):
        divergence = log_prob - _as_tensor_like(sampling_log_prob, log_prob)
        objective = log_prob * _as_tensor_like(advantage, log_prob) - 0.5 * beta * divergence**2
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
