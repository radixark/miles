"""Sum Tinker losses over datums without server-side normalization.

Client loss inputs carry normalization; the trainer accumulates raw sums.
The SDK represents custom-loss gradients as `weights = -dL/dlogprob` with `cross_entropy`.
Per-datum outputs are gathered across the CP group; the loss stays a local-shard sum.
"""

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks, slice_log_prob_with_cp
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


def _local_response_values(args: Namespace, batch: RolloutBatch, values: list) -> list:
    """This rank's zigzag CP shard of each full-length client vector; `rollout_log_probs` arrives pre-sliced."""
    max_seq_lens = batch.get("max_seq_lens", None)
    return [
        slice_log_prob_with_cp(
            value,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_lens[i] if max_seq_lens is not None else None,
        )
        for i, (value, total_length, response_length) in enumerate(
            zip(values, batch["total_lengths"], batch["response_lengths"], strict=True)
        )
    ]


def _local_response_masks(args: Namespace, batch: RolloutBatch, log_probs: list[torch.Tensor]) -> list[torch.Tensor]:
    """This CP rank's slice of each loss mask; a DP-padding datum is all zeros and must not reach the objective."""
    local_masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )
    return [_as_tensor_like(mask, log_prob) for mask, log_prob in zip(local_masks, log_probs, strict=True)]


def _gather_per_datum_outputs(
    args: Namespace,
    batch: RolloutBatch,
    log_probs: list[torch.Tensor],
    per_datum_losses: list[torch.Tensor],
) -> list[dict]:
    """Per-datum outputs reported back to the client; identical on every CP rank."""
    if not per_datum_losses:
        return []
    parallel_state = get_parallel_state()
    logprobs = [log_prob.detach() for log_prob in log_probs]
    if parallel_state.cp.size > 1:
        response_lengths = batch["response_lengths"]
        local_positions = _local_response_values(
            args, batch, [torch.arange(n, device=logprobs[0].device) for n in response_lengths]
        )
        logprobs = [
            logprob.new_zeros(n).index_copy_(0, positions, logprob)
            for logprob, positions, n in zip(logprobs, local_positions, response_lengths, strict=True)
        ]
    losses_and_logprobs = torch.cat([torch.stack([local_loss.detach() for local_loss in per_datum_losses]), *logprobs])
    if parallel_state.cp.size > 1:
        dist.all_reduce(losses_and_logprobs, group=parallel_state.cp.group)
    losses_and_logprobs = losses_and_logprobs.cpu()
    num_datums = len(logprobs)
    full_losses = [full_loss.clone() for full_loss in losses_and_logprobs[:num_datums].unbind()]
    full_logprobs = [
        logprob.clone()
        for logprob in losses_and_logprobs[num_datums:].split([logprob.numel() for logprob in logprobs])
    ]
    return [
        {"sample_index": sample_index, "logprobs": full_logprob, "loss": full_loss}
        for sample_index, full_logprob, full_loss in zip(
            batch["sample_indices"], full_logprobs, full_losses, strict=True
        )
    ]


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
        loss = logits[..., :0].sum(dtype=torch.float32)  # zero, but still connected to logits for backward
    per_datum = _gather_per_datum_outputs(args, batch, log_probs, per_datum_losses)
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
            _local_response_values(args, batch, batch["loss_weights"]),
            _local_response_masks(args, batch, log_probs),
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
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
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
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
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
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
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
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
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
