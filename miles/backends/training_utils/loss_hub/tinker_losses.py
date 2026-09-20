"""Sum Tinker losses over datums without server-side normalization.

Client loss inputs carry normalization; the trainer accumulates raw sums.
The SDK represents custom-loss gradients as `weights = -dL/dlogprob` with `cross_entropy`.
"""

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp, slice_log_prob_with_cp
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


def _partition_loss_inputs(args: Namespace, batch: RolloutBatch) -> RolloutBatch:
    """Make a local view without changing the full-response batch used by recomputation."""
    if get_parallel_state().cp.size == 1:
        return batch
    # rollout_log_probs already follow the local layout from get_rollout_data.
    fields = ("loss_masks", "loss_weights" if batch["loss_fn"] == "cross_entropy" else "advantages")
    local = dict(batch)
    for field in fields:
        local[field] = [
            slice_log_prob_with_cp(values, total, response, args.qkv_format)
            for values, total, response in zip(
                batch[field], batch["total_lengths"], batch["response_lengths"], strict=True
            )
        ]
    return local


@torch.no_grad()
def _collect_response_reports(
    batch: RolloutBatch, log_probs: list[torch.Tensor], per_datum_losses: list[torch.Tensor]
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Reconstruct detached reports with one collective for the whole microbatch."""
    count = len(per_datum_losses)
    report = log_probs[0].new_zeros(count + sum(batch["response_lengths"]))
    report[:count].copy_(torch.stack(per_datum_losses))
    responses = report[count:].split(batch["response_lengths"])
    for local, response, total_length in zip(log_probs, responses, batch["total_lengths"], strict=True):
        prompt_length = total_length - response.numel()
        _, _, _, token_ranges = get_logits_and_tokens_offset_with_cp(total_length, response.numel())
        consumed = 0
        for start, end in token_ranges:
            width = end - start
            if width:
                response[start - prompt_length : end - prompt_length].copy_(local[consumed : consumed + width])
            consumed += width
        assert consumed == local.numel(), "response report does not match the local CP layout"
    dist.all_reduce(report, group=get_parallel_state().cp.group)
    report = report.cpu()
    return report[:count], report[count:].split(batch["response_lengths"])


def _response_masks(batch: RolloutBatch, log_probs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per-datum loss masks; a DP-padding datum is all zeros and must not reach the objective."""
    return [_as_tensor_like(mask, log_prob) for mask, log_prob in zip(batch["loss_masks"], log_probs, strict=True)]


def _sum_loss_and_outputs(
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
    reported_losses, reported_logprobs = per_datum_losses, log_probs
    if per_datum_losses and get_parallel_state().cp.size > 1:
        reported_losses, reported_logprobs = _collect_response_reports(batch, log_probs, per_datum_losses)
    per_datum = [
        {"sample_index": index, "logprobs": log_prob.detach().cpu(), "loss": sample_loss.detach().cpu()}
        for index, log_prob, sample_loss in zip(
            batch["sample_indices"], reported_logprobs, reported_losses, strict=True
        )
    ]
    return loss, {"loss": loss.detach(), "per_datum": per_datum}


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    batch = _partition_loss_inputs(args, batch)
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = [
        -(_as_tensor_like(weights, log_prob) * log_prob * mask).sum()
        for log_prob, weights, mask in zip(
            log_probs, batch["loss_weights"], _response_masks(batch, log_probs), strict=True
        )
    ]
    return _sum_loss_and_outputs(batch, logits, log_probs, per_datum_losses)


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    batch = _partition_loss_inputs(args, batch)
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs, batch["rollout_log_probs"], batch["advantages"], _response_masks(batch, log_probs), strict=True
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        per_datum_losses.append(-(ratio * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(batch, logits, log_probs, per_datum_losses)


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", PPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", PPO_DEFAULTS["clip_high_threshold"])
    batch = _partition_loss_inputs(args, batch)
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs, batch["rollout_log_probs"], batch["advantages"], _response_masks(batch, log_probs), strict=True
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        advantages = _as_tensor_like(advantage, log_prob)
        objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(batch, logits, log_probs, per_datum_losses)


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", CISPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", CISPO_DEFAULTS["clip_high_threshold"])
    batch = _partition_loss_inputs(args, batch)
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs, batch["rollout_log_probs"], batch["advantages"], _response_masks(batch, log_probs), strict=True
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
        per_datum_losses.append(-(coefficient * log_prob * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(batch, logits, log_probs, per_datum_losses)


def dro_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    config = batch.get("loss_fn_config") or {}
    beta = config.get("beta", DRO_DEFAULTS["beta"])
    batch = _partition_loss_inputs(args, batch)
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs, batch["rollout_log_probs"], batch["advantages"], _response_masks(batch, log_probs), strict=True
    ):
        divergence = log_prob - _as_tensor_like(sampling_log_prob, log_prob)
        objective = log_prob * _as_tensor_like(advantage, log_prob) - 0.5 * beta * divergence**2
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(batch, logits, log_probs, per_datum_losses)


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
