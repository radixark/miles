"""Log-probs and the policy loss on the padded microbatch, for accelerators that compile one program per shape.

The per-sample path (`logit_processors.py`, `losses.py`) slices every response out of the packed logits,
so each op after the model sees that response's length. TorchTPU runs every eager op as an XLA program
compiled per input shape and view offset, and RL responses rarely repeat a length, so that path compiles
thousands of new programs per training step. Here every device tensor keeps the padded length T of the
model input instead, which the torchtitan trainer rounds up to a few buckets on TPU:

- log-probs and entropy are scored for all T rows, in fixed-size row chunks, against a `[T]` target row;
- prompt and padding rows are masked out, and a per-sample mean divides each row by its sample's count;
- per-sample tensors (tokens, loss masks, old log-probs, advantages) stay on the host. A microbatch moves
  each of them to the device in one copy, laid out on the rows, and moves results back in one copy.
"""

import itertools
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from miles import platforms
from miles.backends.training_utils.loss.hub.math_utils import (
    _safe_exp_neg_ppo_kl,
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_policy_loss,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.types import RolloutBatch

# the smallest TPU length bucket (torchtitan_utils/trainer.py)
_CHUNK_ROWS = 1024
_ADVANTAGE_ESTIMATORS = ("grpo", "reinforce_plus_plus", "reinforce_plus_plus_baseline")


def fixed_shape_loss_enabled(args: Namespace) -> bool:
    """Whether log-probs and the policy loss run on padded microbatches instead of per-sample slices.

    Only the torchtitan trainer pads microbatches to fixed buckets, and only without CP or PP is every
    padded row a whole packed sample. Features this module does not implement keep the per-sample path.
    """
    return (
        not platforms.current_platform.dynamic_shapes
        and args.train_backend == "torchtitan"
        and args.qkv_format == "thd"
        and args.context_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.loss_type == "policy_loss"
        and args.advantage_estimator in _ADVANTAGE_ESTIMATORS
        and not (
            args.true_on_policy_mode
            or args.allgather_cp
            or args.use_sampling_support_replay
            or args.use_opsm
            or args.use_tis
            or args.get_mismatch_metrics
            or args.use_opd
            or args.custom_pg_loss_reducer_function_path is not None
            or args.dump_details is not None
            or is_multi_lora_enabled(args)
        )
    )


@dataclass(frozen=True)
class PackedResponses:
    """Where each response sits among the rows of a microbatch's padded `[T, V]` logits.

    Row t predicts packed token t + 1, so the response of sample i fills rows
    `[start_i + prompt_i - 1, start_i + total_i - 1)`, where `start_i` is the sample's packed offset.
    """

    rows: torch.Tensor  # [sum(R)] int64 on the host: the row of every response token, in sample order
    response_lengths: list[int]
    num_rows: int
    device: torch.device

    @classmethod
    def build(
        cls, *, total_lengths: list[int], response_lengths: list[int], num_rows: int, device: torch.device
    ) -> "PackedResponses":
        assert get_parallel_state().cp.size == 1, "the padded rows hold whole samples only without context parallelism"
        if sum(total_lengths) > num_rows:
            raise ValueError(f"{sum(total_lengths)} packed tokens do not fit {num_rows} logit rows")
        starts = itertools.accumulate(total_lengths, initial=0)
        rows = torch.cat(
            [
                torch.arange(start + total - response - 1, start + total - 1)
                for start, total, response in zip(starts, total_lengths, response_lengths, strict=False)
            ]
        )
        if rows.numel() and rows[0] < 0:
            raise ValueError("the first sample of a microbatch needs at least one prompt token")
        return cls(rows=rows, response_lengths=list(response_lengths), num_rows=num_rows, device=device)

    def scatter(
        self, per_sample: Sequence[torch.Tensor], dtype: torch.dtype = torch.float32, *, fill: float = 0
    ) -> torch.Tensor:
        """Host per-sample `[R_i]` values -> `[T]` rows on the device (`fill` elsewhere), in one copy."""
        values = torch.full((self.num_rows,), fill, dtype=dtype)
        values[self.rows] = torch.cat([value.to("cpu", dtype) for value in per_sample])
        return values.to(self.device)

    def split(self, values: torch.Tensor) -> list[torch.Tensor]:
        """`[T]` rows on the device -> host per-sample `[R_i]` values, in one copy."""
        return list(values.detach().cpu()[self.rows].split(self.response_lengths))

    def targets(self, unconcat_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        """The token each row predicts: the response tokens on response rows, 0 elsewhere."""
        responses = [
            tokens[tokens.numel() - length :]
            for tokens, length in zip(unconcat_tokens, self.response_lengths, strict=True)
        ]
        return self.scatter(responses, dtype=torch.long)

    def mean_denominators(
        self,
        loss_masks: Sequence[torch.Tensor],
        *,
        per_token: bool,
        denominators: Sequence[torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`[T]` divisors that make `(x * loss_masks / d).sum()` reduce like `get_sum_of_sample_mean`.

        A response row gets its sample's `max(loss_mask.sum(), 1)` (or of `denominators`), every other row 1;
        per-token sums divide by 1. Dividing each row, rather than each sample's sum, keeps the gradient of
        every row exactly what the per-sample reduction gives it.
        """
        if per_token:
            return torch.ones(self.num_rows, device=self.device)
        if denominators is None:
            denominators = [mask.sum() for mask in loss_masks]
        per_sample = [
            torch.clamp_min(denominator.to("cpu", torch.float32), 1).expand(length)
            for denominator, length in zip(denominators, self.response_lengths, strict=True)
        ]
        return self.scatter(per_sample, fill=1)


def get_fixed_shape_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool,
) -> dict[str, list[torch.Tensor]]:
    """`get_log_probs_and_entropy` for the forward-only passes: per-sample `[R]` results, on the host."""
    layout = PackedResponses.build(
        total_lengths=total_lengths, response_lengths=response_lengths, num_rows=logits.size(1), device=logits.device
    )
    log_probs, entropy = _score_rows(
        logits, layout.targets(unconcat_tokens), args=args, with_entropy=with_entropy, entropy_requires_grad=False
    )
    result = {"log_probs": layout.split(log_probs)}
    if with_entropy:
        result["entropy"] = layout.split(entropy)
    return result


def fixed_shape_policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`policy_loss_function` on the padded microbatch, with the same loss and metrics.

    `sum_of_sample_mean` reduces concatenated responses, so it is not used: rows reduce with
    `PackedResponses.mean_denominators` instead.
    """
    del sum_of_sample_mean
    layout = PackedResponses.build(
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        num_rows=logits.size(1),
        device=logits.device,
    )
    loss_masks = layout.scatter(batch["loss_masks"])
    active = loss_masks.bool()
    denominators = layout.mean_denominators(
        batch["loss_masks"],
        per_token=args.backend.calculate_per_token_loss,
        denominators=batch.get("rollout_mask_sums"),
    )

    def sum_of_row_mean(x: torch.Tensor) -> torch.Tensor:
        return (x * loss_masks / denominators).sum()

    calculate_entropy = args.entropy_coef != 0 or args.observe_training_entropy
    log_probs, entropy = _score_rows(
        logits,
        layout.targets(batch["unconcat_tokens"]),
        args=args,
        with_entropy=calculate_entropy,
        entropy_requires_grad=args.entropy_coef != 0,
    )
    # rows outside the loss mask score arbitrary targets; zeroing them keeps exp() in the KL terms finite
    log_probs = torch.where(active, log_probs, 0.0)
    rollout_log_probs = layout.scatter(batch["rollout_log_probs"]) if batch.get("rollout_log_probs") else None
    old_log_probs, scored_log_probs = _old_and_scored_log_probs(args, batch, layout, log_probs, rollout_log_probs)

    ppo_kl = _masked(old_log_probs - log_probs, active)
    advantages = _masked(layout.scatter(batch["advantages"]), active)
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high, args.eps_clip_c)
    ess_ratio = _ess_ratio_contribution(
        ppo_kl, loss_masks, layout, batch["loss_masks"], per_token=args.backend.calculate_per_token_loss
    )

    pg_loss = sum_of_row_mean(pg_loss)
    loss = pg_loss
    entropy_loss = pg_loss.new_zeros(())
    if calculate_entropy:
        entropy_loss = sum_of_row_mean(entropy)
        if args.entropy_coef != 0:
            loss = pg_loss - args.entropy_coef * entropy_loss
        else:
            entropy_loss = entropy_loss.detach()

    reported_loss = {
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": sum_of_row_mean(pg_clipfrac).detach(),
        "ppo_kl": sum_of_row_mean(ppo_kl).detach(),
        "ess_ratio": ess_ratio,
    }
    if rollout_log_probs is not None:
        scored = scored_log_probs if scored_log_probs is not None else log_probs.detach()
        abs_diff = _masked((scored - rollout_log_probs).abs(), active)
        rollout_train_kl = compute_approx_kl(rollout_log_probs, scored, kl_loss_type="low_var_kl")
        reported_loss["train_rollout_logprob_abs_diff"] = sum_of_row_mean(abs_diff).detach()
        reported_loss["train_rollout_kl"] = sum_of_row_mean(_masked(rollout_train_kl, active)).detach()
    if args.use_kl_loss:
        assert batch.get("ref_log_probs") is not None, "ref_log_probs must be provided when --use-kl-loss is set"
        importance_ratio = torch.exp(log_probs - old_log_probs) if args.use_unbiased_kl else None
        kl = compute_approx_kl(
            log_probs,
            layout.scatter(batch["ref_log_probs"]),
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_row_mean(_masked(kl, active))
        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl_loss
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    return loss, {"loss": loss.clone().detach(), **reported_loss}


def _score_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    args: Namespace,
    with_entropy: bool,
    entropy_requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Log-prob of `targets` (and entropy) for every row of `[1, T, V]` logits; `[T]` each."""
    assert logits.dim() == 3 and logits.size(0) == 1, f"thd logits are [1, T, V], got {logits.shape}"
    # -1 means "a whole response at once" on the per-sample path. A padded microbatch is always a full
    # bucket, so it is always chunked: the fp32 copy of a chunk stays [chunk, V].
    chunk_size = args.log_probs_chunk_size if args.log_probs_chunk_size > 0 else _CHUNK_ROWS
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits.squeeze(0),
        targets,
        get_parallel_state().tp.group,
        with_entropy=with_entropy,
        entropy_requires_grad=entropy_requires_grad,
        chunk_size=chunk_size,
        vocab_size=args.backend.vocab_size if args.train_backend == "megatron" else None,
        temperature=args.rollout_temperature,
        debug_unified_grad_fused_logprob=args.debug_unified_grad_fused_logprob,
    )
    return log_probs.squeeze(-1), entropy


def _old_and_scored_log_probs(
    args: Namespace,
    batch: RolloutBatch,
    layout: PackedResponses,
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The ratio's old-policy log-probs, and the trainer-scored ones the diagnostics compare (None if absent)."""
    if args.skip_actor_forward_only:
        scored = log_probs.detach()
    elif batch.get("log_probs") is not None:
        scored = layout.scatter(batch["log_probs"])
    else:
        scored = None
    if args.use_rollout_logprobs:
        assert rollout_log_probs is not None, "rollout_log_probs must be provided when --use-rollout-logprobs is set"
        return rollout_log_probs, scored
    if scored is None:
        raise ValueError("policy loss requires old-policy log-probs")
    return scored, scored


def _masked(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return torch.where(active, torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), x.new_zeros(()))


def _ess_ratio_contribution(
    ppo_kl: torch.Tensor,
    loss_masks: torch.Tensor,
    layout: PackedResponses,
    per_sample_loss_masks: Sequence[torch.Tensor],
    *,
    per_token: bool,
) -> torch.Tensor:
    """`compute_ess_ratio_contribution` on rows: per-sample sums land in a `[T]` buffer indexed by sample."""
    sample_ids = layout.scatter(
        [torch.full((length,), i) for i, length in enumerate(layout.response_lengths)], dtype=torch.long
    )
    num_valid = torch.ones(layout.num_rows)
    num_valid[: len(per_sample_loss_masks)] = torch.stack(
        [torch.clamp_min(mask.to("cpu", torch.float32).sum(), 1) for mask in per_sample_loss_masks]
    )
    num_valid = num_valid.to(layout.device)

    is_weights = _safe_exp_neg_ppo_kl(ppo_kl.detach()) * loss_masks
    sum_w = torch.zeros_like(is_weights).index_add_(0, sample_ids, is_weights)
    sum_w2 = torch.zeros_like(is_weights).index_add_(0, sample_ids, is_weights * is_weights)
    ess_ratio = (sum_w * sum_w) / (num_valid * torch.clamp_min(sum_w2, 1e-8))
    return (ess_ratio * num_valid).sum() if per_token else ess_ratio.sum()
