from argparse import Namespace

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.advantages import compute_advantages, normalize_advantages
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy, get_values  # noqa: F401
from miles.backends.training_utils.loss_hub.losses import TOKEN_NORMALIZED_TRAIN_KEYS, get_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import TrainAdvantageComputationEvent
from miles.utils.types import RolloutBatch


def get_active_token_count(loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Count active tokens without inventing weight for fully masked samples."""
    if not loss_masks:
        raise ValueError("Cannot normalize a loss over an empty batch.")
    return torch.stack([loss_mask.sum() for loss_mask in loss_masks]).sum()


def get_token_counts_by_step(loss_masks: list[torch.Tensor], num_steps: int) -> torch.Tensor:
    if num_steps <= 0 or len(loss_masks) < num_steps or len(loss_masks) % num_steps != 0:
        raise ValueError(f"Cannot split {len(loss_masks)} loss masks across {num_steps} optimizer steps.")
    samples_per_step = len(loss_masks) // num_steps
    return torch.stack(
        [
            get_active_token_count(loss_masks[start : start + samples_per_step])
            for start in range(0, len(loss_masks), samples_per_step)
        ]
    )


def scale_data_parallel_token_mean_loss(
    loss: torch.Tensor,
    *,
    global_num_tokens: torch.Tensor,
    data_parallel_size: int,
) -> torch.Tensor:
    """Scale a local token numerator before FSDP averages gradients across DP."""
    if global_num_tokens.numel() != 1:
        raise ValueError(f"Expected one token normalizer, got shape {tuple(global_num_tokens.shape)}.")
    return loss * data_parallel_size / global_num_tokens.to(device=loss.device)


def _as_scalar_tensor(value, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device).reshape(())
    return torch.tensor(value, device=device)


def _build_train_log_dict(
    log: dict[str, torch.Tensor],
    *,
    num_samples: int,
    num_tokens: torch.Tensor,
    device: torch.device,
    calculate_per_token_loss: bool,
    metrics_token_reduced: bool,
) -> dict[str, list[str] | torch.Tensor]:
    keys = list(log.keys())
    values = torch.stack([_as_scalar_tensor(value, device=device) for value in log.values()])
    if not calculate_per_token_loss:
        return {
            "keys": keys,
            "values": torch.cat([torch.tensor([num_samples], device=device), values]),
        }

    num_tokens_scalar = num_tokens.to(device=device).reshape(())
    num_samples_scalar = torch.tensor(num_samples, device=device)
    normalizers = torch.stack(
        [
            num_tokens_scalar if metrics_token_reduced or key in TOKEN_NORMALIZED_TRAIN_KEYS else num_samples_scalar
            for key in keys
        ]
    )
    return {
        "keys": keys,
        "values": values,
        "normalizers": normalizers,
    }


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". On-policy distillation (OPD) is applied
    orthogonally on top of any estimator via `args.use_opd`. When
    `args.normalize_advantages` is True, advantages are whitened across the
    data-parallel group using masked statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    advantages, returns = compute_advantages(
        args=args,
        kl=kl,
        rewards=rewards,
        log_probs=log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        values=values,
    )

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    if args.normalize_advantages:
        advantages = normalize_advantages(args, advantages, loss_masks, total_lengths, response_lengths, max_seq_lens)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values". Without `calculate_per_token_loss`, "values" is a 1D
          tensor `[num_samples, metric1, metric2, ...]`; with it, "values"
          holds only the metrics and a "normalizers" tensor carries each
          metric's denominator (`num_tokens` or `num_samples`).
    """
    parallel_state = get_parallel_state()
    num_tokens = get_active_token_count(batch["loss_masks"])
    num_samples = len(batch["response_lengths"])

    # Policy loss selects pg_loss aggregation separately; this shared reducer is
    # for metrics and auxiliary terms. Non-policy losses use the legacy reducer
    # axis directly.
    reducer_per_token_loss = args.calculate_per_token_loss and args.loss_type != "policy_loss"
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        calculate_per_token_loss=reducer_per_token_loss,
        qkv_format=args.qkv_format,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    func = get_loss_function(args)

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Forces autograd to traverse the full graph on every rank to avoid hang.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss_parallel_size = (
                parallel_state.intra_dp.size
                if args.true_on_policy_mode and parallel_state.is_ulysses_cp
                else parallel_state.intra_dp_cp.size
            )
            loss = loss * num_microbatches / global_batch_size * loss_parallel_size
        else:
            loss = loss / global_batch_size * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    log_dict = _build_train_log_dict(
        log,
        num_samples=num_samples,
        num_tokens=num_tokens,
        device=logits.device,
        calculate_per_token_loss=args.calculate_per_token_loss,
        metrics_token_reduced=reducer_per_token_loss,
    )

    return (
        loss,
        (
            num_tokens.to(device=logits.device)
            if args.calculate_per_token_loss
            else torch.tensor(1, device=logits.device)
        ),
        log_dict,
    )


def log_train_advantage_computation_event(rollout_data: RolloutBatch) -> None:
    if not is_event_logger_initialized():
        return

    advantages = rollout_data.get("advantages")
    witness_ids = rollout_data.get("witness_ids")
    if advantages is None or witness_ids is None:
        return

    get_event_logger().log(
        TrainAdvantageComputationEvent,
        dict(
            advantages=[x.tolist() for x in advantages],
            witness_ids=[x.tolist() for x in witness_ids],
        ),
        print_log=False,
    )
