from argparse import Namespace
import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks, get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.advantages import compute_advantages, normalize_advantages
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy, get_values  # noqa: F401
from miles.backends.training_utils.loss_hub.losses import get_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import TrainAdvantageComputationEvent
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.types import RolloutBatch


def _detach_rollout_tensor_list(rollout_data: RolloutBatch, key: str) -> list[torch.Tensor] | None:
    tensors = rollout_data.get(key)
    if tensors is None:
        return None

    detached_tensors = [tensor.detach() for tensor in tensors]
    rollout_data[key] = detached_tensors
    return detached_tensors


def compute_advantages_and_returns(
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". On-policy distillation (OPD) is applied
    orthogonally on top of any estimator via `args.use_opd`. When
    `args.normalize_advantages` is True, advantages are whitened across the
    data-parallel group using masked statistics.

    Early returns on intermediate pipeline stages; the last stage also
    returns early when neither `log_probs` nor `values` is present, unless
    it is explicitly allowed to derive zero-KL shapes without the standalone
    actor pass.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    allow_missing_log_probs = args.skip_actor_forward_only and not args.use_rollout_logprobs
    log_probs_key = "rollout_log_probs" if args.use_rollout_logprobs else "log_probs"
    log_probs: list[torch.Tensor] = rollout_data.get(log_probs_key)
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # under --use-rollout-logprobs every stage holds log-probs, so ask the parallel state, not the tensors
    if not get_parallel_state().is_pp_last_stage:
        return
    if log_probs is None and values is None and not allow_missing_log_probs:
        return

    # This is the authoritative persistence boundary: scores produced before
    # the policy update are fixed training data and must not retain a graph.
    _detach_rollout_tensor_list(rollout_data, "log_probs")
    _detach_rollout_tensor_list(rollout_data, "rollout_log_probs")
    _detach_rollout_tensor_list(rollout_data, "ref_log_probs")
    _detach_rollout_tensor_list(rollout_data, "teacher_log_probs")
    log_probs = rollout_data.get(log_probs_key)
    ref_log_probs = rollout_data.get("ref_log_probs")

    if log_probs is None and values is None:
        local_masks = get_local_response_loss_masks(
            total_lengths, response_lengths, loss_masks, args.qkv_format, max_seq_lens
        )
        kl = [torch.zeros_like(mask, dtype=torch.float32) for mask in local_masks]
    elif args.kl_coef == 0 or not log_probs:
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
        max_seq_lens=max_seq_lens,
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
    num_rollouts: int | None = None,
) -> tuple[torch.Tensor, int | torch.Tensor, dict]:
    """Compute the batch-selected loss and apply Megatron scaling.

    The logging result carries scalar `keys`/`values` and any per-datum outputs returned by the loss.
    """
    parallel_state = get_parallel_state()
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
        denominators=batch.get("rollout_mask_sums", None),
    )

    func = get_loss_function(args, batch.get("loss_fn"))

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
    # fp32 sum: an fp16 logits sum can overflow to inf, and 0 * inf is nan.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum(dtype=torch.float32)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if num_rollouts is not None:
        global_batch_size = num_rollouts
    else:
        assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
        global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    # Tinker losses already carry the client's normalization
    loss_normalizer = 1 if is_multi_lora_enabled(args) else global_batch_size
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss_parallel_size = (
                parallel_state.intra_dp.size
                if args.true_on_policy_mode and parallel_state.is_ulysses_cp
                else parallel_state.intra_dp_cp.size
            )
            loss = loss * num_microbatches / loss_normalizer * loss_parallel_size
        else:
            loss = loss / loss_normalizer * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    per_datum = log.pop("per_datum", None)
    return (
        loss,
        torch.tensor(num_tokens if args.calculate_per_token_loss else 1, device=logits.device),
        {
            **({"per_datum": per_datum} if per_datum is not None else {}),
            "keys": list(log.keys()),
            "values": torch.tensor(
                [num_samples if not args.calculate_per_token_loss else num_tokens] + list(log.values()),
                device=logits.device,
            ),
        },
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
