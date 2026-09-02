import os
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from miles.utils.types import RolloutBatch

_POLICY_LOSS_DUMP_COUNTER = 0

# Set by the actor immediately before each micro-batch forward so a dump can name
# the actor update it belongs to. Without it the dump filename carries only a
# monotonic counter, which cannot be attributed to an optimizer step.
_DUMP_CONTEXT: dict[str, int] | None = None


def set_dump_context(*, rollout_id: int, step_id: int, micro_idx: int) -> None:
    """Record which (rollout, actor update, micro-batch) the next dump belongs to."""
    global _DUMP_CONTEXT
    _DUMP_CONTEXT = {"rollout_id": rollout_id, "step_id": step_id, "micro_idx": micro_idx}


def clear_dump_context() -> None:
    global _DUMP_CONTEXT
    _DUMP_CONTEXT = None


def policy_loss_dump_dir(args: Namespace) -> str | None:
    """Where to write policy-loss dumps, or None to stay off.

    MILES_POLICY_LOSS_DUMP is checked FIRST and deliberately: --dump-details also
    sets save_debug_event_data (arguments.py:3203-3207), which initialises the
    pydantic event logger. tracking.py:50 only unwraps torch.Tensor metrics, so a
    numpy scalar reaches model_dump_json() and raises PydanticSerializationError
    in log_rollout_data -- before the train loop runs, making the dump
    unreachable. The env var turns on this dump ALONE.
    """
    return os.environ.get("MILES_POLICY_LOSS_DUMP") or getattr(args, "dump_details", None)


def maybe_dump_policy_loss_debug(
    *,
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor] | None,
    advantages: list[torch.Tensor],
    local_loss_masks: list[torch.Tensor],
    ppo_kl: torch.Tensor,
    pg_loss: torch.Tensor,
) -> None:
    dump_dir = policy_loss_dump_dir(args)
    if dump_dir is None:
        return

    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    ctx = _DUMP_CONTEXT
    if ctx is None:
        name = f"rank_{rank}_call_{counter}.pt"
    else:
        name = "r{rollout_id}_s{step_id}_m{micro_idx}_rank{rank}.pt".format(rank=rank, **ctx)
    path = Path(dump_dir) / "policy_loss_debug" / name
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    # Token ids for the whole packed sample (prompt + response), pre-CP. The log
    # prob tensors cover the RESPONSE only, so position t of a log prob tensor is
    # token index (total_length - response_length + t) of `tokens`. Saved so the
    # dump is self-joining: --balance-data reshuffles samples across ranks, so a
    # micro-batch index cannot be mapped back to rollout order after the fact.
    unconcat_tokens = batch.get("unconcat_tokens")

    samples = []
    for index, train_lp in enumerate(train_log_probs):
        total_length = batch["total_lengths"][index]
        response_length = batch["response_lengths"][index]
        sample = {
            "index": index,
            "total_length": total_length,
            "response_length": response_length,
            "train_log_probs": to_cpu(train_lp),
            "old_log_probs": to_cpu(old_log_probs[index]),
            "advantages": to_cpu(advantages[index]),
            "local_loss_mask": to_cpu(local_loss_masks[index]),
        }
        if unconcat_tokens is not None and index < len(unconcat_tokens):
            tok = unconcat_tokens[index].detach().cpu()
            sample["tokens"] = tok
            # Stable identity for joining the same generation across the 4 actor
            # updates: the token sequence IS the generation.
            sample["token_hash"] = hash(tuple(tok.tolist()))
        if rollout_log_probs is not None:
            sample["rollout_log_probs"] = to_cpu(rollout_log_probs[index])
            if train_lp.shape == rollout_log_probs[index].shape:
                sample["train_rollout_abs_diff"] = to_cpu((train_lp - rollout_log_probs[index]).abs())
        samples.append(sample)

    torch.save(
        {
            "rank": rank,
            "call": counter,
            "rollout_id": None if ctx is None else ctx["rollout_id"],
            "step_id": None if ctx is None else ctx["step_id"],
            "micro_idx": None if ctx is None else ctx["micro_idx"],
            "use_rollout_logprobs": bool(getattr(args, "use_rollout_logprobs", False)),
            "advantage_estimator": getattr(args, "advantage_estimator", None),
            "samples": samples,
            "ppo_kl": to_cpu(ppo_kl),
            "pg_loss": to_cpu(pg_loss),
            "finite": {
                "ppo_kl": torch.isfinite(ppo_kl).all().item(),
                "pg_loss": torch.isfinite(pg_loss).all().item(),
                "train_log_probs": all(torch.isfinite(t).all().item() for t in train_log_probs),
                "old_log_probs": all(torch.isfinite(t).all().item() for t in old_log_probs),
                "advantages": all(torch.isfinite(t).all().item() for t in advantages),
            },
        },
        path,
    )
