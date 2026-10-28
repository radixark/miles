"""Deterministic inputs, args and parallel state for the loss tests."""

from __future__ import annotations

import copy
from argparse import Namespace

import torch
import torch.distributed as dist

from tests.fast.fixtures.args_fixtures import ConfigNamespace

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

# ---------------------------------------------------------------------------
# ParallelState (single-process, no distributed)
# ---------------------------------------------------------------------------


def make_parallel_state(is_pp_last_stage: bool = True) -> ParallelState:
    def _trivial_group() -> GroupInfo:
        return GroupInfo(rank=0, size=1, group=None)

    # The fused vocab-parallel CE needs a real group object; use the 1-rank
    # world group when the test initialized one.
    tp_group = dist.group.WORLD if dist.is_initialized() else None
    state = ParallelState(
        intra_dp=_trivial_group(),
        intra_dp_cp=_trivial_group(),
        cp=_trivial_group(),
        tp=GroupInfo(rank=0, size=1, group=tp_group),
        pp=_trivial_group() if is_pp_last_stage else GroupInfo(rank=0, size=2, group=None),
        ep=_trivial_group(),
        etp=_trivial_group(),
        indep_dp=_trivial_group(),
        is_pp_last_stage=is_pp_last_stage,
    )
    set_parallel_state(state)
    return state


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

_ARGS_DEFAULTS = dict(
    # get_responses / get_log_probs_and_entropy / get_values
    qkv_format="thd",
    rollout_temperature=1.0,
    allgather_cp=False,
    log_probs_chunk_size=-1,
    log_probs_backend="torch",
    true_on_policy_mode=True,
    debug_unified_grad_fused_logprob=False,
    # compute_advantages_and_returns
    advantage_estimator="grpo",
    use_rollout_logprobs=False,
    use_sampling_support_replay=False,
    skip_actor_forward_only=False,
    kl_coef=0.1,
    kl_loss_type="k1",
    gamma=1.0,
    lambd=0.95,
    normalize_advantages=False,
    # on-policy distillation (OPD); orthogonal to the advantage estimator
    use_opd=False,
    opd_type=None,
    opd_kl_coef=1.0,
    # policy_loss_function
    loss_type="policy_loss",
    eps_clip=0.2,
    eps_clip_high=0.2,
    entropy_coef=0.01,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    use_unbiased_kl=False,
    use_tis=False,
    get_mismatch_metrics=False,
    tis_clip=1.5,
    tis_clip_low=0.5,
    custom_tis_function_path=None,
    custom_pg_loss_reducer_function_path=None,
    use_opsm=False,
    opsm_delta=0.1,
    eps_clip_c=None,
    dump_details=None,
    multi_lora=False,
    custom_loss_function_path=None,
    # value_loss_function
    value_clip=0.2,
    # loss_function dispatcher
    use_dynamic_global_batch_size=False,
    recompute_loss_function=False,
)


# fields a trainer reads from its backend namespace
_BACKEND_ARGS_DEFAULTS = dict(
    bf16=False,
    fp16=False,
    calculate_per_token_loss=False,
    vocab_size=None,
    global_batch_size=1,  # overridden by make_inputs
)


def make_args(**overrides) -> Namespace:
    values = {**_ARGS_DEFAULTS, **_BACKEND_ARGS_DEFAULTS, **overrides}
    backend = {name: values.pop(name) for name in _BACKEND_ARGS_DEFAULTS}
    return ConfigNamespace(**values, train_backend="megatron", backend=Namespace(**backend))


def args_to_dict(args: Namespace) -> dict:
    flat = {name: value for name, value in vars(args).items() if name not in {"backend", "train_backend"}}
    return flat | vars(args.backend)


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def make_inputs(
    seed: int,
    batch_size: int,
    prompt_lens: list[int],
    response_lens: list[int],
    vocab_size: int,
    args: Namespace,
) -> dict:
    """Generate deterministic test inputs from a seed."""
    assert len(prompt_lens) == len(response_lens) == batch_size
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]

    g = torch.Generator()
    g.manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32)

    def randint(low, high, shape):
        return torch.randint(low, high, shape, generator=g, dtype=torch.int64)

    # Per-sample tensors
    unconcat_tokens = [randint(0, vocab_size, (tl,)) for tl in total_lens]
    log_probs = [randn((rl,)) * 2 - 3 for rl in response_lens]  # roughly [-5, -1]
    ref_log_probs = [randn((rl,)) * 2 - 3 for rl in response_lens]
    rollout_log_probs = [randn((rl,)) * 2 - 3 for rl in response_lens]
    loss_masks = [torch.ones(rl, dtype=torch.float32) for rl in response_lens]
    rewards = [torch.randn(1, generator=g).item() for _ in range(batch_size)]
    values = [randn((rl,)) for rl in response_lens]
    advantages = [randn((rl,)) for rl in response_lens]
    returns = [randn((rl,)) for rl in response_lens]
    teacher_log_probs = [randn((tl,)) * 2 - 3 for tl in total_lens]

    # Logits shape depends on qkv_format:
    #   thd:  [1, sum(total_lens), V]  — packed sequences
    #   bshd: [batch_size, max_seq_len, V] — padded per sample
    if args.qkv_format == "bshd":
        max_seq_len = max(total_lens)
        max_seq_lens = [max_seq_len] * batch_size
        policy_logits = randn((batch_size, max_seq_len, vocab_size))
        value_logits = randn((batch_size, max_seq_len, 1))
    else:
        max_seq_lens = None
        total_tokens = sum(total_lens)
        policy_logits = randn((1, total_tokens, vocab_size))
        value_logits = randn((1, total_tokens, 1))

    return dict(
        seed=seed,
        batch_size=batch_size,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        vocab_size=vocab_size,
        args_dict=args_to_dict(args),
        total_lens=total_lens,
        max_seq_lens=max_seq_lens,
        unconcat_tokens=unconcat_tokens,
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        rewards=rewards,
        values=values,
        advantages=advantages,
        returns=returns,
        teacher_log_probs=teacher_log_probs,
        policy_logits=policy_logits,
        value_logits=value_logits,
    )


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def make_rollout_data(inputs: dict) -> dict:
    """Build the rollout_data dict consumed by compute_advantages_and_returns."""
    d = dict(
        log_probs=deep_clone(inputs["log_probs"]),
        ref_log_probs=deep_clone(inputs["ref_log_probs"]),
        rollout_log_probs=deep_clone(inputs["rollout_log_probs"]),
        rewards=list(inputs["rewards"]),
        values=deep_clone(inputs["values"]),
        response_lengths=list(inputs["response_lens"]),
        loss_masks=deep_clone(inputs["loss_masks"]),
        total_lengths=list(inputs["total_lens"]),
        teacher_log_probs=deep_clone(inputs["teacher_log_probs"]),
    )
    if inputs.get("max_seq_lens") is not None:
        d["max_seq_lens"] = list(inputs["max_seq_lens"])
    return d


def make_batch(inputs: dict, loss_type: str) -> dict:
    """Build the RolloutBatch dict consumed by loss functions."""
    batch = dict(
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        response_lengths=list(inputs["response_lens"]),
        total_lengths=list(inputs["total_lens"]),
        loss_masks=deep_clone(inputs["loss_masks"]),
        log_probs=deep_clone(inputs["log_probs"]),
        ref_log_probs=deep_clone(inputs["ref_log_probs"]),
        rollout_log_probs=deep_clone(inputs["rollout_log_probs"]),
        advantages=deep_clone(inputs["advantages"]),
        returns=deep_clone(inputs["returns"]),
    )
    if inputs.get("max_seq_lens") is not None:
        batch["max_seq_lens"] = list(inputs["max_seq_lens"])
    if loss_type == "value_loss":
        batch["values"] = deep_clone(inputs["values"])
    return batch


# ---------------------------------------------------------------------------
# Deep clone / comparison
# ---------------------------------------------------------------------------


def deep_clone(obj):
    """Recursively clone tensors so in-place ops don't corrupt shared state."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: deep_clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [deep_clone(x) for x in obj]
        return type(obj)(cloned)
    return copy.copy(obj)
