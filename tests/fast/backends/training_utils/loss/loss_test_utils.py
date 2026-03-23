"""Utilities for loss function snapshot testing.

Provides deterministic input generation, snapshot save/load, and recursive
tensor comparison. Designed to be reusable across refactors — any code path
that implements the same function signatures can be validated against snapshots.

Usage:
    # Generate inputs for a config
    parallel_state = make_parallel_state()
    args = make_args(advantage_estimator="grpo", loss_type="policy_loss")
    inputs = make_inputs(seed=42, batch_size=3, prompt_lens=[20, 64, 40],
                         response_lens=[10, 48, 32], vocab_size=128, args=args)

    # Run functions, collect outputs
    outputs = run_all_functions(args, parallel_state, inputs)

    # Save / load
    save_snapshot(path, inputs, outputs)
    loaded_inputs, loaded_outputs = load_snapshot(path)
"""

from __future__ import annotations

import copy
import subprocess
from argparse import Namespace
from pathlib import Path

import torch

from miles.backends.training_utils.parallel import ParallelState

ARTIFACTS_REPO = "https://github.com/yueming-yuan/miles-artifacts.git"
ARTIFACTS_CACHE = Path.home() / ".cache" / "miles-test-artifacts"


# ---------------------------------------------------------------------------
# ParallelState (single-process, no distributed)
# ---------------------------------------------------------------------------


def make_parallel_state() -> ParallelState:
    return ParallelState(
        dp_rank=0,
        dp_src_rank=0,
        dp_size=1,
        cp_rank=0,
        cp_size=1,
        dp_cp_rank=0,
        dp_cp_size=1,
        dp_group=None,
        dp_cp_group=None,
        dp_cp_group_gloo=None,
        cp_group=None,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        is_pp_last_stage=True,
    )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

_ARGS_DEFAULTS = dict(
    # get_responses / get_log_probs_and_entropy / get_values
    qkv_format="thd",
    rollout_temperature=1.0,
    allgather_cp=False,
    log_probs_chunk_size=-1,
    true_on_policy_mode=True,
    # compute_advantages_and_returns
    advantage_estimator="grpo",
    use_rollout_logprobs=False,
    kl_coef=0.1,
    kl_loss_type="k1",
    gamma=1.0,
    lambd=0.95,
    normalize_advantages=False,
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
    calculate_per_token_loss=False,
    # value_loss_function
    value_clip=0.2,
    # loss_function dispatcher
    global_batch_size=1,  # overridden by make_inputs
    recompute_loss_function=False,
)


def make_args(**overrides) -> Namespace:
    d = {**_ARGS_DEFAULTS, **overrides}
    return Namespace(**d)


def args_to_dict(args: Namespace) -> dict:
    return vars(args)


def args_from_dict(d: dict) -> Namespace:
    return Namespace(**d)


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


def assert_outputs_equal(actual, expected, path: str = "root"):
    """Recursively compare outputs. Tensors compared with torch.equal (bitwise)."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected Tensor, got {type(actual)}"
        assert actual.shape == expected.shape, f"{path}: shape mismatch {actual.shape} vs {expected.shape}"
        assert actual.dtype == expected.dtype, f"{path}: dtype mismatch {actual.dtype} vs {expected.dtype}"
        if not torch.equal(actual, expected):
            diff = (actual - expected).abs()
            raise AssertionError(
                f"{path}: tensors not bitwise equal. "
                f"max_diff={diff.max().item():.2e}, mean_diff={diff.mean().item():.2e}"
            )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"{path}: key mismatch. extra={set(actual.keys()) - set(expected.keys())}, "
            f"missing={set(expected.keys()) - set(actual.keys())}"
        )
        for k in expected:
            assert_outputs_equal(actual[k], expected[k], path=f"{path}[{k!r}]")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{path}: expected list/tuple, got {type(actual)}"
        assert len(actual) == len(expected), f"{path}: length mismatch {len(actual)} vs {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_outputs_equal(a, e, path=f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------


def ensure_snapshot_dir(snapshot_dir: Path) -> Path:
    """Return a directory containing .pt snapshots.

    If *snapshot_dir* already has .pt files, return it as-is.
    Otherwise, shallow-clone yueming-yuan/miles-artifacts into a cache
    directory and return the path to loss_snapshots/ inside the clone.
    """
    if snapshot_dir.exists() and any(snapshot_dir.glob("*.pt")):
        return snapshot_dir
    repo_dir = ARTIFACTS_CACHE / "repo"
    result_dir = repo_dir / "loss_snapshots"
    if result_dir.exists() and any(result_dir.glob("*.pt")):
        return result_dir
    ARTIFACTS_CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", ARTIFACTS_REPO, str(repo_dir)],
        check=True,
    )
    assert result_dir.exists(), f"Expected {result_dir} after clone"
    return result_dir


def save_snapshot(path: str | Path, inputs: dict, outputs: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": inputs, "outputs": outputs}, path)


def load_snapshot(path: str | Path) -> tuple[dict, dict]:
    data = torch.load(path, weights_only=False)
    return data["inputs"], data["outputs"]
