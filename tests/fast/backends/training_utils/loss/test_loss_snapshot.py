"""Snapshot-based regression tests for loss functions.

Verifies that loss function outputs are bitwise identical to saved snapshots.
Use --snapshot to capture known-good outputs, --compare to validate against them.

Usage:
    # Capture snapshots from current code
    python -m pytest tests/fast/backends/training_utils/loss/test_loss_snapshot.py --snapshot -v

    # Validate current code against saved snapshots
    python -m pytest tests/fast/backends/training_utils/loss/test_loss_snapshot.py --compare -v
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from miles.backends.training_utils.loss import compute_advantages_and_returns, loss_function
from miles.backends.training_utils.loss_hub.corrections import icepop_function, vanilla_tis_function
from miles.backends.training_utils.loss_hub.losses import policy_loss_function, sft_loss_function, value_loss_function
from miles.backends.training_utils.loss_hub.logits import get_log_probs_and_entropy, get_values

from .loss_test_utils import (
    assert_outputs_equal,
    deep_clone,
    ensure_snapshot_dir,
    load_snapshot,
    make_args,
    make_batch,
    make_inputs,
    make_parallel_state,
    make_rollout_data,
    save_snapshot,
)

SNAPSHOT_DIR = Path(__file__).parent / "loss_snapshots"
VOCAB_SIZE = 128
SEED = 42

# ---------------------------------------------------------------------------
# Test configurations: (name, args_overrides, batch_size, prompt_lens, response_lens)
# ---------------------------------------------------------------------------

CONFIGS = [
    ("grpo_b3", dict(advantage_estimator="grpo", loss_type="policy_loss"), 3, [20, 64, 40], [10, 48, 32]),
    (
        "grpo_tis_b3",
        dict(advantage_estimator="grpo", loss_type="policy_loss", use_tis=True, get_mismatch_metrics=True),
        3,
        [20, 64, 40],
        [10, 48, 32],
    ),
    ("gspo_b1", dict(advantage_estimator="gspo", loss_type="policy_loss"), 1, [32], [16]),
    (
        "reinforce_pp_baseline_b2",
        dict(advantage_estimator="reinforce_plus_plus_baseline", loss_type="policy_loss"),
        2,
        [50, 80],
        [30, 60],
    ),
    ("opd_b2", dict(advantage_estimator="on_policy_distillation", loss_type="policy_loss"), 2, [40, 60], [20, 40]),
    ("value_loss_b2", dict(advantage_estimator="grpo", loss_type="value_loss"), 2, [30, 50], [15, 35]),
    ("sft_loss_b2", dict(advantage_estimator="grpo", loss_type="sft_loss"), 2, [64, 128], [32, 64]),
    (
        "grpo_kl_loss_b2",
        dict(advantage_estimator="grpo", loss_type="policy_loss", use_kl_loss=True, kl_loss_coef=0.1),
        2,
        [40, 60],
        [20, 40],
    ),
    # bshd format (padded sequences)
    (
        "grpo_bshd_b3",
        dict(advantage_estimator="grpo", loss_type="policy_loss", qkv_format="bshd"),
        3,
        [20, 64, 40],
        [10, 48, 32],
    ),
    ("sft_bshd_b2", dict(advantage_estimator="grpo", loss_type="sft_loss", qkv_format="bshd"), 2, [64, 128], [32, 64]),
    (
        "value_bshd_b2",
        dict(advantage_estimator="grpo", loss_type="value_loss", qkv_format="bshd"),
        2,
        [30, 50],
        [15, 35],
    ),
]


# ---------------------------------------------------------------------------
# pytest hooks & fixtures
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--snapshot", action="store_true", help="Generate snapshots from current code")
    parser.addoption("--compare", action="store_true", help="Compare current code against saved snapshots")


@pytest.fixture
def mode(request):
    snapshot = request.config.getoption("--snapshot", default=False)
    compare = request.config.getoption("--compare", default=False)
    assert not (snapshot and compare), "Cannot use --snapshot and --compare together"
    if snapshot:
        return "snapshot"
    # Default to compare mode (CI runs `pytest tests/fast` without flags)
    return "compare"


# ---------------------------------------------------------------------------
# Runners: call each function and collect outputs
# ---------------------------------------------------------------------------


def _get_sum_of_sample_mean(batch, args, parallel_state):
    from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean

    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        parallel_state,
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )


def run_compute_advantages_and_returns(args, parallel_state, inputs):
    rollout_data = make_rollout_data(inputs)
    compute_advantages_and_returns(args, parallel_state, rollout_data)
    return {"advantages": rollout_data["advantages"], "returns": rollout_data["returns"]}


def run_mismatch_helpers(args, inputs):
    total_response = sum(inputs["response_lens"])
    pg_loss = torch.randn(total_response, generator=torch.Generator().manual_seed(SEED + 1))

    def _call(fn):
        return fn(
            args=args,
            pg_loss=pg_loss.clone(),
            train_log_probs=deep_clone(inputs["log_probs"]),
            rollout_log_probs=deep_clone(inputs["rollout_log_probs"]),
            loss_masks=deep_clone(inputs["loss_masks"]),
        )

    return {"vanilla_tis": _call(vanilla_tis_function), "icepop": _call(icepop_function)}


def run_get_log_probs_and_entropy(args, parallel_state, inputs):
    return get_log_probs_and_entropy(
        deep_clone(inputs["policy_logits"]),
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        total_lengths=list(inputs["total_lens"]),
        response_lengths=list(inputs["response_lens"]),
        with_entropy=True,
        max_seq_lens=inputs.get("max_seq_lens"),
    )


def run_get_values(args, parallel_state, inputs):
    return get_values(
        deep_clone(inputs["value_logits"]),
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        total_lengths=list(inputs["total_lens"]),
        response_lengths=list(inputs["response_lens"]),
        max_seq_lens=inputs.get("max_seq_lens"),
    )


def run_loss_fn(args, parallel_state, inputs):
    loss_type = args.loss_type
    batch = make_batch(inputs, loss_type)
    logits = deep_clone(inputs["policy_logits"] if loss_type != "value_loss" else inputs["value_logits"])
    logits.requires_grad_(True)
    som = _get_sum_of_sample_mean(batch, args, parallel_state)
    fn = {"policy_loss": policy_loss_function, "value_loss": value_loss_function, "sft_loss": sft_loss_function}[
        loss_type
    ]
    loss, metrics = fn(args, parallel_state, batch, logits, som)
    loss.backward()
    result = {"loss": loss.detach(), "metrics": {k: v.detach() for k, v in metrics.items()}}
    result["logits_grad"] = logits.grad.clone()
    return result


def run_loss_function_dispatcher(args, parallel_state, inputs):
    loss_type = args.loss_type
    batch = make_batch(inputs, loss_type)
    logits = deep_clone(inputs["policy_logits"] if loss_type != "value_loss" else inputs["value_logits"])
    logits.requires_grad_(True)
    loss, normalizer, log_dict = loss_function(args, parallel_state, batch, 1, logits)
    loss.backward()
    result = {
        "loss": loss.detach(),
        "normalizer": normalizer.detach() if isinstance(normalizer, torch.Tensor) else normalizer,
        "log_dict_keys": log_dict["keys"],
        "log_dict_values": log_dict["values"].detach(),
        "logits_grad": logits.grad.clone(),
    }
    return result


def run_all(args, parallel_state, inputs):
    outputs = {}
    outputs["compute_advantages_and_returns"] = run_compute_advantages_and_returns(args, parallel_state, inputs)
    outputs["mismatch_helpers"] = run_mismatch_helpers(args, inputs)
    outputs["get_log_probs_and_entropy"] = run_get_log_probs_and_entropy(args, parallel_state, inputs)
    outputs["get_values"] = run_get_values(args, parallel_state, inputs)
    outputs["loss_fn"] = run_loss_fn(args, parallel_state, inputs)
    outputs["loss_function_dispatcher"] = run_loss_function_dispatcher(args, parallel_state, inputs)
    return outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS, ids=[c[0] for c in CONFIGS])
class TestLossSnapshot:

    def _snapshot_path(self, config):
        return SNAPSHOT_DIR / f"{config[0]}.pt"

    def _build(self, config):
        name, args_overrides, batch_size, prompt_lens, response_lens = config
        args = make_args(global_batch_size=batch_size, **args_overrides)
        parallel_state = make_parallel_state()
        inputs = make_inputs(
            seed=SEED,
            batch_size=batch_size,
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            vocab_size=VOCAB_SIZE,
            args=args,
        )
        return args, parallel_state, inputs

    def test_loss_snapshot(self, config, mode):
        args, parallel_state, inputs = self._build(config)

        if mode == "snapshot":
            path = self._snapshot_path(config)
            outputs = run_all(args, parallel_state, inputs)
            save_snapshot(path, inputs, outputs)

        elif mode == "compare":
            snapshot_dir = ensure_snapshot_dir(SNAPSHOT_DIR)
            path = snapshot_dir / f"{config[0]}.pt"
            assert path.exists(), f"Snapshot not found: {path}"
            saved_inputs, saved_outputs = load_snapshot(path)
            saved_args = Namespace(**saved_inputs["args_dict"])
            outputs = run_all(saved_args, parallel_state, saved_inputs)
            assert_outputs_equal(outputs, saved_outputs)
