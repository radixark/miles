import argparse
import json

import pytest

from miles.utils.test_utils.session_verify_runner import (
    SESSION_VERIFY_INVARIANT_ARGS,
    assert_session_verify_metrics,
    namespace_to_train_args,
)


def _build_args(**overrides) -> str:
    values = {
        **SESSION_VERIFY_INVARIANT_ARGS,
        "hf_checkpoint": "/root/models/test-model",
        "tito_model": "qwen3",
        "tito_allowed_append_roles": ["tool", "user"],
        "rollout_num_gpus_per_engine": 2,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 8,
        "n_samples_per_prompt": 4,
        "session_verify_cycles": 3,
        "tool_call_failure_mode": "rollback",
        "sglang_reasoning_parser": "qwen3",
        "sglang_tool_call_parser": "qwen25",
    }
    values.update(overrides)
    return namespace_to_train_args(argparse.Namespace(**values))


def test_namespace_to_train_args_uses_default_rollout_max_response_len():
    train_args = _build_args()

    assert "--rollout-max-response-len 8192" in train_args


def test_namespace_to_train_args_allows_model_specific_rollout_max_response_len():
    train_args = _build_args(rollout_max_response_len=16384)

    assert "--rollout-max-response-len 16384" in train_args


def test_namespace_to_train_args_keeps_ci_test_enabled_for_fsdp_debug_rollout():
    train_args = _build_args()

    assert "--train-backend fsdp" in train_args
    assert "--ci-test" in train_args


def test_namespace_to_train_args_omits_expert_parallel_for_single_expert():
    train_args = _build_args()

    assert "--sglang-expert-parallel-size" not in train_args


def test_namespace_to_train_args_emits_expert_parallel_for_moe():
    train_args = _build_args(sglang_expert_parallel_size=8)

    assert "--sglang-expert-parallel-size 8" in train_args


def _write_metrics(path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


def test_session_verify_metrics_accepts_cross_sample_append_tool(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {"driver_events": ["initial", "append_user"], "had_assistant_mismatch": False},
            {"driver_events": ["initial", "append_tool"], "had_assistant_mismatch": False},
        ],
    )

    assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1)


def test_session_verify_metrics_requires_at_least_one_append_tool(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path, [{"driver_events": ["initial", "append_user"], "had_assistant_mismatch": False}])

    with pytest.raises(AssertionError, match="no sample produced an append_tool action"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1)
