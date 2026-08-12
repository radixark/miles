import argparse
import json

import pytest
from tests.e2e.sglang.test_session_server_multi_role import _common

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
        "rollout_num_gpus_per_engine": 2,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 8,
        "n_samples_per_prompt": 4,
        "session_verify_cycles": 3,
        "tool_call_failure_mode": "rollback",
        "sglang_reasoning_parser": "qwen3",
        "sglang_tool_call_parser": "qwen25",
        "sglang_context_length": None,
        "sglang_cuda_graph_backend_prefill": None,
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


def test_namespace_to_train_args_defaults_to_session_server_v2():
    train_args = _build_args()

    assert "--use-session-server v2" in train_args


def test_namespace_to_train_args_allows_session_server_v1():
    train_args = _build_args(use_session_server="v1")

    assert "--use-session-server v1" in train_args


def test_namespace_to_train_args_has_no_append_role_policy_flag():
    train_args = _build_args()

    assert "allowed-append-roles" not in train_args


def test_namespace_to_train_args_omits_context_length_by_default():
    train_args = _build_args()

    assert "--sglang-context-length" not in train_args


def test_namespace_to_train_args_emits_model_context_length():
    train_args = _build_args(sglang_context_length=32768)

    assert "--sglang-context-length 32768" in train_args


def test_namespace_to_train_args_omits_prefill_cuda_graph_backend_by_default():
    train_args = _build_args()

    assert "--sglang-cuda-graph-backend-prefill" not in train_args


def test_namespace_to_train_args_emits_prefill_cuda_graph_backend():
    train_args = _build_args(sglang_cuda_graph_backend_prefill="disabled")

    assert "--sglang-cuda-graph-backend-prefill disabled" in train_args


@pytest.mark.parametrize(("n_samples_per_prompt", "expected_global_batch_size"), [(1, 16), (4, 64)])
def test_run_one_aligns_global_batch_size_with_sample_count(
    monkeypatch, n_samples_per_prompt, expected_global_batch_size
):
    captured = {}
    monkeypatch.setattr(_common, "run_session_verify", lambda args: captured.setdefault("args", args))
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_max_response_len=4096,
        cuda_graph_backend_prefill="disabled",
    )

    _common.run_one(config)

    assert captured["args"].global_batch_size == expected_global_batch_size
    assert captured["args"].rollout_batch_size == 16
    assert captured["args"].rollout_max_response_len == 4096
    assert captured["args"].sglang_cuda_graph_backend_prefill == "disabled"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_run_one_uses_requested_session_server_version(monkeypatch, version):
    captured = {}
    monkeypatch.setattr(_common, "run_session_verify", lambda args: captured.setdefault("args", args))
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    _common.run_one(config, session_server_version=version)

    assert captured["args"].use_session_server == version


@pytest.mark.parametrize(("n_samples_per_prompt", "expected_global_batch_size"), [(1, 8), (4, 32)])
def test_run_both_versions_splits_rollout_batch(monkeypatch, n_samples_per_prompt, expected_global_batch_size):
    captured = []
    monkeypatch.setattr(_common, "run_session_verify", lambda args: captured.append(args))
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        n_samples_per_prompt=n_samples_per_prompt,
    )

    _common.run_both_versions(config)

    assert [args.use_session_server for args in captured] == ["v1", "v2"]
    assert [args.rollout_batch_size for args in captured] == [8, 8]
    assert [args.global_batch_size for args in captured] == [expected_global_batch_size] * 2


def test_namespace_to_train_args_omits_expert_parallel_for_single_expert():
    train_args = _build_args()

    assert "--sglang-expert-parallel-size" not in train_args


def test_namespace_to_train_args_emits_expert_parallel_for_moe():
    train_args = _build_args(sglang_ep_size=8)

    assert "--sglang-expert-parallel-size 8" in train_args


def test_namespace_to_train_args_omits_speculative_decoding_by_default():
    train_args = _build_args()

    assert "--sglang-speculative-" not in train_args


def test_namespace_to_train_args_enables_eagle_speculative_decoding():
    train_args = _build_args(enable_spec=True)

    assert "--sglang-speculative-algorithm EAGLE" in train_args
    assert "--sglang-speculative-num-steps 2" in train_args
    assert "--sglang-speculative-eagle-topk 1" in train_args
    assert "--sglang-speculative-num-draft-tokens 3" in train_args


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
