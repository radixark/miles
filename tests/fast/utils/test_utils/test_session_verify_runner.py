import argparse
import ast
import json
import os
import subprocess
from pathlib import Path

import pytest
from tests.e2e.sglang.test_session_server_multi_role import _common

from miles.utils.test_utils import session_verify_runner
from miles.utils.test_utils.session_verify_runner import (
    SESSION_VERIFY_INVARIANT_ARGS,
    assert_session_verify_metrics,
    namespace_to_train_args,
)
from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV


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
        "sglang_kv_cache_dtype": None,
        "sglang_mamba_full_memory_ratio": None,
        "sglang_cuda_graph_backend_prefill": None,
        "anthropic_intermediate_system_expectation": None,
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


def test_namespace_to_train_args_emits_session_message_matcher():
    assert "--session-message-matcher strict" in _build_args()
    assert "--session-message-matcher loose_tool_call" in _build_args(session_message_matcher="loose_tool_call")


def test_namespace_to_train_args_emits_anthropic_intermediate_system_expectation():
    assert "--anthropic-intermediate-system-expectation" not in _build_args()
    assert "--anthropic-intermediate-system-expectation required" in _build_args(
        anthropic_intermediate_system_expectation="required"
    )


def test_namespace_to_train_args_has_no_append_role_policy_flag():
    train_args = _build_args()

    assert "allowed-append-roles" not in train_args


def test_namespace_to_train_args_omits_context_length_by_default():
    train_args = _build_args()

    assert "--sglang-context-length" not in train_args


def test_namespace_to_train_args_emits_model_context_length():
    train_args = _build_args(sglang_context_length=32768)

    assert "--sglang-context-length 32768" in train_args


def test_namespace_to_train_args_omits_model_mamba_cache_config_by_default():
    train_args = _build_args()

    assert "--sglang-kv-cache-dtype" not in train_args
    assert "--sglang-mamba-full-memory-ratio" not in train_args


def test_namespace_to_train_args_emits_model_mamba_cache_config():
    train_args = _build_args(sglang_kv_cache_dtype="fp8_e4m3", sglang_mamba_full_memory_ratio=4.59)

    assert "--sglang-kv-cache-dtype fp8_e4m3" in train_args
    assert "--sglang-mamba-full-memory-ratio 4.59" in train_args


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
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.update(args=args, wire_format=wire_format),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_max_response_len=4096,
        kv_cache_dtype="fp8_e4m3",
        mamba_full_memory_ratio=4.59,
        cuda_graph_backend_prefill="disabled",
    )

    _common.run_one(config)

    assert captured["args"].global_batch_size == expected_global_batch_size
    assert captured["args"].rollout_batch_size == 16
    assert captured["args"].rollout_max_response_len == 4096
    assert captured["args"].sglang_kv_cache_dtype == "fp8_e4m3"
    assert captured["args"].sglang_mamba_full_memory_ratio == 4.59
    assert captured["args"].sglang_cuda_graph_backend_prefill == "disabled"
    assert captured["wire_format"] == "openai"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_run_one_uses_requested_session_server_version(monkeypatch, version):
    captured = {}
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.update(args=args, wire_format=wire_format),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    _common.run_one(config, session_server_version=version)

    assert captured["args"].use_session_server == version
    assert captured["wire_format"] == "openai"


def test_run_one_rejects_anthropic_on_v1():
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    with pytest.raises(ValueError, match="requires session server v2"):
        _common.run_one(config, session_server_version="v1", endpoint="anthropic")


def test_run_one_requires_explicit_anthropic_intermediate_system_expectation():
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    with pytest.raises(ValueError, match="requires an intermediate-system expectation"):
        _common.run_one(config, endpoint="anthropic")


def _config_keywords(config_path: Path) -> dict[str, ast.expr]:
    for node in ast.walk(ast.parse(config_path.read_text())):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets)
        ):
            return {keyword.arg: keyword.value for keyword in node.value.keywords if keyword.arg is not None}
    raise AssertionError(f"{config_path} has no CONFIG = ModelConfig(...) assignment")


@pytest.mark.parametrize(
    "config_path",
    sorted(Path(_common.__file__).parent.glob("test_*.py")),
    ids=lambda config_path: config_path.stem,
)
def test_every_anthropic_verified_model_declares_its_intermediate_system_contract(config_path: Path):
    """Each multi-role model config that runs the Anthropic endpoint must state its intermediate-system contract."""
    keywords = _config_keywords(config_path)

    flag = keywords.get("verify_anthropic")
    if isinstance(flag, ast.Constant) and flag.value is False:
        return

    expectation = keywords.get("anthropic_intermediate_system_expectation")
    assert expectation is not None, (
        f"{config_path.name} runs the Anthropic endpoint but declares no "
        "anthropic_intermediate_system_expectation, so run_one() raises before the lane does any work"
    )
    assert isinstance(expectation, ast.Constant)
    assert expectation.value in _common.AnthropicIntermediateSystemExpectation.__args__


@pytest.mark.parametrize(("n_samples_per_prompt", "expected_global_batch_size"), [(1, 8), (4, 32)])
@pytest.mark.parametrize(
    ("threshold_overrides", "expected_thresholds"),
    [
        ({"assistant_text_threshold": 0.25}, [0.25, 0.25, 0.25]),
        (
            {"assistant_text_threshold": 0.25, "anthropic_assistant_text_threshold": 1.0},
            [0.25, 0.25, 1.0],
        ),
    ],
)
def test_run_both_versions_adds_v2_anthropic_pass(
    monkeypatch,
    n_samples_per_prompt,
    expected_global_batch_size,
    threshold_overrides,
    expected_thresholds,
):
    captured = []
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.append((args, wire_format)),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        n_samples_per_prompt=n_samples_per_prompt,
        anthropic_intermediate_system_expectation="required",
        **threshold_overrides,
    )

    _common.run_both_versions(config)

    args = [item[0] for item in captured]
    assert [item[1] for item in captured] == ["openai", "openai", "anthropic"]
    assert [item.use_session_server for item in args] == ["v1", "v2", "v2"]
    assert [item.rollout_batch_size for item in args] == [8, 8, 8]
    assert [item.global_batch_size for item in args] == [expected_global_batch_size] * 3
    assert [item.assistant_text_threshold for item in args] == expected_thresholds
    assert [item.session_message_matcher for item in args] == ["strict", "strict", "loose_tool_call"]
    assert [item.anthropic_intermediate_system_expectation for item in args] == [None, None, "required"]
    assert [item.custom_generate_function_path for item in args] == [
        SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
        SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
        _common._ANTHROPIC_GENERATE,
    ]
    assert [item.custom_agent_function_path for item in args] == [
        SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
        SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
        _common._ANTHROPIC_AGENT,
    ]


def test_run_both_versions_can_disable_anthropic(monkeypatch):
    captured = []
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.append((args, wire_format)),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="minimax-append-think",
        tool_call_parser="minimax-m2",
        tito_model="minimax_m27",
        verify_anthropic=False,
    )

    _common.run_both_versions(config)

    assert [wire_format for _, wire_format in captured] == ["openai", "openai"]
    assert [args.use_session_server for args, _ in captured] == ["v1", "v2"]


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


def test_session_verify_metrics_can_skip_multi_role_append_tool_gate(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [{"driver_events": ["anthropic_tool_use"], "had_assistant_mismatch": False}],
    )

    assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1, require_append_tool=False)


def test_session_verify_metrics_hard_mismatch_precedes_soft_threshold(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {
                "driver_events": [],
                "had_assistant_mismatch": True,
                "hard_mismatch_count": 1,
                "hard_mismatch_types": ["special_token_count"],
                "hard_mismatch_example": {"type": "special_token_count"},
            }
        ],
    )

    with pytest.raises(AssertionError, match="hard TITO mismatches.*special_token_count"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.0)


@pytest.mark.parametrize(
    ("include_clean_sample", "stage", "message", "completed_samples"),
    [
        (True, "anthropic_session_verify_agent.generate", "expected six requests", 1),
        (False, "session_verify_agent.run_agent", "missing rollback", 0),
    ],
)
def test_session_verify_metrics_rejects_verifier_error(
    tmp_path, include_clean_sample, stage, message, completed_samples
):
    metrics_path = tmp_path / "metrics.jsonl"
    entries = [{"verification_error": {"stage": stage, "type": "AssertionError", "message": message}}]
    if include_clean_sample:
        entries.append({"driver_events": ["append_tool"], "had_assistant_mismatch": False})
    _write_metrics(metrics_path, entries)

    with pytest.raises(AssertionError, match=f"completed_samples={completed_samples}.*{message}"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=1.0)


def test_session_verify_metrics_keeps_assistant_text_soft(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [{"driver_events": ["append_tool"], "had_assistant_mismatch": True}],
    )

    assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=1.0)
    with pytest.raises(AssertionError, match="assistant_text mismatch ratio"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.0)


@pytest.mark.parametrize("failure_phase", ["execute_train", "post_gate"])
def test_run_session_verify_preserves_sidecar_on_failure(monkeypatch, tmp_path, failure_phase):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_fd = os.open(metrics_path, os.O_CREAT | os.O_RDWR)
    monkeypatch.setattr(session_verify_runner.tempfile, "mkstemp", lambda **kwargs: (metrics_fd, str(metrics_path)))
    monkeypatch.setattr(
        session_verify_runner,
        "resolve_reasoning_and_tool_call_parser",
        lambda tito_model, reasoning_parser, tool_call_parser: (reasoning_parser, tool_call_parser),
    )
    monkeypatch.setattr(session_verify_runner, "_ensure_prompt_data", lambda: None)
    monkeypatch.setattr(session_verify_runner, "_clear_proxy_env", lambda: None)
    monkeypatch.setattr(session_verify_runner, "_ensure_model_downloaded", lambda checkpoint: checkpoint)
    monkeypatch.setattr(session_verify_runner, "namespace_to_train_args", lambda args: "train args")

    def fake_execute_train(**kwargs):
        sidecar = kwargs["extra_env_vars"]["MILES_SESSION_VERIFY_METRICS_PATH"]
        with open(sidecar, "w") as f:
            f.write('{"hard_mismatch_count": 1}\n')
        if failure_phase == "execute_train":
            raise subprocess.CalledProcessError(1, "ray job submit")

    monkeypatch.setattr(session_verify_runner.U, "execute_train", fake_execute_train)
    args = argparse.Namespace(
        tito_model="qwen3",
        sglang_reasoning_parser="qwen3",
        sglang_tool_call_parser="qwen25",
        hf_checkpoint="/models/test",
        actor_num_gpus_per_node=8,
        assistant_text_threshold=0.2,
    )

    expected_error = subprocess.CalledProcessError if failure_phase == "execute_train" else AssertionError
    with pytest.raises(expected_error):
        session_verify_runner.run_session_verify(args)

    assert not metrics_path.exists()
    assert (tmp_path / "metrics.jsonl.failed").read_text() == '{"hard_mismatch_count": 1}\n'


@pytest.mark.parametrize(("wire_format", "disables_history"), [("openai", False), ("anthropic", True)])
def test_session_verify_env_isolates_anthropic_from_openai_history(wire_format, disables_history):
    args = argparse.Namespace(tito_model="qwen3")

    env = session_verify_runner._session_verify_env(args, "/tmp/metrics.jsonl", wire_format=wire_format)

    assert env["MILES_TITO_MODEL"] == "qwen3"
    assert env["MILES_SESSION_VERIFY_METRICS_PATH"] == "/tmp/metrics.jsonl"
    assert (RECORD_DIR_ENV in env) is disables_history
    if disables_history:
        assert env[RECORD_DIR_ENV] == ""
