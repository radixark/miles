import json

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.utils.test_utils.session_verify_runner import _assert_session_verify_metrics, build_train_args


def _build_args(**overrides) -> str:
    kwargs = {
        "local_model_dir": "/root/models/test-model",
        "tito_model": "qwen3",
        "allowed_append_roles": ["tool", "user"],
        "tp_size": 2,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen25",
    }
    kwargs.update(overrides)
    return build_train_args(**kwargs)


def test_build_train_args_uses_default_rollout_max_response_len():
    train_args = _build_args()

    assert "--rollout-max-response-len 8192" in train_args


def test_build_train_args_allows_model_specific_rollout_max_response_len():
    train_args = _build_args(rollout_max_response_len=16384)

    assert "--rollout-max-response-len 16384" in train_args


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

    _assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1)


def test_session_verify_metrics_requires_at_least_one_append_tool(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path, [{"driver_events": ["initial", "append_user"], "had_assistant_mismatch": False}])

    with pytest.raises(AssertionError, match="no sample produced an append_tool action"):
        _assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1)
