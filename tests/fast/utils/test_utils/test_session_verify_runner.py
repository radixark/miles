from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

from miles.utils.test_utils.session_verify_runner import build_train_args


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

    assert "--rollout-max-response-len 4096" in train_args


def test_build_train_args_allows_model_specific_rollout_max_response_len():
    train_args = _build_args(rollout_max_response_len=8192)

    assert "--rollout-max-response-len 8192" in train_args
