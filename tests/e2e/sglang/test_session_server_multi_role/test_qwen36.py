from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=800, suite="stage-c-4-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=500, suite="nightly-stage-c-4-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3.6-35B-A3B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tito_model="qwen36",
    tp_size=2,
    enable_spec=True,
    cycles=2,
    # Qwen3.6 loops on the APPEND_TOOL sentinel: told "the previous turn did
    # not emit a tool_call, retry" after it already called and answered, it
    # re-derives that contradiction until max_tokens instead of retrying or
    # answering.  ROLLBACK re-samples the turn instead; a genuine no-tool-call
    # failure still surfaces after MAX_CONSECUTIVE_TOOL_CALL_FAILURE_ROLLBACKS.
    tool_call_failure_mode="rollback",
    # Anthropic tool-call conversion changes raw assistant serialization;
    # keep this endpoint-only formatting mismatch soft while hard gates stay at 0.
    anthropic_assistant_text_threshold=1.0,
    anthropic_intermediate_system_expectation="forbidden",
)


def test_qwen36():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen36()
