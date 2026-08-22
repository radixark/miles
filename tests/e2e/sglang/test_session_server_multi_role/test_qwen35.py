from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1200, suite="stage-c-4-gpu-h200", labels=["sglang"])
register_rocm_ci(est_time=750, suite="nightly-stage-c-4-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3.5-35B-A3B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tito_model="qwen35",
    tp_size=2,
    enable_spec=True,
    cycles=2,
    tool_call_failure_mode="append_tool",
    # Anthropic tool-call conversion changes raw assistant serialization;
    # keep this endpoint-only formatting mismatch soft while hard gates stay at 0.
    anthropic_assistant_text_threshold=1.0,
    anthropic_intermediate_system_expectation="forbidden",
)


def test_qwen35():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen35()
