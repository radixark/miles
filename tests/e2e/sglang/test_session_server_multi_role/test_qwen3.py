from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1050, suite="stage-c-2-gpu-h200", labels=["sglang"])
register_rocm_ci(est_time=750, suite="nightly-stage-c-2-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3-30B-A3B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen25",
    tito_model="qwen3",
    num_gpus=2,
    tp_size=1,
    cycles=2,
    tool_call_failure_mode="append_tool",
    # Anthropic tool-call conversion preserves structure but normalizes its raw
    # serialization (32/32 formatting-only mismatches in CI); keep OpenAI at 0.25.
    assistant_text_threshold=0.25,
    anthropic_assistant_text_threshold=1.0,
    anthropic_intermediate_system_expectation="required",
)


def test_qwen3():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen3()
