from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=800, suite="stage-c-2-gpu-h200", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3.8-27B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tito_model="qwen38small",
    num_gpus=2,
    kv_cache_dtype="fp8_e4m3",
    mamba_full_memory_ratio=4.59,
    cycles=2,
    tool_call_failure_mode="append_tool",
    anthropic_intermediate_system_expectation="forbidden",
)


def test_qwen38small():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen38small()
