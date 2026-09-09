from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1800, suite="stage-c-4-gpu-h200", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="thinkingmachines/Inkling-Small-NVFP4",
    reasoning_parser="inkling",
    tool_call_parser="inkling",
    tito_model="inkling",
    tp_size=4,
    # The official 1M-context H200 recipe needs TP8. This short-session lane
    # caps the context to make TP4 viable for this bounded session test.
    context_length=32768,
    rollout_max_response_len=4096,
    cuda_graph_backend_prefill="disabled",
    cycles=2,
    n_samples_per_prompt=1,
    tool_call_failure_mode="append_tool",
    anthropic_intermediate_system_expectation="required",
)


def test_inkling():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_inkling()
