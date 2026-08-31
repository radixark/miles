from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=2100, suite="stage-c-4-gpu-h200", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    # The official deepseek-ai/DeepSeek-V4-Flash ships mxfp4-packed routed
    # experts, which no Hopper MoE kernel can serve; this FP8 repackage is
    # what scripts/run_deepseek_v4.py serves too.  Tokenizer is byte-identical.
    model_name="sgl-project/DeepSeek-V4-Flash-FP8",
    reasoning_parser="deepseek-v4",
    tool_call_parser="deepseekv4",
    tito_model="deepseekv4",
    tp_size=4,
    # V4-Flash serving recipe (scripts/run_deepseek_v4.py): tp=4, ep=4.
    ep_size=4,
    enable_spec=True,
    cycles=2,
    assistant_text_threshold=0.05,
    # V4 sorts tool_result blocks by the preceding assistant's tool_calls
    # order, so a sentinel tool_call_id would not roundtrip; use the
    # universal rollback recovery when the model emits no tool_calls.
    tool_call_failure_mode="rollback",
    anthropic_intermediate_system_expectation="forbidden",
)


def test_deepseekv4():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_deepseekv4()
