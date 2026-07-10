from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

register_cuda_ci(est_time=1700, suite="stage-c-4-gpu-h200", labels=["sglang"])


CONFIG = ModelConfig(
    # The official deepseek-ai/DeepSeek-V4-Flash ships mxfp4-packed routed
    # experts, which no Hopper MoE kernel can serve; this FP8 repackage is
    # what scripts/run_deepseek_v4.py serves too.  Tokenizer is byte-identical.
    model_name="sgl-project/DeepSeek-V4-Flash-FP8",
    reasoning_parser="deepseek-v4",
    tool_call_parser="deepseekv4",
    tito_model="deepseekv4",
    allowed_append_roles=("tool", "user"),
    tp_size=4,
    # V4-Flash serving recipe (scripts/run_deepseek_v4.py): tp=4, ep=4.
    ep_size=4,
    cycles=2,
    # V4 sorts tool_result blocks by the preceding assistant's tool_calls
    # order, so a sentinel tool_call_id would not roundtrip; use the
    # universal rollback recovery when the model emits no tool_calls.
    tool_call_failure_mode="rollback",
)


def test_deepseekv4():
    run_one(CONFIG)


if __name__ == "__main__":
    test_deepseekv4()
