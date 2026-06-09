from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["sglang"])


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3.5-35B-A3B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tito_model="qwen35",
    allowed_append_roles=("tool", "user"),
    tp_size=2,
    cycles=2,
    tool_call_failure_mode="append_tool",
)


def test_qwen35():
    run_one(CONFIG)


if __name__ == "__main__":
    test_qwen35()
