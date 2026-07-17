from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

register_cuda_ci(
    est_time=900,
    suite="stage-c-4-gpu-h200",
    labels=["sglang"],
    disabled="Temporarily disabled while debugging QwenNext session verifier timeout in sglang v0.5.12 bump.",
)


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen25",
    tito_model="qwennext",
    allowed_append_roles=("tool", "user"),
    tp_size=4,
    cycles=2,
    n_samples_per_prompt=1,
    tool_call_failure_mode="append_tool",
)


def test_qwennext():
    run_one(CONFIG)


if __name__ == "__main__":
    test_qwennext()
