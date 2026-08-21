from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=700, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


# This suite reserves two H200s, but Nano fits one GPU and has no MTP head, so
# use TP1 without EAGLE. The pinned SGLang serves Nano through `nemotron_3`
# with `qwen3_coder`; its trailing-newline drift still requires threshold 1.0.
CONFIG = ModelConfig(
    model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    reasoning_parser="nemotron_3",
    tool_call_parser="qwen3_coder",
    tito_model="nemotron3",
    num_gpus=2,
    tp_size=1,
    cycles=2,
    assistant_text_threshold=1.0,
    tool_call_failure_mode="append_tool",
)


def test_nemotron3():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_nemotron3()
