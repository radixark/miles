from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=600, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="zai-org/GLM-4.7-Flash",
    reasoning_parser="glm45",
    tool_call_parser="glm47",
    tito_model="glm47",
    num_gpus=2,
    tp_size=1,
    enable_spec=True,
    # Lenient template: tool message is rendered without validating that the
    # preceding assistant carries a matching tool_call.id, so the APPEND_TOOL
    # sentinel ("tool_call_id": "none") roundtrips cleanly.
    tool_call_failure_mode="append_tool",
)


def test_glm47():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_glm47()
