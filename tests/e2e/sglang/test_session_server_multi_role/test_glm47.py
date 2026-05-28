import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["sglang"])


# ROCm: bypass two SGLang/aiter paths that crash on MI350 for GLM-4.7-Flash
# (MLA + MoE). Refs sgl-project/sglang#19824, #20691 and miles PR #1126.
_ROCM_ENV = (
    {"SGLANG_ROCM_FUSED_DECODE_MLA": "0", "SGLANG_USE_AITER": "0"}
    if torch.version.hip is not None
    else {}
)


CONFIG = ModelConfig(
    model_name="zai-org/GLM-4.7-Flash",
    reasoning_parser="glm45",
    tool_call_parser="glm47",
    tito_model="glm47",
    allowed_append_roles=("tool", "user", "system"),
    tp_size=4,
    # Lenient template: tool message is rendered without validating that the
    # preceding assistant carries a matching tool_call.id, so the APPEND_TOOL
    # sentinel ("tool_call_id": "none") roundtrips cleanly.
    tool_call_failure_mode="append_tool",
    assistant_text_threshold=0.4,
    extra_env=_ROCM_ENV,
)


def test_glm47():
    run_one(CONFIG)


if __name__ == "__main__":
    test_glm47()
