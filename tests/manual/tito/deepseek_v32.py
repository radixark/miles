import os

from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

CONFIG = ModelConfig(
    model_name=os.environ.get("DEEPSEEK_V32_MODEL", "deepseek-ai/DeepSeek-V3.2"),
    reasoning_parser="deepseek-v3",
    tool_call_parser="deepseekv32",
    tito_model="deepseekv32",
    allowed_append_roles=("tool",),
    num_gpus=8,
    tp_size=8,
    ep_size=8,
    enable_spec=True,
)


def test_deepseek_v32_session_tito():
    run_one(CONFIG)


if __name__ == "__main__":
    test_deepseek_v32_session_tito()
