"""E2E test: multi-role session-server TITO verification under real model inference.

Thin wrapper around
``miles.utils.test_utils.session_verify_runner.run_session_verify`` (driver
and coverage assertions live in ``session_verify_agent``).  Requires 8 GPUs.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-b-short-8-gpu", num_gpus=8)


import os
from dataclasses import dataclass

from miles.utils.test_utils.session_verify_runner import run_session_verify


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None
    tito_model: str
    allowed_append_roles: tuple[str, ...]
    tp_size: int = 1
    cycles: int = 3


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "glm47-multi-role": ModelConfig(
        model_name="zai-org/GLM-4.7-Flash",
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        tito_model="glm47",
        allowed_append_roles=("tool", "user", "system"),
        tp_size=4,
    ),
    "qwen3-tool-user": ModelConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        allowed_append_roles=("tool", "user"),
        tp_size=1,
        cycles=2,
    ),
    "qwen35-tool-user": ModelConfig(
        model_name="Qwen/Qwen3.5-35B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        tito_model="qwen35",
        allowed_append_roles=("tool", "user"),
        tp_size=1,
        cycles=2,
    ),
    "qwennext-tool-user": ModelConfig(
        model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwennext",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
    ),
}

DEFAULT_MODEL_FAMILY = "glm47-multi-role"


def _get_config() -> ModelConfig:
    family = os.environ.get("SESSION_TEST_MODEL_FAMILY", DEFAULT_MODEL_FAMILY)
    if family not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown SESSION_TEST_MODEL_FAMILY={family!r}. " f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[family]


def test_session_server_multi_role():
    cfg = _get_config()
    run_session_verify(
        hf_checkpoint=cfg.model_name,
        tito_model=cfg.tito_model,
        allowed_append_roles=list(cfg.allowed_append_roles),
        reasoning_parser=cfg.reasoning_parser,
        tool_call_parser=cfg.tool_call_parser,
        tp_size=cfg.tp_size,
        cycles=cfg.cycles,
    )


if __name__ == "__main__":
    test_session_server_multi_role()
