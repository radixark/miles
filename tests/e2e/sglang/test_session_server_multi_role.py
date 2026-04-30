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

# ---------------------------------------------------------------------------
# Model registry — one entry per (model, allowed_role) surface to verify.
# Selected by env var SESSION_TEST_MODEL_FAMILY (mirrors the legacy tool-only
# e2e knob); defaults to glm47-multi-role since GLM-4.7 is the model family
# whose ``clear_thinking=False`` auto-merge in TITOTokenizer.SUPPORTED_TEMPLATES
# only kicks in when ``user`` is in the role surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None
    tito_model: str
    allowed_append_roles: tuple[str, ...]
    # Engine tensor-parallel slice (``--rollout-num-gpus-per-engine``).
    # Picked per-model so that ``num_attention_heads % tp_size == 0`` and the
    # weights fit in ``tp_size × 143GB H200``.  Default 1 keeps small-model
    # paths working; larger models override explicitly.
    tp_size: int = 1
    cycles: int = 3


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "glm47-multi-role": ModelConfig(
        # GLM-4.7-Flash: ~100B MoE, num_attention_heads=20.  TP must divide 20,
        # so tp_size=4 (5 heads/rank); tp_size=8 fails the sglang
        # ``num_heads % attn_tp_size == 0`` assertion.
        model_name="zai-org/GLM-4.7-Flash",
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        tito_model="glm47",
        allowed_append_roles=("tool", "user", "system"),
        tp_size=4,
    ),
    "qwen3-tool-user": ModelConfig(
        # Qwen3-30B-A3B (MoE, ~3B activated, ~60GB bf16) over Qwen3-4B for
        # higher tool-call success rate — small Qwen3 emits tool_calls
        # inconsistently, which makes the cross-sample append_tool coverage
        # assertion flaky.  Fits in a single H200 so tp_size=1.
        model_name="Qwen/Qwen3-30B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        allowed_append_roles=("tool", "user"),
        tp_size=1,
        # cycles=2 keeps schedule depth × 4K response budget within Qwen3's 32K
        # context window with headroom for prompt + chat template overhead.
        # GLM-4.7-Flash has a larger context so it stays at the default 3.
        cycles=2,
    ),
    "qwen35-tool-user": ModelConfig(
        # Qwen3.5-35B-A3B (MoE, ~3B activated, ~70GB bf16). Same boundary
        # handling as Qwen3 via Qwen35TITOTokenizer; the {tool, user} row uses
        # clear_thinking=False to keep <think> history across multi-user turns.
        # Qwen3.5 emits tool calls as <tool_call><function=...><parameter=...>;
        # the qwen3_coder parser handles that XML-style wrapping (qwen25 parser
        # only understands <tool_call>{json}</tool_call>).  Fits in a single
        # H200 so tp_size=1.
        model_name="Qwen/Qwen3.5-35B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        tito_model="qwen35",
        allowed_append_roles=("tool", "user"),
        tp_size=1,
        cycles=2,
    ),
    "qwennext-tool-user": ModelConfig(
        # Qwen3-Next-80B-A3B-Thinking (MoE, ~3B activated, ~160GB bf16).
        # Doesn't fit one 143GB H200 — tp_size=2 minimum.  num_key_value_heads
        # is 2, so TP > 2 would either replicate KV or hit divisibility
        # asserts; tp_size=2 is the safe ceiling.  Thinking-only model, so
        # reasoning_parser stays qwen3 and tool_call_parser stays qwen25;
        # the {tool, user} row uses clear_thinking=False to preserve reasoning
        # across user turns.
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
