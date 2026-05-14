"""E2E test: multi-role session-server TITO verification under real model inference.

Thin wrapper around
``miles.utils.test_utils.session_verify_runner.run_session_verify`` (driver
and coverage assertions live in ``session_verify_agent``).  Requires 8 GPUs.
"""

from tests.ci.ci_register import register_cuda_ci

# Six model families run sequentially in one file, including large TP4 models.
# Keep est_time in sync with the full sweep so the CI runner does not kill an
# otherwise progressing session-server verification before it reaches the later
# families.
register_cuda_ci(est_time=9600, suite="stage-b-sglang-8-gpu", num_gpus=8)


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
    # Soft-threshold override for assistant_text mismatch ratio.  Default
    # 0.1 matches session_verify_runner; raise per-family when an upstream
    # sglang reasoning parser is known to roundtrip imperfectly (e.g.
    # nemotron_3 keeps trailing newline in reasoning_content) so the gate
    # does not block on a documented out-of-scope issue.
    assistant_text_threshold: float = 0.1
    # Recovery mode when a TOOL_RESULT step finds the assistant emitted no
    # tool_calls.  Default "rollback" is universal (pop assistant + retry);
    # see ToolCallFailureMode for "append_tool" / "append_user" variants.
    tool_call_failure_mode: str = "rollback"


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "glm47-multi-role": ModelConfig(
        model_name="zai-org/GLM-4.7-Flash",
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        tito_model="glm47",
        allowed_append_roles=("tool", "user", "system"),
        tp_size=4,
        # Lenient template: tool message is rendered without validating that
        # the preceding assistant carries a matching tool_call.id, so the
        # APPEND_TOOL sentinel ("tool_call_id": "none") roundtrips cleanly.
        tool_call_failure_mode="append_tool",
    ),
    "qwen3-tool-user": ModelConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
        tool_call_failure_mode="append_tool",
    ),
    "qwen35-tool-user": ModelConfig(
        model_name="Qwen/Qwen3.5-35B-A3B",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        tito_model="qwen35",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
        tool_call_failure_mode="append_tool",
    ),
    "qwennext-tool-user": ModelConfig(
        model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwennext",
        allowed_append_roles=("tool", "user"),
        tp_size=4,
        cycles=2,
        tool_call_failure_mode="append_tool",
    ),
    "minimaxm2-tool": ModelConfig(
        # MiniMax-M2.7 shares the M2 chat template (]~!b[ / [e~[ / ]~b] tag
        # set) — reasoning is gated by last_user_index, so appending a new
        # ``user`` strips prior assistant <think> blocks and breaks
        # append-only.  Only the {tool} surface is registered upstream, so
        # allowed_append_roles is limited to ("tool",).
        model_name="MiniMaxAI/MiniMax-M2.7",
        reasoning_parser="minimax-append-think",
        tool_call_parser="minimax-m2",
        tito_model="minimax_m2",
        allowed_append_roles=("tool",),
        tp_size=4,
        cycles=2,
    ),
    "minimax-m2-tool-user": ModelConfig(
        # MiniMax-M2 (~225GB fp8 native, 256 experts x 8 active, 62 layers).
        # num_key_value_heads=8, so tp_size in {1, 2, 4, 8}; tp=2 is the
        # smallest that fits (one H200 holds 140GB, model alone is ~225GB).
        # cycles=2 to keep wall-time bounded given the 192K context budget.
        #
        # Surface is {tool, user}: M2's chat template gates reasoning_content
        # on last_user_index, so a scheduled USER_FOLLOWUP step strips prior
        # <think> blocks — that's a documented template behavior; the TITO
        # subclass tolerates it via the same last-user gating in its segment
        # boundary computation.
        #
        # tool_call_failure_mode="append_user": M2's strict template hard-
        # asserts that any ``tool`` role MUST follow an assistant with
        # non-empty ``tool_calls``, so APPEND_TOOL would be server-rejected.
        # Splicing a user-role parse-failure message gives the model a clean
        # retry hint without breaking the tool-call invariant.
        #
        # assistant_text_threshold=1.0 because sglang's ``minimax`` reasoning
        # parser is bound to ``Qwen3Detector`` (parser/reasoning_parser.py),
        # which does not strip the trailing ``\n`` from the captured
        # reasoning_text (the newline that immediately precedes ``</think>``).
        # The M2 chat template re-renders ``<think>\n{reasoning_content}\n</think>``
        # without stripping when ``reasoning_content`` is provided as a string
        # field, so canonical render gains an extra ``\n`` before ``</think>``
        # vs. the model's raw decode.  Same documented upstream-parser issue
        # as the nemotron3 row below; hard mismatches (special_token_count /
        # special_token_type / non_assistant_text) still gate at 0.
        model_name="MiniMaxAI/MiniMax-M2",
        reasoning_parser="minimax",
        tool_call_parser="minimax-m2",
        tito_model="minimax_m2",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
        assistant_text_threshold=1.0,
        tool_call_failure_mode="append_user",
    ),
    "nemotron3-tool-user": ModelConfig(
        # Nemotron-3-Super-120B-A12B-BF16 (~240GB bf16, A12B activated).
        # num_attention_heads=32, num_key_value_heads=2 — same KV-bottleneck
        # as Qwen3-Next, so tp_size=2 is the safe ceiling.  Tool calls use
        # the same <tool_call><function=...><parameter=...> XML wrapping as
        # Qwen3.5, so qwen3_coder is the right tool_call_parser.  The
        # nemotron_3 reasoning parser is documented (in Nemotron3TITOTokenizer)
        # to leave a trailing newline in reasoning_content — assistant_text
        # roundtrip mismatches on every plain-text turn until upstream sglang
        # is patched, so the soft threshold is relaxed to 1.0 for this row;
        # hard mismatches (special tokens / non-assistant text) still gate.
        model_name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        reasoning_parser="nemotron_3",
        tool_call_parser="qwen3_coder",
        tito_model="nemotron3",
        allowed_append_roles=("tool", "user"),
        tp_size=2,
        cycles=2,
        assistant_text_threshold=1.0,
        tool_call_failure_mode="append_tool",
    ),
}

# Default CI sweep. ``SESSION_TEST_MODEL_FAMILY`` (single family) overrides
# this list, primarily for local debugging.
CONFIGS: list[str] = list(MODEL_REGISTRY)


def _resolve_configs() -> list[str]:
    override = os.environ.get("SESSION_TEST_MODEL_FAMILY")
    if override:
        return [override]
    return list(CONFIGS)


def _get_config(model_family: str) -> ModelConfig:
    if model_family not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown SESSION_TEST_MODEL_FAMILY={model_family!r}. " f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_family]


def _run_one(model_family: str):
    cfg = _get_config(model_family)
    run_session_verify(
        hf_checkpoint=cfg.model_name,
        tito_model=cfg.tito_model,
        allowed_append_roles=list(cfg.allowed_append_roles),
        reasoning_parser=cfg.reasoning_parser,
        tool_call_parser=cfg.tool_call_parser,
        tp_size=cfg.tp_size,
        cycles=cfg.cycles,
        assistant_text_threshold=cfg.assistant_text_threshold,
        tool_call_failure_mode=cfg.tool_call_failure_mode,
    )


def test_session_server_multi_role():
    for model_family in _resolve_configs():
        print(
            f"\n{'=' * 60}\nRunning model_family: {model_family}\n{'=' * 60}\n",
            flush=True,
        )
        _run_one(model_family)


if __name__ == "__main__":
    test_session_server_multi_role()
