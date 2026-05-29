"""MiniMax-M2.7 covers the MiniMax-M2.x e2e surface.

MiniMax-M2.5 is intentionally omitted: M2.5 and M2.7 share tokenizer.json
(sha256-identical), arch (MiniMaxM2ForCausalLM), and the same sglang
reasoning_parser / tool_call_parser bindings, so this lane exercises the
same TITO code paths.  Stage-2 CPU coverage for M2.5 stays in
tests/fast/utils/chat_template_utils/.
"""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-h200", labels=["sglang"])


# MiniMax-M2.7 (MiniMaxM2ForCausalLM arch, 62 layers, 8 KV heads, ~215GB fp8).
# CI runs this lane on 80GB GPUs; tp=2 OOMs while SGLang allocates fp8 MoE
# weights, so use tp=4.  cycles=2 to keep wall-time bounded given the 192K
# context budget.
#
# Surface is {tool, user}: M2.7's chat template gates reasoning_content on
# last_user_index, so a scheduled USER_FOLLOWUP step strips prior <think>
# blocks — that's a documented template behavior; the fixed jinja
# (clear_thinking=False) preserves history reasoning across user turns to
# keep append-only.
#
# tool_call_failure_mode="append_user": M2.7's strict template hard-asserts
# that any ``tool`` role MUST follow an assistant with non-empty
# ``tool_calls``, so APPEND_TOOL would be server-rejected.  Splicing a
# user-role parse-failure message gives the model a clean retry hint without
# breaking the tool-call invariant.
#
# ``reasoning_parser="minimax-append-think"`` matches the binding on
# ``MinimaxM27TITOTokenizer``; ``resolve_reasoning_and_tool_call_parser``
# hard-asserts equality with the class-bound value.
CONFIG = ModelConfig(
    model_name="MiniMaxAI/MiniMax-M2.7",
    reasoning_parser="minimax-append-think",
    tool_call_parser="minimax-m2",
    tito_model="minimax_m27",
    allowed_append_roles=("tool", "user"),
    tp_size=4,
    cycles=2,
    assistant_text_threshold=0.1,
    tool_call_failure_mode="append_user",
)


def test_minimax_m27():
    run_one(CONFIG)


if __name__ == "__main__":
    test_minimax_m27()
