from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

register_cuda_ci(est_time=1200, suite="stage-c-4-gpu-h200", labels=["sglang"])


# Nemotron-3-Super-120B-A12B-FP8 (~120GB fp8, A12B activated).
# num_attention_heads=32, num_key_value_heads=2.  SGLang replicates KV heads
# when tp_size > num_key_value_heads, requiring tp_size to be divisible by
# num_key_value_heads; tp=4 satisfies that and avoids 80GB-runner OOM while
# creating MoE expert weights.  FP8 also loads ~2x faster than the BF16
# variant, cutting Stage 3 wall-time.
#
# Tool calls use the same <tool_call><function=...><parameter=...> XML
# wrapping as Qwen3.5, so qwen3_coder is the right tool_call_parser.  The
# nemotron_3 reasoning parser is documented (in Nemotron3TITOTokenizer) to
# leave a trailing newline in reasoning_content — assistant_text roundtrip
# mismatches on every plain-text turn until upstream sglang is patched, so
# the soft threshold is relaxed to 1.0 for this row; hard mismatches
# (special tokens / non-assistant text) still gate.
CONFIG = ModelConfig(
    model_name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    reasoning_parser="nemotron_3",
    tool_call_parser="qwen3_coder",
    tito_model="nemotron3",
    allowed_append_roles=("tool", "user"),
    tp_size=4,
    cycles=2,
    assistant_text_threshold=1.0,
    tool_call_failure_mode="append_tool",
)


def test_nemotron3():
    run_one(CONFIG)


if __name__ == "__main__":
    test_nemotron3()
