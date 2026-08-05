from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=800, suite="stage-c-2-gpu-h200", labels=["sglang"])


# Nemotron-3-Super-120B-A12B-FP8 is ~120GB. TP2 spans both H200s and
# matches the model's two KV heads; larger TP is unnecessary for this lane.
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
    num_gpus=2,
    tp_size=2,
    enable_spec=True,
    cycles=2,
    assistant_text_threshold=1.0,
    tool_call_failure_mode="append_tool",
)


def test_nemotron3():
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_nemotron3()
