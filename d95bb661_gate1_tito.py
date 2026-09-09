"""Gate 1 for run 260908-e3d52c5e: nemotron3 TITO round-trip on the BF16 checkpoint.

Drives the same harness as tests/e2e/sglang/test_session_server_multi_role/
test_nemotron3.py, but points it at the local BF16 checkpoint this run will
actually train instead of the FP8 variant the CI lane pins.

The CLI wrapper scripts/tools/verify_session_tito_tokenizer.py cannot be used
here: it calls miles' full parse_args, which under --debug-rollout-only forces
args.colocate = False, and the runner then serializes that mutated namespace
into the inner train.py invocation without a --rollout-num-gpus, so the inner
parse dies on min(8, None). Building the Namespace directly -- exactly what the
e2e harness does -- side-steps that.

assistant_text_threshold=1.0 is the documented per-family setting: sglang's
upstream nemotron_3 reasoning parser keeps a trailing newline in
reasoning_content. Hard mismatch tiers still gate.
"""

from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

CONFIG = ModelConfig(
    model_name="/scratch/260908-e3d52c5e/model",
    reasoning_parser="nemotron_3",
    tool_call_parser="qwen3_coder",
    tito_model="nemotron3",
    num_gpus=8,
    tp_size=1,
    cycles=2,
    assistant_text_threshold=1.0,
    tool_call_failure_mode="append_tool",
    anthropic_intermediate_system_expectation="required",
)


if __name__ == "__main__":
    # OpenAI wire format on session server v2 is what the training run uses.
    run_one(CONFIG, session_server_version="v2", endpoint="openai")
    print("GATE1_PASS")
