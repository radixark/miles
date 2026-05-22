"""Shared types and runner for multi-role session-server TITO e2e tests.

Each test file in this directory owns a single ``ModelConfig`` and drives it
through ``run_one(cfg)``.  The runner is a thin wrapper around
``miles.utils.test_utils.session_verify_runner.run_session_verify`` with the
4-GPU H200 ``num_gpus`` override applied centrally.
"""

from dataclasses import dataclass

from miles.utils.test_utils.session_verify_runner import ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD, run_session_verify


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None
    tito_model: str
    allowed_append_roles: tuple[str, ...]
    num_gpus: int = 4
    tp_size: int = 1
    cycles: int = 3
    n_samples_per_prompt: int = 4
    # Soft-threshold override for assistant_text mismatch ratio.  Default
    # mirrors session_verify_runner; raise per-family when an upstream sglang
    # reasoning parser is known to roundtrip imperfectly (e.g. nemotron_3
    # keeps trailing newline in reasoning_content) so the gate does not
    # block on a documented out-of-scope issue.
    assistant_text_threshold: float = ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD
    # Recovery mode when a TOOL_RESULT step finds the assistant emitted no
    # tool_calls.  Default "rollback" is universal (pop assistant + retry);
    # see ToolCallFailureMode for "append_tool" / "append_user" variants.
    tool_call_failure_mode: str = "rollback"


def run_one(cfg: ModelConfig) -> None:
    run_session_verify(
        hf_checkpoint=cfg.model_name,
        tito_model=cfg.tito_model,
        allowed_append_roles=list(cfg.allowed_append_roles),
        reasoning_parser=cfg.reasoning_parser,
        tool_call_parser=cfg.tool_call_parser,
        tp_size=cfg.tp_size,
        cycles=cfg.cycles,
        n_samples_per_prompt=cfg.n_samples_per_prompt,
        # run_session_verify defaults num_gpus=8 (H100 era); the suite runs on
        # 4-GPU H200, so allocate 4 actor GPUs to match the runner.
        num_gpus=cfg.num_gpus,
        assistant_text_threshold=cfg.assistant_text_threshold,
        tool_call_failure_mode=cfg.tool_call_failure_mode,
    )
