"""Shared types and runner for multi-role session-server TITO e2e tests.

Each test file in this directory owns a single ``ModelConfig`` and drives it
through ``run_one(cfg)``.  The runner is a thin wrapper around
``miles.utils.test_utils.session_verify_runner.run_session_verify`` with the
4-GPU H200 ``num_gpus`` override applied centrally.
"""

import argparse
from dataclasses import dataclass

from miles.utils.test_utils.session_verify_runner import (
    ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD,
    SESSION_VERIFY_INVARIANT_ARGS,
    run_session_verify,
)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None
    tito_model: str
    allowed_append_roles: tuple[str, ...]
    num_gpus: int = 4
    tp_size: int = 1
    # sglang expert-parallel size.  MoE archs like DeepSeek V4 hit a fused-moe
    # shape assert at ep=1; mirror the family's serving recipe (usually =tp).
    ep_size: int = 1
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
    invariants = dict(SESSION_VERIFY_INVARIANT_ARGS)
    invariants["sglang_expert_parallel_size"] = cfg.ep_size
    args = argparse.Namespace(
        hf_checkpoint=cfg.model_name,
        tito_model=cfg.tito_model,
        tito_allowed_append_roles=list(cfg.allowed_append_roles),
        sglang_reasoning_parser=cfg.reasoning_parser,
        sglang_tool_call_parser=cfg.tool_call_parser,
        rollout_num_gpus_per_engine=cfg.tp_size,
        actor_num_nodes=1,
        actor_num_gpus_per_node=cfg.num_gpus,
        n_samples_per_prompt=cfg.n_samples_per_prompt,
        session_verify_cycles=cfg.cycles,
        tool_call_failure_mode=cfg.tool_call_failure_mode,
        assistant_text_threshold=cfg.assistant_text_threshold,
        **invariants,
    )
    run_session_verify(args=args)
