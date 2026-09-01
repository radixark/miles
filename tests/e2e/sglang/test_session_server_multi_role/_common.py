"""Shared types and runner for multi-role session-server TITO e2e tests.

Each test file in this directory owns a single ``ModelConfig`` and drives it
through ``run_both_versions(cfg)``. Each model runs OpenAI on session v1 and
v2, then Anthropic Messages on v2 unless that model explicitly disables the
unsupported endpoint. The runner applies the model-specific GPU topology centrally.
"""

import argparse
from dataclasses import dataclass
from typing import Literal

from miles.utils.test_utils.session_verify_runner import (
    ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD,
    SESSION_VERIFY_INVARIANT_ARGS,
    run_session_verify,
)

SessionServerVersion = Literal["v1", "v2"]
SessionEndpoint = Literal["openai", "anthropic"]
AnthropicIntermediateSystemExpectation = Literal["required", "forbidden"]
_SESSION_SERVER_VERSIONS: tuple[SessionServerVersion, ...] = ("v1", "v2")
_SESSION_RUNS: tuple[tuple[SessionServerVersion, SessionEndpoint], ...] = (
    ("v1", "openai"),
    ("v2", "openai"),
    ("v2", "anthropic"),
)
_ANTHROPIC_GENERATE = "miles.utils.test_utils.anthropic_session_verify_agent.generate"
_ANTHROPIC_AGENT = "miles.utils.test_utils.anthropic_session_verify_agent.run_agent"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    reasoning_parser: str
    tool_call_parser: str | None
    tito_model: str
    num_gpus: int = 4
    tp_size: int = 1
    context_length: int | None = None
    kv_cache_dtype: str | None = None
    mamba_full_memory_ratio: float | None = None
    rollout_max_response_len: int = SESSION_VERIFY_INVARIANT_ARGS["rollout_max_response_len"]
    cuda_graph_backend_prefill: str | None = None
    # sglang expert-parallel size.  MoE archs like DeepSeek V4 hit a fused-moe
    # shape assert at ep=1; mirror the family's serving recipe (usually =tp).
    ep_size: int = 1
    enable_spec: bool = False
    cycles: int = 3
    n_samples_per_prompt: int = 4
    # Soft-threshold override for assistant_text mismatch ratio.  Default
    # mirrors session_verify_runner; raise per-family when an upstream sglang
    # reasoning parser is known to roundtrip imperfectly (e.g. nemotron_3
    # keeps trailing newline in reasoning_content) so the gate does not
    # block on a documented out-of-scope issue.
    assistant_text_threshold: float = ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD
    # Optional Anthropic-only override; None inherits the per-model threshold.
    anthropic_assistant_text_threshold: float | None = None
    # Endpoint capability gate; unsupported families still run both OpenAI versions.
    verify_anthropic: bool = True
    # Required per-model Anthropic E2E contract.  The agent still derives the
    # live capability from the route and fixed template, then checks it against
    # this expectation before issuing a request with an intermediate system.
    anthropic_intermediate_system_expectation: AnthropicIntermediateSystemExpectation | None = None
    # Recovery mode when a TOOL_RESULT step finds the assistant emitted no
    # tool_calls.  Default "rollback" is universal (pop assistant + retry);
    # see ToolCallFailureMode for "append_tool" / "append_user" variants.
    tool_call_failure_mode: str = "rollback"


def run_one(
    cfg: ModelConfig,
    *,
    session_server_version: SessionServerVersion = "v2",
    endpoint: SessionEndpoint = "openai",
    rollout_batch_size: int = SESSION_VERIFY_INVARIANT_ARGS["rollout_batch_size"],
) -> None:
    if endpoint == "anthropic" and session_server_version != "v2":
        raise ValueError("Anthropic per-model verification requires session server v2")
    if endpoint == "anthropic" and cfg.anthropic_intermediate_system_expectation is None:
        raise ValueError("Anthropic per-model verification requires an intermediate-system expectation")

    invariants = dict(SESSION_VERIFY_INVARIANT_ARGS)
    invariants["use_session_server"] = session_server_version
    invariants["rollout_batch_size"] = rollout_batch_size
    if endpoint == "anthropic":
        invariants["custom_generate_function_path"] = _ANTHROPIC_GENERATE
        invariants["custom_agent_function_path"] = _ANTHROPIC_AGENT
        invariants["session_message_matcher"] = "loose_tool_call"
    # This harness produces one rollout batch, so its train-side batch divisor
    # must track the actual sample count when large-model lanes reduce samples.
    invariants["global_batch_size"] = invariants["rollout_batch_size"] * cfg.n_samples_per_prompt
    invariants["rollout_max_response_len"] = cfg.rollout_max_response_len
    invariants["sglang_cuda_graph_backend_prefill"] = cfg.cuda_graph_backend_prefill
    invariants["sglang_ep_size"] = cfg.ep_size
    invariants["sglang_context_length"] = cfg.context_length
    invariants["sglang_kv_cache_dtype"] = cfg.kv_cache_dtype
    invariants["sglang_mamba_full_memory_ratio"] = cfg.mamba_full_memory_ratio
    invariants["enable_spec"] = cfg.enable_spec
    assistant_text_threshold = cfg.assistant_text_threshold
    if endpoint == "anthropic" and cfg.anthropic_assistant_text_threshold is not None:
        assistant_text_threshold = cfg.anthropic_assistant_text_threshold
    args = argparse.Namespace(
        hf_checkpoint=cfg.model_name,
        tito_model=cfg.tito_model,
        sglang_reasoning_parser=cfg.reasoning_parser,
        sglang_tool_call_parser=cfg.tool_call_parser,
        rollout_num_gpus_per_engine=cfg.tp_size,
        actor_num_nodes=1,
        actor_num_gpus_per_node=cfg.num_gpus,
        n_samples_per_prompt=cfg.n_samples_per_prompt,
        session_verify_cycles=cfg.cycles,
        tool_call_failure_mode=cfg.tool_call_failure_mode,
        assistant_text_threshold=assistant_text_threshold,
        anthropic_intermediate_system_expectation=(
            cfg.anthropic_intermediate_system_expectation if endpoint == "anthropic" else None
        ),
        **invariants,
    )
    run_session_verify(args=args, wire_format=endpoint)


def run_both_versions(cfg: ModelConfig) -> None:
    rollout_batch_size = SESSION_VERIFY_INVARIANT_ARGS["rollout_batch_size"] // len(_SESSION_SERVER_VERSIONS)
    for version, endpoint in _SESSION_RUNS:
        if endpoint == "anthropic" and not cfg.verify_anthropic:
            continue
        run_one(cfg, session_server_version=version, endpoint=endpoint, rollout_batch_size=rollout_batch_size)
