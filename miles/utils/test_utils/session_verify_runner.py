"""Boot a real ``miles`` rollout pipeline + run the multi-role TITO driver.

Used by both consumers:

- pytest e2e: ``tests/e2e/sglang/test_session_server_multi_role/`` (one test
  file per model family)
- CLI: ``scripts/tools/verify_session_tito_tokenizer.py``

Both forms run the same ``execute_train(--debug-rollout-only)`` path: full miles
pipeline (sglang + miles-router with session support) is launched, ``train`` is
skipped, and the rollout drives ``session_verify_agent.run_agent`` against the
session server.

Args flow through miles' canonical ``parse_args`` Namespace.

# Backend choice

``execute_train`` asserts ``("--train-backend fsdp" in train_args) == (megatron_model_type is None)``,
so the ``fsdp`` + ``None`` pair is the only consistent way to skip megatron init
in ``--debug-rollout-only`` mode.  We use that.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from typing import Any

import miles.utils.external_utils.command_utils as U
from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

logger = logging.getLogger(__name__)

# Soft cap on how many samples may report any assistant_text mismatch.  Hard
# mismatch types (special_token_count / special_token_type / non_assistant_text)
# are asserted per-sample inside the agent wrapper — those must be 0.
ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD = 0.2

PROMPT_DATA_PATH = "/root/datasets/session_multi_role_verify.jsonl"
LOCAL_MODELS_ROOT = "/root/models"

# The driver agent synthesizes its own initial conversation, but the rollout
# pipeline still needs a non-empty prompt-data file as input.  This placeholder
# matches the agent's own initial prompt so the prompt is well-formed even if
# something downstream inspects it.
_PLACEHOLDER_PROMPT_RECORD = {
    "messages": [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"},
    ],
}

SESSION_VERIFY_INVARIANT_ARGS: dict[str, Any] = {
    "prompt_data": PROMPT_DATA_PATH,
    "input_key": "messages",
    "num_rollout": 1,
    "rollout_batch_size": 16,
    "rollout_max_response_len": 8192,
    "rollout_temperature": 0.7,
    "global_batch_size": 64,
    "rm_type": "random",
    "custom_generate_function_path": "miles.utils.test_utils.session_verify_agent.generate",
    "custom_agent_function_path": "miles.utils.test_utils.session_verify_agent.run_agent",
    "use_session_server": True,
    "debug_rollout_only": True,
    "ci_test": True,
    "colocate": True,
    "train_backend": "fsdp",
    "sglang_expert_parallel_size": 1,
}


def session_verify_extras(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """``add_custom_arguments`` hook for ``miles.utils.arguments.parse_args``.

    Adds the wrapper-only ``--assistant-text-threshold`` knob (a post-process
    gate on the per-sample metrics JSONL, NOT in ``train_args``) and applies
    session-verify invariants as parser defaults — user CLI still overrides
    these via the canonical miles flags.
    """
    parser.add_argument(
        "--assistant-text-threshold",
        type=float,
        default=ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD,
        help=(
            "Soft threshold for assistant_text mismatch ratio.  Default 0.1.  "
            "Raise to 1.0 for families whose upstream sglang reasoning parser "
            "is known to roundtrip imperfectly (e.g. nemotron_3 keeps a "
            "trailing newline in reasoning_content) — hard mismatches still "
            "gate.  Post-process gate on per-sample JSONL metrics; not "
            "forwarded to ``train_args``."
        ),
    )
    parser.set_defaults(**SESSION_VERIFY_INVARIANT_ARGS)
    return parser


def _ensure_prompt_data() -> str:
    os.makedirs(os.path.dirname(PROMPT_DATA_PATH), exist_ok=True)
    with open(PROMPT_DATA_PATH, "w") as f:
        f.write(json.dumps(_PLACEHOLDER_PROMPT_RECORD) + "\n")
    return PROMPT_DATA_PATH


def _ensure_model_downloaded(hf_checkpoint: str) -> str:
    """Return a local model path, downloading HF repos when needed.

    Lets callers pass either a HuggingFace repo id (downloaded under
    ``/root/models/<short-name>``) or an existing local checkpoint path
    (returned as-is, no download).
    """
    if os.path.exists(hf_checkpoint):
        return hf_checkpoint

    short = hf_checkpoint.split("/")[-1]
    local_dir = os.path.join(LOCAL_MODELS_ROOT, short)
    os.makedirs(LOCAL_MODELS_ROOT, exist_ok=True)
    U.exec_command(f"hf download {hf_checkpoint} --local-dir {local_dir}")
    return local_dir


def _clear_proxy_env() -> None:
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)


def namespace_to_train_args(ns: argparse.Namespace) -> str:
    """Serialize a fully-shaped Namespace into the ``train_args`` string.

    Reads miles-canonical field names off ``ns``; emits the exact flag set
    ``execute_train`` re-parses downstream.  ``actor_num_nodes`` is written
    explicitly from ``ns.actor_num_nodes`` (NOT defaulted at the serializer
    level) so the runner stays pinned to whatever the caller's Namespace
    declared, regardless of any drift in miles' upstream default.
    """
    allowed_roles_arg = " ".join(ns.tito_allowed_append_roles)
    parts: list[str] = [
        f"--hf-checkpoint {ns.hf_checkpoint}",
        f"--prompt-data {ns.prompt_data}",
        f"--input-key {ns.input_key}",
        f"--num-rollout {ns.num_rollout}",
        f"--rollout-batch-size {ns.rollout_batch_size}",
        f"--n-samples-per-prompt {ns.n_samples_per_prompt}",
        f"--rollout-max-response-len {ns.rollout_max_response_len}",
        f"--rollout-temperature {ns.rollout_temperature}",
        f"--global-batch-size {ns.global_batch_size}",
        f"--custom-generate-function-path {ns.custom_generate_function_path}",
        f"--custom-agent-function-path {ns.custom_agent_function_path}",
        f"--session-verify-cycles {ns.session_verify_cycles}",
        f"--tool-call-failure-mode {ns.tool_call_failure_mode}",
        f"--tito-model {ns.tito_model}",
        f"--tito-allowed-append-roles {allowed_roles_arg}",
        f"--rollout-num-gpus-per-engine {ns.rollout_num_gpus_per_engine}",
        f"--sglang-reasoning-parser {ns.sglang_reasoning_parser}",
        f"--rm-type {ns.rm_type}",
        f"--actor-num-nodes {ns.actor_num_nodes}",
        f"--actor-num-gpus-per-node {ns.actor_num_gpus_per_node}",
        f"--train-backend {ns.train_backend}",
    ]
    if ns.sglang_tool_call_parser:
        parts.append(f"--sglang-tool-call-parser {ns.sglang_tool_call_parser}")
    # DeepSeek V3.2 (and other NSA/MoE archs) requires expert-parallel > 1 in
    # sglang; the default is 1, which is fatal at engine init.  Only emit the
    # flag when the caller asks for ep>1 so single-expert models stay untouched.
    if ns.sglang_expert_parallel_size > 1:
        parts.append(f"--sglang-expert-parallel-size {ns.sglang_expert_parallel_size}")
    if ns.use_session_server:
        parts.append("--use-session-server")
    if ns.debug_rollout_only:
        parts.append("--debug-rollout-only")
    if ns.ci_test:
        parts.append("--ci-test")
    if ns.colocate:
        parts.append("--colocate")
    return " ".join(parts) + " "


def run_session_verify(args: argparse.Namespace) -> None:
    """Boot ``miles`` rollout pipeline and run the multi-role driver.

    Returns nothing on success; raises ``AssertionError`` on TITO mismatch
    (HTTP 500 from server-side prefix check) or coverage shortfall (raised by
    ``session_verify_agent.generate``).

    ``args`` MUST be a fully-shaped Namespace carrying miles-canonical field
    names plus the session-verify-specific fields (``session_verify_cycles``,
    ``tool_call_failure_mode``, ``assistant_text_threshold``).  Build it via
    ``parse_args(add_custom_arguments=session_verify_extras)`` for the CLI
    path or by spreading ``SESSION_VERIFY_INVARIANT_ARGS`` into
    ``argparse.Namespace(...)`` for tests.

    Mutates ``args`` in three places before composing train_args:
    - ``args.sglang_reasoning_parser`` / ``args.sglang_tool_call_parser`` are
      resolved against the TITO subclass's bound values via
      ``resolve_reasoning_and_tool_call_parser`` — caller-passed values that
      disagree with the bound values raise ``ValueError`` here, before any
      GPU work starts.
    - ``args.hf_checkpoint`` is replaced with the local download path so the
      composed train_args points at the downloaded model, not the HF id.
    - ``args.tito_allowed_append_roles`` is normalized (lowercase, dedup,
      ensure ``'tool'`` is in) to match the schedule contract in
      ``session_verify_agent._SUPPORTED_ROLE_SURFACES``.
    """
    args.sglang_reasoning_parser, args.sglang_tool_call_parser = resolve_reasoning_and_tool_call_parser(
        args.tito_model, args.sglang_reasoning_parser, args.sglang_tool_call_parser
    )
    args.tito_allowed_append_roles = sorted(set(r.lower() for r in args.tito_allowed_append_roles) | {"tool"})

    _ensure_prompt_data()
    _clear_proxy_env()
    args.hf_checkpoint = _ensure_model_downloaded(args.hf_checkpoint)

    train_args = namespace_to_train_args(args)

    # Per-sample token-seq metrics file: rollout workers append one JSONL line
    # per sample inside session_verify_agent.generate; we aggregate after
    # execute_train returns to apply the assistant_text soft threshold.
    metrics_fd, metrics_path = tempfile.mkstemp(prefix="session_verify_metrics_", suffix=".jsonl")
    os.close(metrics_fd)

    preserved_metrics_path = None
    try:
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            megatron_model_type=None,
            extra_env_vars={
                "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
                "MILES_TITO_MODEL": args.tito_model,
                "MILES_SESSION_VERIFY_METRICS_PATH": metrics_path,
            },
        )
        try:
            assert_session_verify_metrics(metrics_path, assistant_text_threshold=args.assistant_text_threshold)
        except AssertionError:
            preserved_metrics_path = metrics_path + ".failed"
            shutil.copy(metrics_path, preserved_metrics_path)
            logger.error("Preserved per-sample mismatch payloads at %s for post-mortem", preserved_metrics_path)
            raise
    finally:
        try:
            os.unlink(metrics_path)
        except OSError:
            pass


def assert_session_verify_metrics(metrics_path: str, *, assistant_text_threshold: float) -> None:
    """Read per-sample JSONL metrics and assert cross-sample verifier gates.

    Forbidden mismatch types (special_*, non_assistant_text) are caught
    per-sample in the agent wrapper and would have already raised by now.
    Here we only cross-check the soft assistant_text rate against the
    caller-provided threshold (per-model: some upstream sglang reasoning
    parsers — notably ``nemotron_3`` — leave a trailing ``\\n`` in
    ``reasoning_content`` that breaks the canonical roundtrip until the
    parser is patched, so those families ride at threshold=1.0).
    """
    samples_with_mismatch = 0
    total_samples = 0
    has_append_tool = False
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total_samples += 1
            has_append_tool = has_append_tool or "append_tool" in entry.get("driver_events", [])
            if entry.get("had_assistant_mismatch"):
                samples_with_mismatch += 1

    if total_samples == 0:
        raise AssertionError(
            f"Session multi-role e2e: no per-sample metrics found at {metrics_path}.  "
            "Either the rollout produced 0 samples, or the agent wrapper failed to "
            "run before any sample completed.  Check rollout logs."
        )

    if not has_append_tool:
        raise AssertionError(
            "Session multi-role e2e: no sample produced an append_tool action — "
            "the model may not be tool-calling.  Check sampling temperature, "
            "the tool spec, or parser configuration."
        )

    ratio = samples_with_mismatch / total_samples
    logger.info(
        "Token-seq metric summary: samples=%d, with_assistant_text_mismatch=%d, ratio=%.3f, threshold=%.3f",
        total_samples,
        samples_with_mismatch,
        ratio,
        assistant_text_threshold,
    )
    if ratio > assistant_text_threshold:
        raise AssertionError(
            f"Session multi-role e2e: assistant_text mismatch ratio "
            f"{samples_with_mismatch}/{total_samples}={ratio:.3f} exceeds "
            f"threshold {assistant_text_threshold}.  TITO "
            "tokenization for assistant content has drifted from the chat "
            "template's canonical render — investigate via "
            "verify_session_tito_tokenizer.py + sample-level mismatch logs."
        )
