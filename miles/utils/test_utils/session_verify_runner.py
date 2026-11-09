"""Boot a real ``miles`` rollout pipeline + run the multi-role TITO driver.

Used by both consumers:

- pytest e2e: ``tests/e2e/sglang/test_session_server_multi_role/`` (one test
  file per model family)
- CLI: ``scripts/tools/verify_session_tito_tokenizer.py``

Both forms run the same ``execute_train(--debug-rollout-only)`` path: full miles
pipeline (sglang + miles-router with session support) is launched, ``train`` is
skipped, and the rollout drives ``session_verify_agent.run_agent`` against the
session server.

CLI args are validated by miles' canonical ``parse_args`` and adapted to a local Namespace.

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
from typing import Any, Literal

from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend
from miles.utils.test_utils.session_verify_agent import SESSION_VERIFY_KIND_KEY, SessionVerifyKind
from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV

logger = logging.getLogger(__name__)

SessionWireFormat = Literal["openai", "anthropic"]

# Soft cap on how many normal-rollout samples may report any assistant_text
# mismatch.  Hard mismatch types (special_token_count / special_token_type /
# non_assistant_text) are journaled per sample and rejected by the run-level
# gate in both rollouts — those must be 0.
ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD = 0.2
# How many per-sample assistant_text examples a failed soft gate quotes, so a CI
# log shows what drifted without the preserved sidecar file.
_ASSISTANT_MISMATCH_EXAMPLES = 3

PROMPT_DATA_PATH = "/root/datasets/session_multi_role_verify.jsonl"
LOCAL_MODELS_ROOT = "/root/models"

# The driver agent synthesizes its own initial conversation, but the rollout
# pipeline still needs a non-empty prompt-data file as input.  These placeholder
# messages match the agent's own initial prompt so the prompt is well-formed
# even if something downstream inspects it.
_PLACEHOLDER_MESSAGES = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "What's the weather in Beijing?"},
]

SESSION_VERIFY_INVARIANT_ARGS: dict[str, Any] = {
    "prompt_data": PROMPT_DATA_PATH,
    "input_key": "messages",
    "rollout_batch_size": 16,
    "rollout_max_response_len": 8192,
    "rollout_temperature": 0.7,
    "global_batch_size": 64,
    "rm_type": "random",
    "custom_generate_function_path": "miles.utils.test_utils.session_verify_agent.generate",
    "custom_agent_function_path": "miles.utils.test_utils.session_verify_agent.run_agent",
    "use_session_server": "v2",
    "session_message_matcher": "strict",
    "debug_rollout_only": True,
    "ci_test": True,
    "enable_sample_ownership_checker": False,
    "colocate": True,
    "train_backend": "fsdp",
    "sglang_ep_size": 1,
    "enable_spec": False,
}


def session_verify_kinds(spec_trajectory: bool) -> tuple[SessionVerifyKind, ...]:
    """The rollouts of one verifier run, one per rollout step, in order.

    The spec rollout runs first, so the run's last step is the normal rollout,
    the one the metric-history gate judges (``steps="last"``).
    """
    if spec_trajectory:
        return (SessionVerifyKind.SPEC, SessionVerifyKind.NORMAL)
    return (SessionVerifyKind.NORMAL,)


def session_verify_extras(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """``add_custom_arguments`` hook for ``miles.utils.arguments.parse_args``.

    Adds the wrapper-only ``--assistant-text-threshold`` and
    ``--session-verify-spec-trajectory`` knobs (consumed by this runner, NOT in
    ``train_args``) and applies session-verify invariants as parser defaults —
    user CLI still overrides these via the canonical miles flags.
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
    parser.add_argument(
        "--session-verify-spec-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the spec rollout (every continuation the family's continuation roles "
            "accept or refuse) as a first rollout step before the normal rollout.  Turn "
            "off for families whose spec continuations fall outside token append-only "
            "(GLM replaces its sampled stop token)."
        ),
    )
    parser.set_defaults(**SESSION_VERIFY_INVARIANT_ARGS)
    return parser


def _ensure_prompt_data(kinds: tuple[SessionVerifyKind, ...], *, rollout_batch_size: int) -> str:
    """Write one rollout step's prompts per kind, in step order.

    The agent reads its rollout from each record's metadata; with
    ``--no-rollout-shuffle`` the data source hands the records out in file
    order, ``rollout_batch_size`` per step.
    """
    os.makedirs(os.path.dirname(PROMPT_DATA_PATH), exist_ok=True)
    with open(PROMPT_DATA_PATH, "w") as f:
        for kind in kinds:
            record = {"messages": _PLACEHOLDER_MESSAGES, "metadata": {SESSION_VERIFY_KIND_KEY: str(kind)}}
            f.writelines(json.dumps(record) + "\n" for _ in range(rollout_batch_size))
    return PROMPT_DATA_PATH


def _ensure_model_downloaded(hf_checkpoint: str, *, backend: BaseCommandBackend) -> str:
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
    backend.exec_command_cpu(f"hf download {hf_checkpoint} --local-dir {local_dir}")
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
    parts: list[str] = [
        f"--hf-checkpoint {ns.hf_checkpoint}",
        f"--prompt-data {ns.prompt_data}",
        f"--input-key {ns.input_key}",
        f"--num-rollout {len(session_verify_kinds(ns.session_verify_spec_trajectory))}",
        # The prompt file lays out one rollout step per kind; keep it in file order.
        "--no-rollout-shuffle",
        f"--rollout-batch-size {ns.rollout_batch_size}",
        f"--n-samples-per-prompt {ns.n_samples_per_prompt}",
        f"--rollout-max-response-len {ns.rollout_max_response_len}",
        f"--rollout-temperature {ns.rollout_temperature}",
        f"--global-batch-size {ns.global_batch_size}",
        f"--custom-generate-function-path {ns.custom_generate_function_path}",
        f"--custom-agent-function-path {ns.custom_agent_function_path}",
        f"--session-verify-cycles {ns.session_verify_cycles}",
        f"--tito-model {ns.tito_model}",
        f"--session-message-matcher {ns.session_message_matcher}",
        f"--rollout-num-gpus-per-engine {ns.rollout_num_gpus_per_engine}",
        f"--sglang-reasoning-parser {ns.sglang_reasoning_parser}",
        f"--rm-type {ns.rm_type}",
        f"--actor-num-nodes {ns.actor_num_nodes}",
        f"--actor-num-gpus-per-node {ns.actor_num_gpus_per_node}",
        f"--train-backend {ns.train_backend}",
    ]
    if (
        getattr(ns, "rollout_num_gpus", None) is not None
    ):  # config-access-exempt: launcher helper accepts namespaces from different CLI providers
        parts.append(f"--rollout-num-gpus {ns.rollout_num_gpus}")
    if ns.sglang_tool_call_parser:
        parts.append(f"--sglang-tool-call-parser {ns.sglang_tool_call_parser}")
    if ns.sglang_context_length is not None:
        parts.append(f"--sglang-context-length {ns.sglang_context_length}")
    if ns.sglang_kv_cache_dtype is not None:
        parts.append(f"--sglang-kv-cache-dtype {ns.sglang_kv_cache_dtype}")
    if ns.sglang_mamba_full_memory_ratio is not None:
        parts.append(f"--sglang-mamba-full-memory-ratio {ns.sglang_mamba_full_memory_ratio}")
    if (
        getattr(ns, "sglang_cuda_graph_max_bs_decode", None) is not None
    ):  # config-access-exempt: launcher helper accepts namespaces from different CLI providers
        parts.append(f"--sglang-cuda-graph-max-bs-decode {ns.sglang_cuda_graph_max_bs_decode}")
    if ns.sglang_cuda_graph_backend_prefill is not None:
        parts.append(f"--sglang-cuda-graph-backend-prefill {ns.sglang_cuda_graph_backend_prefill}")
    # DeepSeek V3.2 (and other NSA/MoE archs) requires expert-parallel > 1 in
    # sglang; the default is 1, which is fatal at engine init.  Only emit the
    # flag when the caller asks for ep>1 so single-expert models stay untouched.
    if ns.sglang_ep_size > 1:
        parts.append(f"--sglang-expert-parallel-size {ns.sglang_ep_size}")
    if ns.enable_spec:
        parts.extend(
            [
                "--sglang-speculative-algorithm EAGLE",
                "--sglang-speculative-num-steps 2",
                "--sglang-speculative-eagle-topk 1",
                "--sglang-speculative-num-draft-tokens 3",
            ]
        )
    if (
        getattr(ns, "anthropic_intermediate_system_expectation", None) is not None
    ):  # config-access-exempt: launcher helper accepts namespaces from different CLI providers
        parts.append("--anthropic-intermediate-system-expectation " f"{ns.anthropic_intermediate_system_expectation}")
    if ns.use_session_server:
        # Preserve an explicit version string ("v2"); a bare True stays the bare flag.
        if isinstance(ns.use_session_server, str):
            parts.append(f"--use-session-server {ns.use_session_server}")
        else:
            parts.append("--use-session-server")
    if ns.debug_rollout_only:
        parts.append("--debug-rollout-only")
    if ns.ci_test:
        parts.append("--ci-test")
    if ns.ci_tito_special_token_count_threshold:
        parts.append(f"--ci-tito-special-token-count-threshold {ns.ci_tito_special_token_count_threshold}")
    if not ns.enable_sample_ownership_checker:
        parts.append("--no-enable-sample-ownership-checker")
    if ns.colocate:
        parts.append("--colocate")
    return " ".join(parts) + " "


def _session_verify_env(
    args: argparse.Namespace, metrics_path: str, *, wire_format: SessionWireFormat
) -> dict[str, str]:
    env = {
        "MILES_TITO_MODEL": args.tito_model,
        "MILES_SESSION_VERIFY_METRICS_PATH": metrics_path,
    }
    if wire_format == "anthropic":
        # This run still uses the local metrics file, but must not overwrite
        # another endpoint's CI-history series under the same v2 metric key.
        env[RECORD_DIR_ENV] = ""
    return env


def run_session_verify(args: argparse.Namespace, *, wire_format: SessionWireFormat = "openai") -> None:
    """Boot ``miles`` rollout pipeline and run the session-verification driver.

    Returns nothing on success; raises ``AssertionError`` on TITO mismatch
    (HTTP 500 from server-side prefix check), an unmet spec expectation, or a
    coverage shortfall raised by the selected generate wrapper.

    ``args`` MUST be a fully-shaped Namespace carrying miles-canonical field
    names plus the session-verify-specific fields (``session_verify_cycles``,
    ``session_verify_spec_trajectory``, ``session_verify_gate``,
    ``assistant_text_threshold``).  Build it via
    the CLI entrypoint's ``_parse_session_verify_args()`` adapter for the CLI
    path or by spreading ``SESSION_VERIFY_INVARIANT_ARGS`` into
    ``argparse.Namespace(...)`` for tests.

    Mutates ``args`` in two places before composing train_args:
    - ``args.sglang_reasoning_parser`` / ``args.sglang_tool_call_parser`` are
      resolved against the TITO subclass's bound values via
      ``resolve_reasoning_and_tool_call_parser`` — caller-passed values that
      disagree with the bound values raise ``ValueError`` here, before any
      GPU work starts.
    - ``args.hf_checkpoint`` is replaced with the local download path so the
      composed train_args points at the downloaded model, not the HF id.
    """
    if wire_format not in ("openai", "anthropic"):
        raise ValueError(f"unsupported session verification wire format: {wire_format}")
    if wire_format == "anthropic" and args.session_verify_spec_trajectory:
        raise ValueError("the Anthropic session verification agent runs only the normal rollout")
    kinds = session_verify_kinds(args.session_verify_spec_trajectory)

    backend = command_utils.default_config().create_backend()
    args.sglang_reasoning_parser, args.sglang_tool_call_parser = resolve_reasoning_and_tool_call_parser(
        args.tito_model, args.sglang_reasoning_parser, args.sglang_tool_call_parser
    )
    _ensure_prompt_data(kinds, rollout_batch_size=args.rollout_batch_size)
    _clear_proxy_env()
    args.hf_checkpoint = _ensure_model_downloaded(args.hf_checkpoint, backend=backend)

    train_args = namespace_to_train_args(args)

    # Per-sample token-seq metrics file: rollout workers append one JSONL line
    # per sample inside the selected generate wrapper; we aggregate after
    # execute_train returns to apply the assistant_text soft threshold.
    metrics_fd, metrics_path = tempfile.mkstemp(prefix="session_verify_metrics_", suffix=".jsonl")
    os.close(metrics_fd)

    try:
        backend.execute_train(
            train_args=train_args,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            megatron_model_type=None,
            extra_env_vars=_session_verify_env(args, metrics_path, wire_format=wire_format),
        )
        assert_session_verify_metrics(
            metrics_path,
            kinds=kinds,
            assistant_text_threshold=args.assistant_text_threshold,
            special_token_count_threshold=args.ci_tito_special_token_count_threshold,
            require_append_tool=wire_format == "openai",
            gate=args.session_verify_gate,
        )
    except Exception:
        preserved_metrics_path = metrics_path + ".failed"
        try:
            shutil.copy(metrics_path, preserved_metrics_path)
        except Exception:
            logger.exception("Failed to preserve per-sample mismatch payloads at %s", preserved_metrics_path)
        else:
            logger.error("Preserved per-sample mismatch payloads at %s for post-mortem", preserved_metrics_path)
        raise
    finally:
        try:
            os.unlink(metrics_path)
        except OSError:
            pass


def assert_session_verify_metrics(
    metrics_path: str,
    *,
    kinds: tuple[SessionVerifyKind, ...],
    assistant_text_threshold: float,
    special_token_count_threshold: float = 0.0,
    require_append_tool: bool = True,
    gate: bool = True,
) -> None:
    """Read per-sample JSONL metrics and apply the cross-sample gates per rollout kind.

    A journaled verifier assertion, a rollout of *kinds* with no recorded
    sample, a spec sample whose probe got another status than the spec
    decides, and a normal rollout that never answered a tool call (when
    ``require_append_tool``) always fail.  With ``gate``, both rollouts also
    fail on hard mismatches (samples whose only hard type is
    special_token_count are tolerated up to ``special_token_count_threshold``),
    and the normal rollout on an assistant_text ratio above
    ``assistant_text_threshold``; without it those are logged only.  The spec
    rollout's assistant_text is not gated.
    """
    entries = _read_sample_records(metrics_path)
    recorded_kinds = {entry.get("kind") for entry in entries}
    if not recorded_kinds <= set(kinds):
        raise AssertionError(
            f"Session multi-role e2e: samples of kinds {sorted(recorded_kinds - set(kinds))} "
            f"in a run of {[str(kind) for kind in kinds]}"
        )
    summaries = {kind: _summarize([e for e in entries if e.get("kind") == kind]) for kind in kinds}
    for kind, summary in summaries.items():
        logger.info("Session verification %s rollout: %s", kind, summary)
        if summary["samples"] == 0:
            raise AssertionError(
                f"Session multi-role e2e: no per-sample metrics for the {kind} rollout at {metrics_path}.  "
                "Either the rollout produced 0 samples, or the agent wrapper failed to "
                "run before any sample completed.  Check rollout logs."
            )
        if summary["unmet_expectations"]:
            raise AssertionError(
                f"Session multi-role e2e: spec probes got another status than the family's continuation "
                f"roles decide in {len(summary['unmet_expectations'])} cases "
                f"(first={summary['unmet_expectations'][:3]})."
            )
        if gate:
            _assert_hard_mismatches(kind, summary, special_token_count_threshold=special_token_count_threshold)

    if SessionVerifyKind.NORMAL not in summaries:
        return
    normal = summaries[SessionVerifyKind.NORMAL]
    if require_append_tool and not normal["has_append_tool"]:
        raise AssertionError(
            "Session multi-role e2e: no normal sample produced an append_tool action — "
            "the model may not be tool-calling.  Check sampling temperature, "
            "the tool spec, or parser configuration."
        )
    ratio = normal["with_assistant_mismatch"] / normal["samples"]
    if gate and ratio > assistant_text_threshold:
        raise AssertionError(
            f"Session multi-role e2e: assistant_text mismatch ratio "
            f"{normal['with_assistant_mismatch']}/{normal['samples']}={ratio:.3f} exceeds "
            f"threshold {assistant_text_threshold}.  TITO "
            "tokenization for assistant content has drifted from the chat "
            "template's canonical render — investigate via "
            "verify_session_tito_tokenizer.py + sample-level mismatch logs.  "
            f"First examples (expected vs actual): {normal['assistant_examples']!r}"
        )


def _read_sample_records(metrics_path: str) -> list[dict]:
    """Per-sample records of the sidecar; a journaled verifier assertion fails here."""
    entries = []
    errors = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "verification_error" in entry:
                errors.append(entry["verification_error"])
            else:
                entries.append(entry)
    if errors:
        stages = sorted({error.get("stage") for error in errors if isinstance(error, dict) and error.get("stage")})
        first = errors[0]
        if isinstance(first, dict):
            first = {
                "stage": first.get("stage"),
                "type": first.get("type"),
                "message": str(first.get("message") or "")[:1000],
            }
        else:
            first = str(first)[:1000]
        raise AssertionError(
            "Session multi-role e2e: verifier assertions failed in "
            f"{len(errors)} attempted trajectories "
            f"(completed_samples={len(entries)}, stages={stages}, first={first})."
        )
    return entries


def _summarize(entries: list[dict]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(entries),
        "with_assistant_mismatch": 0,
        "assistant_examples": [],
        "with_hard_mismatch": 0,
        "with_special_token_count_only": 0,
        "hard_mismatch_count": 0,
        "hard_mismatch_types": set(),
        "hard_mismatch_example": None,
        "has_append_tool": False,
        "unmet_expectations": [],
    }
    for entry in entries:
        summary["has_append_tool"] |= "append_tool" in entry.get("driver_events", [])
        summary["unmet_expectations"] += [
            e for e in entry.get("expectations", []) if e["actual_status"] != e["expected_status"]
        ]
        if entry.get("had_assistant_mismatch"):
            summary["with_assistant_mismatch"] += 1
            if len(summary["assistant_examples"]) < _ASSISTANT_MISMATCH_EXAMPLES and entry.get(
                "assistant_mismatch_example"
            ):
                summary["assistant_examples"].append(entry["assistant_mismatch_example"])
        hard_types = entry.get("hard_mismatch_types", [])
        hard_count = entry.get("hard_mismatch_count", len(hard_types))
        if not (hard_count or hard_types):
            continue
        summary["with_hard_mismatch"] += 1
        summary["with_special_token_count_only"] += set(hard_types) == {"special_token_count"}
        summary["hard_mismatch_count"] += hard_count
        summary["hard_mismatch_types"].update(hard_types)
        if summary["hard_mismatch_example"] is None:
            example = entry.get("hard_mismatch_example")
            if isinstance(example, dict):
                example = {
                    "type": example.get("type"),
                    "segment_index": example.get("segment_index"),
                    "detail": str(example.get("detail") or "")[:500],
                }
            elif example is not None:
                example = str(example)[:500]
            summary["hard_mismatch_example"] = example
    return summary


def _assert_hard_mismatches(
    kind: SessionVerifyKind, summary: dict[str, Any], *, special_token_count_threshold: float
) -> None:
    special_token_count_ratio = summary["with_special_token_count_only"] / summary["samples"]
    if (
        summary["with_hard_mismatch"] > summary["with_special_token_count_only"]
        or special_token_count_ratio > special_token_count_threshold
    ):
        raise AssertionError(
            f"Session multi-role e2e: hard TITO mismatches found in "
            f"{summary['with_hard_mismatch']}/{summary['samples']} attempted {kind} samples "
            f"({summary['hard_mismatch_count']} mismatches, types={sorted(summary['hard_mismatch_types'])}, "
            f"first={summary['hard_mismatch_example']}).  These types must be 0 for any "
            f"TITO-correct setup (special_token_count-only samples tolerated up to "
            f"{special_token_count_threshold})."
        )
