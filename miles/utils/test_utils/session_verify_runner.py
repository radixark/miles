"""Boot a real ``miles`` rollout pipeline + run the multi-role TITO driver.

Used by both consumers:

- pytest e2e: ``tests/e2e/sglang/test_session_server_multi_role.py``
- CLI: ``scripts/tools/verify_session_tito_tokenizer.py``

Both forms run the same ``execute_train(--debug-rollout-only)`` path: full miles
pipeline (sglang + miles-router with session support) is launched, ``train`` is
skipped, and the rollout drives ``session_verify_agent.run_agent`` against the
session server.

# Backend choice

``execute_train`` asserts ``("--train-backend fsdp" in train_args) == (megatron_model_type is None)``,
so the ``fsdp`` + ``None`` pair is the only consistent way to skip megatron init
in ``--debug-rollout-only`` mode.  We use that.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Soft cap on how many samples may report any assistant_text mismatch.  Hard
# mismatch types (special_token_count / special_token_type / non_assistant_text)
# are asserted per-sample inside the agent wrapper — those must be 0.
ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD = 0.05

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


def _ensure_prompt_data() -> str:
    os.makedirs(os.path.dirname(PROMPT_DATA_PATH), exist_ok=True)
    with open(PROMPT_DATA_PATH, "w") as f:
        f.write(json.dumps(_PLACEHOLDER_PROMPT_RECORD) + "\n")
    return PROMPT_DATA_PATH


def _ensure_model_downloaded(hf_checkpoint: str) -> str:
    """Download the model to ``/root/models/<short-name>`` if not already there."""
    import miles.utils.external_utils.command_utils as U

    short = hf_checkpoint.split("/")[-1]
    local_dir = os.path.join(LOCAL_MODELS_ROOT, short)
    os.makedirs(LOCAL_MODELS_ROOT, exist_ok=True)
    U.exec_command(f"hf download {hf_checkpoint} --local-dir {local_dir}")
    return local_dir


def _clear_proxy_env() -> None:
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)


def build_train_args(
    *,
    local_model_dir: str,
    tito_model: str,
    allowed_append_roles,
    tp_size: int,
    num_gpus: int = 8,
    reasoning_parser: str,
    tool_call_parser: str | None,
    n_samples_per_prompt: int = 4,
    cycles: int = 3,
) -> str:
    """Compose the ``train_args`` string for ``execute_train``.

    Caller-supplied ``allowed_append_roles`` is forwarded verbatim to
    ``--tito-allowed-append-roles``; the agent's ``run_agent`` will read it back
    from ``args.tito_allowed_append_roles`` to pick a schedule.

    Rollout batching: ``--rollout-batch-size 16`` × ``--n-samples-per-prompt`` ×
    ``--num-rollout 1`` = ``--global-batch-size 64``.  The single placeholder
    prompt is cycled inside the batch — that's the path miles' rollout layer
    actually parallelizes on, so leave it at 16 even though the prompt-data
    file is single-record.  Setting batch-size=1 with a multi-sample n triggers
    unexpected agent re-invocation in the rollout loop.
    """
    allowed_roles_arg = " ".join(allowed_append_roles)

    ckpt_args = f"--hf-checkpoint {local_model_dir} "

    rollout_args = (
        f"--prompt-data {PROMPT_DATA_PATH} "
        "--input-key messages "
        "--num-rollout 1 "
        "--rollout-batch-size 16 "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.7 "
        "--global-batch-size 64 "
    )

    generate_args = (
        "--custom-generate-function-path "
        "miles.utils.test_utils.session_verify_agent.generate "
        "--custom-agent-function-path "
        "miles.utils.test_utils.session_verify_agent.run_agent "
        f"--session-verify-cycles {cycles} "
    )

    router_args = (
        "--use-miles-router "
        "--use-session-server "
        f"--tito-model {tito_model} "
        f"--tito-allowed-append-roles {allowed_roles_arg} "
    )

    sglang_args = f"--rollout-num-gpus-per-engine {tp_size} " f"--sglang-reasoning-parser {reasoning_parser} "
    if tool_call_parser:
        sglang_args += f"--sglang-tool-call-parser {tool_call_parser} "
    sglang_args += "--rm-type random "

    infra_args = (
        "--debug-rollout-only "
        "--ci-test "
        "--actor-num-nodes 1 "
        # ``num_gpus`` controls the actor's full-node allocation (default 8 =
        # whole node).  The sglang engine's tensor-parallel slice is the
        # separate ``tp_size`` parameter — a small model can run TP=1/2/4
        # while the test still occupies the whole node, keeping the CI lane
        # allocation stable regardless of the engine TP.
        f"--actor-num-gpus-per-node {num_gpus} "
        "--colocate "
        "--train-backend fsdp "
    )

    return ckpt_args + rollout_args + generate_args + router_args + sglang_args + infra_args


def run_session_verify(
    *,
    hf_checkpoint: str,
    tito_model: str,
    allowed_append_roles,
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
    tp_size: int = 1,
    num_gpus: int = 8,
    n_samples_per_prompt: int = 4,
    cycles: int = 3,
    assistant_text_threshold: float = ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD,
) -> None:
    """Boot ``miles`` rollout pipeline and run the multi-role driver.

    Returns nothing on success; raises ``AssertionError`` on TITO mismatch
    (HTTP 500 from server-side prefix check) or coverage shortfall (raised by
    ``session_verify_agent.generate``).

    Both parsers are resolved against the TITO subclass's bound values via
    ``resolve_reasoning_and_tool_call_parser`` — caller-passed values that
    disagree with the bound values raise ``ValueError`` here, before any GPU
    work starts.  Pass ``None`` for either to auto-resolve from the TITO
    subclass.
    """
    import miles.utils.external_utils.command_utils as U
    from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(
        tito_model, reasoning_parser, tool_call_parser
    )

    _ensure_prompt_data()
    _clear_proxy_env()
    local_model_dir = _ensure_model_downloaded(hf_checkpoint)

    train_args = build_train_args(
        local_model_dir=local_model_dir,
        tito_model=tito_model,
        allowed_append_roles=allowed_append_roles,
        tp_size=tp_size,
        num_gpus=num_gpus,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
        n_samples_per_prompt=n_samples_per_prompt,
        cycles=cycles,
    )

    # Per-sample token-seq metrics file: rollout workers append one JSONL line
    # per sample inside session_verify_agent.generate; we aggregate after
    # execute_train returns to apply the assistant_text soft threshold.
    metrics_fd, metrics_path = tempfile.mkstemp(prefix="session_verify_metrics_", suffix=".jsonl")
    os.close(metrics_fd)

    preserved_metrics_path = None
    try:
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=num_gpus,
            megatron_model_type=None,
            extra_env_vars={
                "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
                "SGLANG_E2E_MODEL_PATH": local_model_dir,
                "MILES_TITO_MODEL": tito_model,
                "MILES_SESSION_VERIFY_METRICS_PATH": metrics_path,
            },
        )
        try:
            _assert_assistant_text_mismatch_ratio(metrics_path, threshold=assistant_text_threshold)
        except AssertionError:
            import shutil

            preserved_metrics_path = metrics_path + ".failed"
            shutil.copy(metrics_path, preserved_metrics_path)
            logger.error("Preserved per-sample mismatch payloads at %s for post-mortem", preserved_metrics_path)
            raise
    finally:
        try:
            os.unlink(metrics_path)
        except OSError:
            pass


def _assert_assistant_text_mismatch_ratio(metrics_path: str, *, threshold: float) -> None:
    """Read the per-sample JSONL metrics and assert the soft threshold.

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
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total_samples += 1
            if entry.get("had_assistant_mismatch"):
                samples_with_mismatch += 1

    if total_samples == 0:
        raise AssertionError(
            f"Session multi-role e2e: no per-sample metrics found at {metrics_path}.  "
            "Either the rollout produced 0 samples, or the agent wrapper failed to "
            "run before any sample completed.  Check rollout logs."
        )

    ratio = samples_with_mismatch / total_samples
    logger.info(
        "Token-seq metric summary: samples=%d, with_assistant_text_mismatch=%d, ratio=%.3f, threshold=%.3f",
        total_samples,
        samples_with_mismatch,
        ratio,
        threshold,
    )
    if ratio > threshold:
        raise AssertionError(
            f"Session multi-role e2e: assistant_text mismatch ratio "
            f"{samples_with_mismatch}/{total_samples}={ratio:.3f} exceeds "
            f"threshold {threshold}.  TITO "
            "tokenization for assistant content has drifted from the chat "
            "template's canonical render — investigate via "
            "verify_session_tito_tokenizer.py + sample-level mismatch logs."
        )
