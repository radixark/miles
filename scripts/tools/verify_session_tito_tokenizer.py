#!/usr/bin/env python3
"""CLI: run the multi-role TITO session-server verifier against a real model.

Boots the miles rollout pipeline (sglang + miles-router) under
``--debug-rollout-only`` and drives the schedule registered for the selected
family's ``FixedTemplate.allowed_append_roles`` capability (see
``miles.utils.test_utils.session_verify_agent``).  PASS iff every sample
completes without HTTP error from the server-side prefix check and the
custom-generate coverage assertion is satisfied.

This script is a thin entrypoint over miles' canonical ``parse_args``: all
flags are miles' canonical flags (``--rollout-num-gpus-per-engine`` instead of
the old ``--tp-size``, ``--actor-num-gpus-per-node`` instead of ``--num-gpus``,
``--n-samples-per-prompt`` instead of ``--n-samples``, ``--sglang-reasoning-parser``
instead of ``--reasoning-parser``, etc.).  The only wrapper-only knob is
``--assistant-text-threshold`` (post-process gate on per-sample metrics).

Usage examples::

    # GLM-4.7-Flash with tool + user + system surface, single-node, TP=4
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint zai-org/GLM-4.7-Flash \\
        --tito-model glm47 \\
        --sglang-reasoning-parser glm45 \\
        --sglang-tool-call-parser glm47 \\
        --rollout-num-gpus-per-engine 4

    # Qwen3-4B with tool + user surface, single-node, TP=1
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint Qwen/Qwen3-4B \\
        --tito-model qwen3 \\
        --sglang-reasoning-parser qwen3 \\
        --sglang-tool-call-parser qwen25 \\
        --rollout-num-gpus-per-engine 1

    # Multi-node example: ray cluster must already be up across N nodes
    # (e.g. via rcli / slurm) and MILES_SCRIPT_EXTERNAL_RAY=1 set so
    # execute_train skips its head-only ray start.
    MILES_SCRIPT_EXTERNAL_RAY=1 \\
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint zai-org/GLM-4.7-Flash \\
        --tito-model glm47 \\
        --sglang-reasoning-parser glm45 \\
        --sglang-tool-call-parser glm47 \\
        --rollout-num-gpus-per-engine 4 \\
        --actor-num-nodes 2 --actor-num-gpus-per-node 8
"""

from __future__ import annotations

import logging
import sys

from miles.utils.arguments import parse_args
from miles.utils.test_utils.session_verify_agent import fixed_template_append_roles, select_schedule
from miles.utils.test_utils.session_verify_runner import (
    SESSION_VERIFY_INVARIANT_ARGS,
    run_session_verify,
    session_verify_extras,
)


def _with_session_verify_defaults(argv: list[str]) -> list[str]:
    defaults = [
        "--prompt-data",
        SESSION_VERIFY_INVARIANT_ARGS["prompt_data"],
        "--input-key",
        SESSION_VERIFY_INVARIANT_ARGS["input_key"],
        "--num-rollout",
        str(SESSION_VERIFY_INVARIANT_ARGS["num_rollout"]),
        "--rollout-batch-size",
        str(SESSION_VERIFY_INVARIANT_ARGS["rollout_batch_size"]),
        "--rollout-max-response-len",
        str(SESSION_VERIFY_INVARIANT_ARGS["rollout_max_response_len"]),
        "--rollout-temperature",
        str(SESSION_VERIFY_INVARIANT_ARGS["rollout_temperature"]),
        "--global-batch-size",
        str(SESSION_VERIFY_INVARIANT_ARGS["global_batch_size"]),
        "--rm-type",
        SESSION_VERIFY_INVARIANT_ARGS["rm_type"],
        "--custom-generate-function-path",
        SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
        "--custom-agent-function-path",
        SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
        "--use-session-server",
        SESSION_VERIFY_INVARIANT_ARGS["use_session_server"],
        "--train-backend",
        SESSION_VERIFY_INVARIANT_ARGS["train_backend"],
        "--sglang-ep-size",
        str(SESSION_VERIFY_INVARIANT_ARGS["sglang_ep_size"]),
    ]
    for key in ("debug_rollout_only", "ci_test", "colocate"):
        if SESSION_VERIFY_INVARIANT_ARGS[key]:
            defaults.append("--" + key.replace("_", "-"))
    return [*defaults, *argv]


def _print_action_table(allowed_roles: list[str], *, cycles: int) -> None:
    schedule = select_schedule(allowed_roles, cycles=cycles)
    print("Driver schedule (after initial completion):")
    for i, action in enumerate(schedule, 1):
        print(f"  {i}. {action.value}")
    print()
    print("Required per-sample driver events (asserted in generate wrapper):")
    print("  - rollback         (deterministic; always required)")
    if "user" in allowed_roles:
        print("  - append_user      (deterministic; required because 'user' in roles)")
    if "system" in allowed_roles:
        print("  - append_system    (deterministic; required because 'system' in roles)")
    if "assistant" in allowed_roles:
        print("  - append_assistant (deterministic; required because 'assistant' in roles)")
    print()
    print("Required cross-sample driver events (asserted in generate wrapper):")
    print("  - append_tool      (model-dependent; >=1 sample must emit a tool_call)")
    print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sys.argv[1:] = _with_session_verify_defaults(sys.argv[1:])
    args = parse_args(add_custom_arguments=session_verify_extras)

    # Resolve the family-owned capability before any GPU work starts so an
    # unsupported verifier schedule fails immediately.
    allowed_roles = list(fixed_template_append_roles(args.tito_model))

    print(f"Model:                  {args.hf_checkpoint}")
    print(f"TITO model family:      {args.tito_model}")
    print(f"Template append roles:  {allowed_roles}")
    print(f"sglang reasoning parser:{args.sglang_reasoning_parser}")
    print(f"sglang tool-call parser:{args.sglang_tool_call_parser or '(none)'}")
    print(f"Rollout GPUs per engine:{args.rollout_num_gpus_per_engine}")
    print(f"sglang expert-parallel: {args.sglang_ep_size}")
    print(f"Actor nodes:            {args.actor_num_nodes}")
    print(f"Actor GPUs per node:    {args.actor_num_gpus_per_node}")
    print(f"Samples per prompt:     {args.n_samples_per_prompt}")
    print(f"Cycles per sample:      {args.session_verify_cycles}")
    print(f"Tool-call failure mode: {args.tool_call_failure_mode}")
    print()

    try:
        select_schedule(allowed_roles, cycles=args.session_verify_cycles)
    except ValueError as e:
        print(f"Verdict: FAIL -- {e}", file=sys.stderr)
        return 1

    _print_action_table(allowed_roles, cycles=args.session_verify_cycles)

    try:
        run_session_verify(args=args)
    except Exception as e:
        print()
        print(f"Verdict: FAIL -- {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print(
        "Verdict: PASS -- TITO incremental tokenization matched standard re-tokenize "
        "across all required driver actions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
