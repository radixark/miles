#!/usr/bin/env python3
"""CLI: run the multi-role TITO session-server verifier against a real model.

Boots the miles rollout pipeline (sglang + miles-router) under
``--debug-rollout-only`` and drives the schedule registered for the requested
``--tito-allowed-append-roles`` surface (see
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
        --tito-allowed-append-roles tool user system \\
        --sglang-reasoning-parser glm45 \\
        --sglang-tool-call-parser glm47 \\
        --rollout-num-gpus-per-engine 4

    # Qwen3-4B with tool + user surface, single-node, TP=1
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint Qwen/Qwen3-4B \\
        --tito-model qwen3 \\
        --tito-allowed-append-roles tool user \\
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
        --tito-allowed-append-roles tool user system \\
        --sglang-reasoning-parser glm45 \\
        --sglang-tool-call-parser glm47 \\
        --rollout-num-gpus-per-engine 4 \\
        --actor-num-nodes 2 --actor-num-gpus-per-node 8
"""

from __future__ import annotations

import logging
import sys

from miles.utils.arguments import parse_args
from miles.utils.test_utils.session_verify_agent import select_schedule
from miles.utils.test_utils.session_verify_runner import run_session_verify, session_verify_extras


def _print_action_table(allowed_roles: list[str]) -> None:
    schedule = select_schedule(allowed_roles)
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
    print()
    print("Required cross-sample driver events (asserted in generate wrapper):")
    print("  - append_tool      (model-dependent; >=1 sample must emit a tool_call)")
    print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args(add_custom_arguments=session_verify_extras)

    # Normalize role surface up-front for the printed summary.  ``run_session_verify``
    # also normalizes internally (lowercase + dedup + ensure 'tool'), but doing it
    # here lets ``select_schedule`` validate the surface before any GPU work starts.
    allowed_roles = sorted(set(r.lower() for r in args.tito_allowed_append_roles) | {"tool"})

    print(f"Model:                  {args.hf_checkpoint}")
    print(f"TITO model family:      {args.tito_model}")
    print(f"Allowed append roles:   {allowed_roles}")
    print(f"sglang reasoning parser:{args.sglang_reasoning_parser}")
    print(f"sglang tool-call parser:{args.sglang_tool_call_parser or '(none)'}")
    print(f"Rollout GPUs per engine:{args.rollout_num_gpus_per_engine}")
    print(f"sglang expert-parallel: {args.sglang_expert_parallel_size}")
    print(f"Actor nodes:            {args.actor_num_nodes}")
    print(f"Actor GPUs per node:    {args.actor_num_gpus_per_node}")
    print(f"Samples per prompt:     {args.n_samples_per_prompt}")
    print(f"Cycles per sample:      {args.session_verify_cycles}")
    print(f"Tool-call failure mode: {args.tool_call_failure_mode}")
    print()

    try:
        select_schedule(allowed_roles)
    except ValueError as e:
        print(f"Verdict: FAIL -- {e}", file=sys.stderr)
        return 1

    _print_action_table(allowed_roles)

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
