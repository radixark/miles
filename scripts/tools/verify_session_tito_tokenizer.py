#!/usr/bin/env python3
"""CLI: run the multi-role TITO session-server verifier against a real model.

Boots the miles rollout pipeline (sglang + miles-router) under
``--debug-rollout-only`` and drives the schedule registered for the requested
``--tito-allowed-append-roles`` surface (see
``miles.utils.test_utils.session_verify_agent``).  PASS iff every sample
completes without HTTP error from the server-side prefix check and the
custom-generate coverage assertion is satisfied.

Usage examples::

    # GLM-4.7-Flash with tool + user + system surface
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint zai-org/GLM-4.7-Flash \\
        --tito-model glm47 \\
        --tito-allowed-append-roles tool user system \\
        --reasoning-parser glm45 \\
        --tool-call-parser glm47

    # Qwen3-4B with tool + user surface
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint Qwen/Qwen3-4B \\
        --tito-model qwen3 \\
        --tito-allowed-append-roles tool user \\
        --reasoning-parser qwen3 \\
        --tool-call-parser qwen25
"""

from __future__ import annotations

import argparse
import logging
import sys

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType
from miles.utils.test_utils.session_verify_agent import select_schedule
from miles.utils.test_utils.session_verify_runner import run_session_verify


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
    parser = argparse.ArgumentParser(
        description="Verify a model's TITO tokenizer under multi-role session-server "
        "driver against real model inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-checkpoint",
        required=True,
        help="HuggingFace model ID, e.g. zai-org/GLM-4.7-Flash.",
    )
    parser.add_argument(
        "--tito-model",
        required=True,
        choices=[t.value for t in TITOTokenizerType],
        help="TITO tokenizer family (e.g. qwen3, glm47).",
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        required=True,
        choices=["tool", "user", "system"],
        help=(
            "Role surface to verify.  Must match a registered schedule in "
            "session_verify_agent._SUPPORTED_ROLE_SURFACES.  'tool' is implicitly "
            "added if omitted."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        required=True,
        help="--sglang-reasoning-parser value (e.g. qwen3, glm45).",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=None,
        help="--sglang-tool-call-parser value (e.g. qwen25, glm47).  Optional.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="sglang engine tensor-parallel size "
        "(``--rollout-num-gpus-per-engine``).  Pick the smallest TP that fits "
        "the model — small models can run TP=1 even on a full 8-GPU node.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Actor / ray-cluster allocation per node "
        "(``--actor-num-gpus-per-node``).  Defaults to 8 (full node) — leave "
        "alone unless you know what you're doing; the engine TP is a separate "
        "knob (``--tp-size``).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="--n-samples-per-prompt; total samples generated equals "
        "rollout-batch-size (16, fixed) * n-samples-per-prompt.  Coverage "
        "assertion is per-sample for deterministic actions and cross-sample "
        "for tool_call.  Group-of-N is needed because miles' rollout loop "
        "drops the whole group on any TRUNCATED sample, then refills — "
        "small N with one TRUNCATED-prone sample cycles forever.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Driver schedule cycles per sample (default 3).  Drop to 2 for "
        "tighter-context models (e.g. Qwen3 32K with 4K response budget).",
    )
    parser.add_argument(
        "--assistant-text-threshold",
        type=float,
        default=0.05,
        help="Soft threshold for assistant_text mismatch ratio.  Default 0.05.  "
        "Raise to 1.0 for families whose upstream sglang reasoning parser "
        "is known to roundtrip imperfectly (e.g. nemotron_3 keeps a trailing "
        "newline in reasoning_content) — hard mismatches still gate.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Normalize role surface: lowercase, dedup, ensure 'tool' is in.  Same convention
    # as miles/utils/arguments.py:1828 so the CLI matches train pipeline behavior.
    allowed_roles = sorted(set(r.lower() for r in args.tito_allowed_append_roles) | {"tool"})

    print(f"Model:                 {args.hf_checkpoint}")
    print(f"TITO model family:     {args.tito_model}")
    print(f"Allowed append roles:  {allowed_roles}")
    print(f"Reasoning parser:      {args.reasoning_parser}")
    print(f"Tool call parser:      {args.tool_call_parser or '(none)'}")
    print(f"Engine TP size:        {args.tp_size}")
    print(f"Actor GPUs per node:   {args.num_gpus}")
    print(f"Samples per prompt:    {args.n_samples}")
    print(f"Cycles per sample:     {args.cycles}")
    print()

    try:
        select_schedule(allowed_roles)
    except ValueError as e:
        print(f"Verdict: FAIL -- {e}", file=sys.stderr)
        return 1

    _print_action_table(allowed_roles)

    try:
        run_session_verify(
            hf_checkpoint=args.hf_checkpoint,
            tito_model=args.tito_model,
            allowed_append_roles=allowed_roles,
            reasoning_parser=args.reasoning_parser,
            tool_call_parser=args.tool_call_parser,
            tp_size=args.tp_size,
            num_gpus=args.num_gpus,
            n_samples_per_prompt=args.n_samples,
            cycles=args.cycles,
            assistant_text_threshold=args.assistant_text_threshold,
        )
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
