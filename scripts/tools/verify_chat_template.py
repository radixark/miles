#!/usr/bin/env python3
"""One-click verification: is a chat template append-only after last user message?

Usage examples::

    # Verify a local .jinja template file at raw template-string level
    python scripts/tools/verify_chat_template.py --template path/to/template.jinja

    # Verify a HuggingFace model's chat template at raw template-string level
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B

    # Verify through the registered TITO tokenizer family
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \\
        --tito-model qwen3 --tito-allowed-append-roles tool

    # Verify through TITO while testing a local template override
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \\
        --tito-model qwen3 --template miles/utils/chat_template_utils/templates/qwen3_fixed.jinja

    # Restrict which append roles the session is allowed to use (tool is implicit)
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \\
        --tito-allowed-append-roles user

    # Run thinking cases: off (default) / on / both
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --thinking both
"""

from __future__ import annotations

import argparse
import json
import sys

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType


def _load_template_from_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load_template_from_model(
    model_id: str,
) -> tuple[str, str]:
    """Load the raw HuggingFace chat template for a model."""
    from miles.utils.chat_template_utils.template import load_hf_chat_template

    return load_hf_chat_template(model_id), f"HuggingFace: {model_id}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a chat template is append-only after last user message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--template",
        metavar="PATH",
        help=(
            "Path to a local .jinja chat template file. Without --tito-model this "
            "runs the legacy raw template-string verifier; with --tito-model this "
            "overrides the registered TITO template while still using --model for tokenizer loading."
        ),
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID or local checkpoint path (e.g. Qwen/Qwen3-0.6B).",
    )

    parser.add_argument(
        "--tito-model",
        choices=[t.value for t in TITOTokenizerType],
        default=None,
        help=(
            "Verify through the registered TITO tokenizer family under the given "
            "--tito-allowed-append-roles surface. Requires --model and exercises "
            "the production-shape TITO merge/tokenize path instead of the raw "
            "template-string verifier."
        ),
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        default=["tool"],
        choices=["tool", "user", "system"],
        metavar="ROLE",
        help=(
            "Roles the session may append after an assistant turn.  'tool' is "
            "implicitly always allowed (listing it is fine).  Trajectories that "
            "require roles outside this set are skipped.  Default: tool."
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "on", "both"],
        default="on",
        help=(
            "Thinking-mode filter.  off: non-thinking trajectories only.  "
            "on: thinking trajectories with enable_thinking=True.  "
            "both: non-thinking + thinking with enable_thinking={True,False}.  "
            "Default: on."
        ),
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        metavar="JSON",
        help=(
            "Extra kwargs threaded into the chat template on every render, as a "
            "JSON object (e.g. '{\"clear_thinking\": false}' for GLM).  Same "
            "convention as --apply-chat-template-kwargs in the training entrypoint."
        ),
    )

    args = parser.parse_args()

    if args.model is None and args.template is None:
        parser.error("one of --model or --template is required")
    if args.tito_model is not None and args.model is None:
        parser.error("--tito-model requires --model so the TITO verifier can load the tokenizer")

    extra_template_kwargs = dict(args.chat_template_kwargs or {})

    # ``--tito-allowed-append-roles`` lists the *optional* extra roles; ``tool``
    # is implicit for tool-capable agentic workflows and unioned in here so both
    # the fixed-template lookup and the trajectory filter see the same surface.
    allowed_roles = set(args.tito_allowed_append_roles) | {"tool"}

    use_tito_instance = args.tito_model is not None

    # ── Load template/tokenizer ────────────────────────────────────────
    if use_tito_instance:
        from miles.utils.chat_template_utils import resolve_fixed_chat_template
        from miles.utils.processing_utils import load_tokenizer

        fixed_path, resolved_kwargs = resolve_fixed_chat_template(args.tito_model, sorted(allowed_roles))
        for key, value in resolved_kwargs.items():
            if key in extra_template_kwargs:
                continue
            extra_template_kwargs[key] = value
            print(f"Auto-set --chat-template-kwargs {key}={value!r} (from --tito-model={args.tito_model})")

        template_path = args.template or fixed_path
        tokenizer = load_tokenizer(args.model, chat_template_path=template_path, trust_remote_code=True)
        if args.template:
            source_desc = f"template override via TITO: {args.template}"
        elif fixed_path:
            source_desc = f"fixed template via TITO: {fixed_path}"
        elif getattr(tokenizer, "chat_template", None) is not None:
            source_desc = f"HuggingFace via TITO: {args.model}"
        else:
            source_desc = f"TITO encoder: {args.tito_model}"
    elif args.template:
        chat_template = _load_template_from_file(args.template)
        source_desc = f"file: {args.template}"
    else:
        chat_template, source_desc = _load_template_from_model(args.model)

    from miles.utils.test_utils.chat_template_verify import (
        ALL_CASES,
        check_coverage,
        run_all_checks,
        run_all_checks_via_tito,
        select_cases,
    )

    is_thinking_filter = {"off": False, "on": True, "both": None}[args.thinking]
    selected = select_cases(allowed_append_roles=allowed_roles, is_thinking=is_thinking_filter)

    print(f"Template source:       {source_desc}")
    print(f"Allowed append roles:  {sorted(allowed_roles)}")
    print(f"Thinking mode:         {args.thinking}")
    if extra_template_kwargs:
        print(f"Template kwargs:       {extra_template_kwargs}")
    print(f"Selected trajectories: {len(selected)} of {len(ALL_CASES)} (after filtering)")
    print()

    # Global coverage sanity check — reports gaps in the mock-trajectory pool,
    # not gaps caused by the current CLI flags.  A gap here means some CLI
    # setting exercises no trajectory; fixing it requires adding a trajectory
    # in mock_trajectories.py.
    coverage = check_coverage()
    if coverage.missing:
        print("Trajectory coverage gaps ((thinking, append_roles \\ {tool}) with no trajectory):")
        for is_thinking, roles in coverage.missing:
            label = "thinking    " if is_thinking else "non-thinking"
            roles_str = "{" + ", ".join(roles) + "}" if roles else "{}"
            print(f"  - {label}  x  {roles_str}")
        print()

    # ── Run verification ───────────────────────────────────────────────
    if use_tito_instance:
        results = run_all_checks_via_tito(
            tokenizer,
            args.tito_model,
            allowed_append_roles=allowed_roles,
            thinking=args.thinking,
            extra_template_kwargs=extra_template_kwargs,
        )
    else:
        results = run_all_checks(
            chat_template,
            allowed_append_roles=allowed_roles,
            thinking=args.thinking,
            extra_template_kwargs=extra_template_kwargs,
        )

    # ── Print results ──────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    max_name_len = max((len(r.case_name) for r in results), default=0)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = f"  [{status}] {r.case_name:<{max_name_len}}"
        if r.error:
            first_line = r.error.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            line += f"  -- {first_line}"
        print(line)

    print()
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")

    if failed:
        if use_tito_instance:
            print("\nVerdict: FAIL - TITO incremental tokenization did NOT match standard render")
        else:
            print("\nVerdict: FAIL - template is NOT append-only after last user message")
        return 1
    else:
        if use_tito_instance:
            print("\nVerdict: PASS - TITO incremental tokenization matched standard render")
        else:
            print("\nVerdict: PASS - template IS append-only after last user message")
        return 0


if __name__ == "__main__":
    sys.exit(main())
