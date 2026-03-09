#!/usr/bin/env python3
"""One-click verification: is a chat template append-only after last user message?

Usage examples::

    # Verify a local .jinja template file
    python scripts/tools/verify_chat_template.py --template path/to/template.jinja

    # Verify a HuggingFace model's chat template
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B

    # Verify with autofix (use our fixed template if available)
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B --autofix

    # Also run thinking-specific cases
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --thinking
"""

from __future__ import annotations

import argparse
import sys


def _load_template_from_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load_template_from_model(model_id: str, *, autofix: bool) -> tuple[str, str]:
    """Load chat template for a HF model. Returns (template_str, source_description)."""
    if autofix:
        from miles.utils.chat_template_utils.autofix import try_get_fixed_chat_template

        fixed_path = try_get_fixed_chat_template(model_id)
        if fixed_path:
            return _load_template_from_file(fixed_path), f"fixed template: {fixed_path}"

    from miles.utils.chat_template_utils.template import load_hf_chat_template

    return load_hf_chat_template(model_id), f"HuggingFace: {model_id}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a chat template is append-only after last user message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--template",
        metavar="PATH",
        help="Path to a local .jinja chat template file.",
    )
    source.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID (e.g. Qwen/Qwen3-0.6B).",
    )

    parser.add_argument(
        "--autofix",
        action="store_true",
        help="When using --model, apply our fixed template if one exists.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Also run thinking-specific cases (enable_thinking=True/False).",
    )

    args = parser.parse_args()

    # ── Load template ──────────────────────────────────────────────────
    if args.template:
        chat_template = _load_template_from_file(args.template)
        source_desc = f"file: {args.template}"
    else:
        chat_template, source_desc = _load_template_from_model(args.model, autofix=args.autofix)

    print(f"Template source: {source_desc}")
    print(f"Thinking cases:  {'enabled' if args.thinking else 'disabled'}")
    print()

    # ── Run verification ───────────────────────────────────────────────
    from miles.utils.test_utils.chat_template_verify import run_all_checks

    results = run_all_checks(chat_template, include_thinking=args.thinking)

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
        print("\nVerdict: FAIL - template is NOT append-only after last user message")
        return 1
    else:
        print("\nVerdict: PASS - template IS append-only after last user message")
        return 0


if __name__ == "__main__":
    sys.exit(main())
