#!/usr/bin/env python3
"""
Sync and validate Miles server argument docs against `miles/utils/arguments.py`.

Managed scope:
- Only flags explicitly defined in `miles/utils/arguments.py` via
  `parser.add_argument("--...")` or `reset_arg(parser, "--...", ...)`.
- A small explicit exclusion list is applied for intentionally undocumented flags.

Modes:
- `--write`: update managed docs descriptions from code help strings.
- `--check`: validation-only; fail when managed docs and code are out of sync.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOC_PATH = Path("docs/en/advanced/miles_server_args.md")
DEFAULT_ARGS_PATH = Path("miles/utils/arguments.py")

# Explicitly defined in arguments.py but intentionally not documented in the table.
EXCLUDED_FLAGS = {
    "--offload",
    "--offload-train",
    "--offload-rollout",
    "--miles-router-enable-token-input-for-chat-completions",
}


@dataclass(frozen=True)
class DocRow:
    line_index: int
    flag: str
    description: str
    cells: tuple[str, str, str, str, str]
    line_ending: str


@dataclass(frozen=True)
class ArgDef:
    flag: str
    help_text: str | None
    lineno: int
    kind: str


@dataclass(frozen=True)
class ValidationResult:
    missing_in_docs: list[str]
    missing_help_in_code: list[str]
    stale_rows: list[DocRow]
    duplicate_doc_flags: list[str]
    duplicate_arg_flags: list[str]
    managed_doc_rows: list[DocRow]

    @property
    def has_issues(self) -> bool:
        return bool(
            self.missing_in_docs
            or self.missing_help_in_code
            or self.stale_rows
            or self.duplicate_doc_flags
            or self.duplicate_arg_flags
        )


def _literal_str(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(value, str):
        return value
    return None


def _split_markdown_row(line: str) -> list[str] | None:
    """
    Split a markdown table row while honoring escaped pipes (\\|) and inline code.
    """
    if not line.startswith("|"):
        return None

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    in_code = False

    for ch in line:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            continue
        if ch == "`":
            current.append(ch)
            in_code = not in_code
            continue
        if ch == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(ch)

    cells.append("".join(current).strip())

    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _extract_doc_rows(doc_path: Path) -> tuple[list[str], list[DocRow], list[str]]:
    lines = doc_path.read_text().splitlines(keepends=True)
    rows: list[DocRow] = []
    seen: set[str] = set()
    duplicate_flags: list[str] = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        line_ending = "\n" if raw_line.endswith("\n") else ""
        cells = _split_markdown_row(line)
        if cells is None or len(cells) != 5:
            continue

        match = re.fullmatch(r"`(--[^`]+)`", cells[0])
        if not match:
            continue
        flag = match.group(1)
        if flag in seen and flag not in duplicate_flags:
            duplicate_flags.append(flag)
        seen.add(flag)

        rows.append(
            DocRow(
                line_index=idx,
                flag=flag,
                description=cells[1],
                cells=(cells[0], cells[1], cells[2], cells[3], cells[4]),
                line_ending=line_ending,
            )
        )

    duplicate_flags.sort()
    return lines, rows, duplicate_flags


def _extract_arg_defs(args_path: Path) -> tuple[dict[str, ArgDef], list[str]]:
    src = args_path.read_text()
    module = ast.parse(src)
    calls: list[ast.Call] = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))

    arg_defs: dict[str, ArgDef] = {}
    duplicate_flags: list[str] = []

    for call in calls:
        kind: str | None = None
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "parser"
        ):
            kind = "add_argument"
        elif isinstance(call.func, ast.Name) and call.func.id == "reset_arg":
            kind = "reset_arg"
        if kind is None:
            continue

        flag: str | None = None
        if kind == "add_argument" and call.args:
            value = _literal_str(call.args[0])
            if value is not None and value.startswith("--"):
                flag = value
        elif kind == "reset_arg" and len(call.args) >= 2:
            if not (isinstance(call.args[0], ast.Name) and call.args[0].id == "parser"):
                continue
            value = _literal_str(call.args[1])
            if value is not None and value.startswith("--"):
                flag = value
        if flag is None:
            continue

        help_text: str | None = None
        for kw in call.keywords:
            if kw.arg == "help":
                help_text = _literal_str(kw.value)
                break

        if flag in arg_defs and flag not in duplicate_flags:
            duplicate_flags.append(flag)
        arg_defs[flag] = ArgDef(flag=flag, help_text=help_text, lineno=call.lineno, kind=kind)

    duplicate_flags.sort()
    return arg_defs, duplicate_flags


def _escape_unescaped_pipes(text: str) -> str:
    out: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "|":
            out.append("\\|")
        else:
            out.append(ch)
    return "".join(out)


def _render_row_with_description(row: DocRow, description: str) -> str:
    desc = _escape_unescaped_pipes(description)
    c0, _, c2, c3, c4 = row.cells
    return f"| {c0} | {desc} | {c2} | {c3} | {c4} |{row.line_ending}"


def _validate(
    doc_rows: list[DocRow], arg_defs: dict[str, ArgDef], duplicate_doc_flags: list[str], duplicate_arg_flags: list[str]
) -> ValidationResult:
    managed_flags = sorted(set(arg_defs) - EXCLUDED_FLAGS)
    doc_by_flag = {row.flag: row for row in doc_rows}

    missing_in_docs = sorted(flag for flag in managed_flags if flag not in doc_by_flag)
    missing_help_in_code = sorted(
        flag for flag in managed_flags if flag in doc_by_flag and arg_defs[flag].help_text is None
    )

    managed_doc_rows = [doc_by_flag[flag] for flag in managed_flags if flag in doc_by_flag]
    stale_rows: list[DocRow] = []
    for row in managed_doc_rows:
        help_text = arg_defs[row.flag].help_text
        if help_text is None:
            continue
        if row.description != help_text:
            stale_rows.append(row)

    stale_rows.sort(key=lambda row: row.flag)
    return ValidationResult(
        missing_in_docs=missing_in_docs,
        missing_help_in_code=missing_help_in_code,
        stale_rows=stale_rows,
        duplicate_doc_flags=duplicate_doc_flags,
        duplicate_arg_flags=duplicate_arg_flags,
        managed_doc_rows=managed_doc_rows,
    )


def _format_flag_list(flags: list[str], limit: int = 25) -> str:
    preview = flags[:limit]
    text = ", ".join(preview)
    if len(flags) > limit:
        text += f", ... (+{len(flags) - limit} more)"
    return text


def _print_report(result: ValidationResult, mode: str) -> None:
    print(f"[sync-param-docs] mode={mode}")
    print(f"[sync-param-docs] managed rows in docs: {len(result.managed_doc_rows)}")
    print(f"[sync-param-docs] stale description rows: {len(result.stale_rows)}")
    print(f"[sync-param-docs] missing managed flags in docs: {len(result.missing_in_docs)}")
    print(f"[sync-param-docs] missing help in code: {len(result.missing_help_in_code)}")
    print(f"[sync-param-docs] duplicate doc flags: {len(result.duplicate_doc_flags)}")
    print(f"[sync-param-docs] duplicate code flags: {len(result.duplicate_arg_flags)}")

    if result.duplicate_doc_flags:
        print(f"[sync-param-docs] duplicate doc flags: {_format_flag_list(result.duplicate_doc_flags)}")
    if result.duplicate_arg_flags:
        print(f"[sync-param-docs] duplicate code flags: {_format_flag_list(result.duplicate_arg_flags)}")
    if result.missing_in_docs:
        print(f"[sync-param-docs] missing in docs: {_format_flag_list(result.missing_in_docs)}")
    if result.missing_help_in_code:
        print(f"[sync-param-docs] missing help in code: {_format_flag_list(result.missing_help_in_code)}")
    if result.stale_rows:
        stale_flags = [row.flag for row in result.stale_rows]
        print(f"[sync-param-docs] stale rows: {_format_flag_list(stale_flags)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync and validate Miles server argument docs.")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC_PATH, help="Path to miles_server_args markdown file.")
    parser.add_argument(
        "--args-path",
        type=Path,
        default=DEFAULT_ARGS_PATH,
        help="Path to arguments.py source-of-truth file.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Sync managed docs descriptions from code.")
    mode.add_argument("--check", action="store_true", help="Validation-only mode; fail when drift is detected.")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()

    if not cli.doc.exists():
        print(f"[sync-param-docs] docs file not found: {cli.doc}", file=sys.stderr)
        return 2
    if not cli.args_path.exists():
        print(f"[sync-param-docs] args file not found: {cli.args_path}", file=sys.stderr)
        return 2

    lines, doc_rows, duplicate_doc_flags = _extract_doc_rows(cli.doc)
    arg_defs, duplicate_arg_flags = _extract_arg_defs(cli.args_path)
    result = _validate(doc_rows, arg_defs, duplicate_doc_flags, duplicate_arg_flags)

    mode = "write" if cli.write else "check"

    _print_report(result, mode)

    if mode == "check":
        if result.has_issues:
            print("[sync-param-docs] failed: managed docs and args are out of sync.")
            return 1
        print("[sync-param-docs] pass: managed docs and args are in sync.")
        return 0

    # write mode: only write when we can safely sync all managed entries.
    blocking_issues = (
        result.duplicate_doc_flags
        or result.duplicate_arg_flags
        or result.missing_in_docs
        or result.missing_help_in_code
    )
    if blocking_issues:
        print("[sync-param-docs] failed: fix blocking issues before write mode can sync.", file=sys.stderr)
        return 1

    if not result.stale_rows:
        print("[sync-param-docs] no changes needed.")
        return 0

    stale_by_line = {row.line_index: row for row in result.stale_rows}
    for idx, row in stale_by_line.items():
        help_text = arg_defs[row.flag].help_text
        assert help_text is not None
        lines[idx] = _render_row_with_description(row, help_text)

    cli.doc.write_text("".join(lines))
    print(f"[sync-param-docs] updated {len(result.stale_rows)} row(s) in {cli.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
