"""Keep docs/user-guide/cli-reference.md in step with the flags arguments.py defines.

Nothing else does, so a flag lands undocumented and stays invisible until someone
greps the source (#543). ``cli_reference_undocumented.txt`` holds the flags that
were already undocumented when this landed; the per-test docstrings below say how
the two files move it toward empty.

Two flag classes are out of scope and excluded: Megatron's inherited flags, which
the doc covers by reference, and the ``--sglang-*`` / ``--eval-sglang-*``
passthrough families, which the doc covers as families rather than per-flag.

Reads source and Markdown only -- never imports ``miles`` -- so it needs none of
the training dependencies.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "user-guide" / "cli-reference.md"
BASELINE_PATH = Path(__file__).with_name("cli_reference_undocumented.txt")
_TEST_REL_PATH = Path(__file__).resolve().relative_to(REPO_ROOT)
_REGEN_CMD = f"python {_TEST_REL_PATH} --regen"

# Modules whose ``add_argument()`` calls feed ``get_miles_extra_args_provider()``,
# i.e. the ``train.py`` / ``train_async.py`` CLI. A new module that defines
# user-facing training flags belongs in this tuple.
_ARG_SOURCES = (
    "miles/utils/arguments.py",
    "miles/dashboard/args.py",
)

_PASSTHROUGH_PREFIXES = ("--sglang-", "--eval-sglang-")

# A parsing regression that finds almost nothing must not pass silently.
_MIN_EXPECTED_FLAGS = 300


def _defined_flags() -> set[str]:
    """Long-option strings passed as the first arg to ``parser.add_argument(...)``.

    ``reset_arg(parser, "--x", ...)`` calls are deliberately not matched: those
    re-default Megatron-owned flags, which are out of scope.
    """
    flags: set[str] = set()
    for rel in _ARG_SOURCES:
        source = (REPO_ROOT / rel).read_text()
        tree = ast.parse(source, rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
            ):
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("--"):
                    flags.add(first.value)
    return flags


def _documented_flags() -> set[str]:
    """Every ``--flag`` token anywhere in cli-reference.md.

    This scans the whole file, not just the reference table, so a flag named only
    in prose -- no table row -- still counts as documented. Deliberately loose:
    this is a backlog guard, not a table-completeness check.
    """
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", DOC_PATH.read_text()))


def _counterpart(flag: str) -> str:
    """``--foo`` <-> ``--no-foo``; either polarity in the doc counts as covered."""
    if flag.startswith("--no-"):
        return "--" + flag[len("--no-") :]
    return "--no-" + flag[len("--") :]


def _is_covered(flag: str, documented: set[str]) -> bool:
    return flag in documented or _counterpart(flag) in documented


def _undocumented(defined: set[str], documented: set[str]) -> set[str]:
    return {
        flag for flag in defined if not flag.startswith(_PASSTHROUGH_PREFIXES) and not _is_covered(flag, documented)
    }


def _baseline_entries() -> list[str]:
    lines = BASELINE_PATH.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _require_checkout() -> None:
    if not DOC_PATH.is_file() or not (REPO_ROOT / _ARG_SOURCES[0]).is_file():
        import pytest

        pytest.skip("cli-reference.md / arguments.py not on disk; nothing to compare")


_BASELINE_HEADER = (
    "# Miles training-CLI flags not yet in docs/user-guide/cli-reference.md.\n"
    f"# Baseline for {_TEST_REL_PATH} -- built to shrink.\n"
    "# Document a flag in cli-reference.md, then delete its line here.\n"
    "# Regenerate after a flag rename on main re-reds the branch:\n"
    f"#   {_REGEN_CMD}\n"
)


def _regenerate_baseline() -> None:
    """Rewrite the baseline file from the current source + doc.

    Use when a rename or new flag on main makes the branch fail before merge:
    it re-snapshots today's undocumented set instead of hand-patching lines.

    Refuses to write when the ``add_argument`` scan finds implausibly few flags,
    the same regression guard ``test_no_new_undocumented_flag`` applies: a broken
    scan must not overwrite the backlog with an empty set.
    """
    defined = _defined_flags()
    if len(defined) < _MIN_EXPECTED_FLAGS:
        raise SystemExit(
            f"only {len(defined)} flags parsed from {list(_ARG_SOURCES)} (expected "
            f">= {_MIN_EXPECTED_FLAGS}); refusing to overwrite {BASELINE_PATH.name}. "
            "The add_argument scan probably broke -- fix it before regenerating."
        )
    undocumented = sorted(_undocumented(defined, _documented_flags()))
    text = (
        _BASELINE_HEADER
        + f"# {len(undocumented)} flags ({len(defined)} scanned) as of generation.\n"
        + "".join(f"{flag}\n" for flag in undocumented)
    )
    BASELINE_PATH.write_text(text)


def test_no_new_undocumented_flag() -> None:
    """The undocumented set may not grow: a new flag gets a doc row, or a
    deliberate line in cli_reference_undocumented.txt."""
    _require_checkout()
    defined = _defined_flags()
    assert len(defined) >= _MIN_EXPECTED_FLAGS, (
        f"only {len(defined)} flags parsed from {list(_ARG_SOURCES)}; the add_argument "
        "scan probably broke rather than the CLI shrinking that far"
    )
    new = sorted(_undocumented(defined, _documented_flags()) - set(_baseline_entries()))
    assert not new, (
        "these flags are defined for train.py / train_async.py but missing from "
        "docs/user-guide/cli-reference.md. Add a row to the Complete reference "
        f"section, or -- if the flag is genuinely internal -- append it to "
        f"{BASELINE_PATH.name}. If main renamed a flag, re-snapshot with "
        f"`{_REGEN_CMD}`:\n  " + "\n  ".join(new)
    )


def test_baseline_has_no_stale_entry() -> None:
    """Every backlog line still names a real, still-undocumented flag, so a line
    whose flag was renamed, removed, or documented must be deleted."""
    _require_checkout()
    defined = _defined_flags()
    documented = _documented_flags()
    gone = sorted(f for f in _baseline_entries() if f not in defined)
    now_documented = sorted(f for f in _baseline_entries() if f in defined and _is_covered(f, documented))
    stale = gone + now_documented
    assert not stale, (
        f"{BASELINE_PATH.name} lines that no longer name an undocumented flag -- "
        "delete them (or re-snapshot with "
        f"`{_REGEN_CMD}` if main renamed a flag).\n"
        f"  renamed or removed: {gone or '[]'}\n"
        f"  now documented (thank you): {now_documented or '[]'}"
    )


def test_baseline_is_sorted_and_unique() -> None:
    """The backlog file stays sorted and duplicate-free, so each add/remove is a
    one-line diff."""
    _require_checkout()
    entries = _baseline_entries()
    assert entries == sorted(entries), "keep cli_reference_undocumented.txt sorted so diffs stay small"
    assert len(entries) == len(set(entries)), "duplicate line in cli_reference_undocumented.txt"


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--regen"]:
        raise SystemExit(f"usage: {_REGEN_CMD}")
    _regenerate_baseline()
    print(f"regenerated {BASELINE_PATH.relative_to(REPO_ROOT)}")
