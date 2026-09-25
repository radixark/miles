"""Every miles/ or miles_plugins/ path a doc page names must exist in the repo.

Source files move and get deleted in refactors, and the docs that point at them are not
part of the diff, so the references rot silently: #2162 deleted
miles/ray/train/actor_factory.py and two pages kept sending readers there. This test
scans docs/ for package source paths, written bare or as github.com/radixark/miles
blob/tree links, and fails on any that no longer resolve.

Three kinds of page are out of scope:
- docs/diffusion/ documents the standalone radixark/miles_diffusion repo, whose paths
  do not live in this tree.
- A page that opens with the "> **Outdated.**" banner is already flagged to readers and
  is waiting for a rewrite.
- A model recipe whose implementation sits on an open PR branch is listed in
  _PAGES_AHEAD_OF_MAIN; an entry must be dropped once its code lands, which
  test_pages_ahead_of_main_still_reference_unmerged_code enforces.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs"

_OUT_OF_SCOPE_DIRS = ("diffusion",)
_OUTDATED_BANNER = "> **Outdated.**"

# doc page (relative to docs/) -> the open PR its source paths come from
_PAGES_AHEAD_OF_MAIN = {
    "models/deepseek/deepseek-v4-1-flash.md": "radixark/miles:deepseek-v41 image branch",
    "models/kimi/kimi-k3.md": "radixark/miles#1825",
    "models/qwen/qwen3-8-flash-next.md": "radixark/miles#2777",
}

# A bare path must not be glued to a longer path or word on its left, so "foo/miles/x.py"
# and "sglang_miles/x.py" are not matches. The last character may not be "." so a path
# that ends a sentence drops the full stop.
_BARE_PATH = re.compile(r"(?<![\w./-])((?:miles|miles_plugins)/[\w./-]*[\w/])")
_GITHUB_PATH = re.compile(r"github\.com/radixark/miles/(?:blob|tree)/main/((?:miles|miles_plugins)/[\w./-]*[\w/])")


def extract_source_paths(line: str) -> list[str]:
    return _BARE_PATH.findall(line) + _GITHUB_PATH.findall(line)


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + len("\n---\n") :]
    return text


def is_outdated_page(text: str) -> bool:
    return _strip_front_matter(text).lstrip().startswith(_OUTDATED_BANNER)


def find_missing_paths(doc: Path) -> list[str]:
    missing = []
    for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
        for path in extract_source_paths(line):
            if not (REPO_ROOT / path).exists():
                missing.append(f"docs/{doc.relative_to(DOCS_ROOT)}:{lineno}: {path}")
    return missing


def _pages_in_scope() -> list[Path]:
    pages = []
    for doc in sorted(DOCS_ROOT.rglob("*.md*")):
        rel = doc.relative_to(DOCS_ROOT)
        if rel.parts[0] in _OUT_OF_SCOPE_DIRS:
            continue
        if is_outdated_page(doc.read_text(encoding="utf-8")):
            continue
        pages.append(doc)
    return pages


class TestExtractSourcePaths:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ("see `miles/ray/specs/train.py` for the env", ["miles/ray/specs/train.py"]),
            ("lives in miles/ray/specs/train.py.", ["miles/ray/specs/train.py"]),
            ("per-arch specs under `miles_plugins/models/`", ["miles_plugins/models/"]),
            ("at miles/utils/arguments.py:1639", ["miles/utils/arguments.py"]),
            (
                "[x](https://github.com/radixark/miles/blob/main/miles/router/router.py)",
                ["miles/router/router.py"],
            ),
            ("the sgl-project/sglang/miles/x.py mirror", []),
            ("import from sglang_miles/utils.py", []),
            ("the miles.backends.fsdp_utils module path", []),
        ],
    )
    def test_extracts_repo_relative_paths(self, line: str, expected: list[str]) -> None:
        assert extract_source_paths(line) == expected

    def test_outdated_banner_is_detected_after_front_matter(self) -> None:
        page = "---\ntitle: X\n---\n> **Outdated.** This page describes the old stack.\n"

        assert is_outdated_page(page)
        assert not is_outdated_page("---\ntitle: X\n---\nBody that mentions > **Outdated.** later.\n")


class TestDocSourcePaths:
    def test_docs_reference_existing_source_paths(self) -> None:
        missing = [
            ref
            for doc in _pages_in_scope()
            if str(doc.relative_to(DOCS_ROOT)) not in _PAGES_AHEAD_OF_MAIN
            for ref in find_missing_paths(doc)
        ]

        assert not missing, (
            "these doc pages point at source paths that do not exist; update the reference to "
            "where the code lives now:\n" + "\n".join(missing)
        )

    @pytest.mark.parametrize("page", sorted(_PAGES_AHEAD_OF_MAIN))
    def test_pages_ahead_of_main_still_reference_unmerged_code(self, page: str) -> None:
        doc = DOCS_ROOT / page

        assert doc.exists(), f"{page} is in _PAGES_AHEAD_OF_MAIN but no longer exists; drop the entry"
        assert find_missing_paths(doc), (
            f"every source path in {page} now exists ({_PAGES_AHEAD_OF_MAIN[page]} landed); "
            "drop the page from _PAGES_AHEAD_OF_MAIN so it is checked like the rest"
        )
