"""List the CI test files a branch's changes can reach, with their registrations.

Run from the repo root: PYTHONPATH=. python3 .claude/skills/pr-ci-targeted/affected_tests.py [--base origin/main]

A changed Python module is referenced by its dotted path (and by the package path when the package
__init__ re-exports it), a model definition under scripts/models/ by its name as a string literal (launchers
load it by --model-name), and any other file by its path. References are followed through source files up to
--depth hops, and every registered test file found along the way is reported. Files referenced by more than
--hub-limit files (arguments, command utils) are reported as hubs and not followed: a change there needs
domain labels, not a file list. It is a text search, so it can over-report and it misses other dynamic
imports; read the list.
"""

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files

SOURCE_ROOTS = ("miles", "miles_plugins", "scripts", "examples", "tests")


def changed_files(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=d", f"{base}...HEAD"], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def module_tokens(path: str) -> list[str]:
    """Dotted names under which other files import `path`, most specific first."""
    if not path.endswith(".py"):
        return [path]
    if path.startswith("scripts/models/"):
        return [path, f'"{Path(path).stem}"']
    parts = Path(path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    tokens = [".".join(parts)]
    while len(parts) > 1:
        package_init = Path(*parts[:-1], "__init__.py")
        if not package_init.exists() or not re.search(rf"\b{re.escape(parts[-1])}\b", package_init.read_text()):
            break
        parts = parts[:-1]
        tokens.append(".".join(parts))
    return tokens + [path]


def reference_pattern(tokens: list[str]) -> re.Pattern:
    return re.compile(
        "|".join(
            (
                rf"{re.escape(token[:-1])}(?=[\"_{{-])"
                if token.startswith('"')
                else rf"(?<![\w.]){re.escape(token)}(?![\w])"
            )
            for token in tokens
        )
    )


def python_files(roots: tuple[str, ...]) -> list[str]:
    return sorted(str(p) for root in roots if Path(root).exists() for p in Path(root).rglob("*.py") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--depth", type=int, default=3, help="source-to-source hops to follow")
    parser.add_argument("--hub-limit", type=int, default=25, help="max referencing files before a file is a hub")
    args = parser.parse_args()

    changed = changed_files(args.base)
    registered = set(discover_ci_files())
    sources = {path: Path(path).read_text() for path in python_files(SOURCE_ROOTS) if path not in registered}
    tests = {path: Path(path).read_text() for path in registered}

    reached: dict[str, str] = {path: "changed" for path in changed if path in registered}
    frontier = [path for path in changed if path not in registered and (path.endswith(".py") or "/" in path)]
    seen = set(frontier)
    hubs: dict[str, int] = {}
    unmatched = {
        path for path in frontier if not path.endswith((".py", ".md", ".mdx")) and not path.startswith("docs/")
    }
    for hop in range(args.depth + 1):
        next_frontier = []
        for path in frontier:
            pattern = reference_pattern(module_tokens(path))
            test_refs = [test for test, text in tests.items() if pattern.search(text)]
            source_refs = [source for source, text in sources.items() if source != path and pattern.search(text)]
            if test_refs or source_refs:
                unmatched.discard(path)
            if len(test_refs) + len(source_refs) > args.hub_limit:
                hubs[path] = len(test_refs) + len(source_refs)
                continue
            for test in test_refs:
                reached.setdefault(test, f"{path} (hop {hop})")
            if hop == args.depth:
                continue
            for source in source_refs:
                if source not in seen:
                    seen.add(source)
                    next_frontier.append(source)
        frontier = next_frontier

    untraced = sorted(
        unmatched | {path for path in changed if "/" not in path and not path.endswith((".py", ".md", ".mdx"))}
    )
    rows = []
    for test in sorted(reached):
        for registry in collect_tests([test]):
            rows.append((test, registry))

    print(f"{len(changed)} changed files vs {args.base}; {len(rows)} registered tests reached\n")
    print("| test | backend | suite | labels | hardware | est_time | disabled | reached via |")
    print("|---|---|---|---|---|---|---|---|")
    for test, r in rows:
        print(
            f"| {test} | {r.backend.name} | {r.suite} | {','.join(r.labels)} | {','.join(r.hardware)} "
            f"| {r.est_time:.0f} | {r.disabled or ''} | {reached[test]} |"
        )

    gpu = [(test, r) for test, r in rows if r.backend == HWBackend.CUDA and not r.disabled]
    print(f"\n/rerun-test comments for the {len(gpu)} enabled CUDA tests (post each as its own comment):")
    for test, _ in gpu:
        print(f"/rerun-test {test}")

    label_cost = Counter(
        label
        for path in registered
        for r in collect_tests([path])
        if r.backend == HWBackend.CUDA and not r.disabled
        for label in r.labels
    )
    needed = Counter(label for _, r in gpu for label in r.labels)
    print("\nDomain labels of those tests (reached / all enabled CUDA tests carrying the label):")
    for label, count in needed.most_common():
        print(f"  run-ci-{label}: {count} / {label_cost[label]}")
    if hubs:
        print("\nHubs not followed (referenced by many files; a changed hub needs domain labels):")
        for path, count in sorted(hubs.items(), key=lambda item: -item[1]):
            print(f"  {path}: {count} referencing files{' (changed)' if path in changed else ''}")
    if untraced:
        print("\nChanged non-Python paths no source or test refers to (they may need broader CI):")
        for path in untraced:
            print(f"  {path}")


if __name__ == "__main__":
    main()
