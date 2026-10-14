"""List the CI test files a branch's changes can reach, with their registrations.

Run from the repo root: PYTHONPATH=. python3 .claude/skills/pr-ci-targeted/affected_tests.py [--base origin/main]

Python dependencies come from the import graph: absolute, relative and `from pkg import module` imports, bare
imports of a sibling file put on sys.path, and dotted module paths or file paths inside string literals
(--custom-loss-function-path, `python tools/x.py` subprocesses). A model definition under scripts/models/ is
referenced by its name in a string literal (launchers load it by --model-name), a conftest.py by every test under
its directory, and any other changed file by its path. Dependents are followed up to --depth hops, and every
registered test found is reported. A file whose dependents reach more than --hub-limit GPU tests within --depth hops
(arguments, launcher utils) is a hub and is not followed: a change there needs domain labels, not a file list. The
depth is a locality bound, not a completeness one: the import graph is dense enough (model plugins register their
arguments in miles/utils/arguments.py, which nearly everything imports) that every training-side file reaches
every e2e test at a fixpoint. Files left unfollowed at the bound are reported. Other dynamic loading is not seen;
read the list.
"""

import argparse
import ast
import functools
import re
import subprocess
import warnings
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tests.ci.ci_policy import registration_matches_selection
from tests.ci.ci_register import CIRegistry, HWBackend, collect_tests, discover_ci_files
from tests.ci.stage_selection import known_no_gpu_path

SOURCE_ROOTS = ("miles", "miles_plugins", "scripts", "examples", "tests")
ENTRY_POINTS = ("train.py", "train_async.py", "train_multi_policy.py", "serve_tinker.py")
MAX_UNFOLLOWED_LISTED = 20
DOTTED_NAME = re.compile(r"(?<![\w./-])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")


def changed_files(base: str) -> list[str]:
    """Every path the branch touches; a rename or deletion keeps its old path so its importers are traced."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def in_import_graph(path: str) -> bool:
    return path in ENTRY_POINTS or path.startswith(tuple(f"{root}/" for root in SOURCE_ROOTS))


def load_registrations(files: list[str]) -> tuple[dict[str, list[CIRegistry]], dict[str, str]]:
    """Registrations per test file, and the files whose registration cannot be read, with the reason."""
    registrations, unreadable = {}, {}
    for path in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                registries = collect_tests([path], sanity_check=False)
            except (SyntaxError, ValueError) as error:
                unreadable[path] = str(error).splitlines()[0].removeprefix(f"{path}: ")
                continue
        if registries:
            registrations[path] = registries
        else:
            unreadable[path] = "no CI registration"
    return registrations, unreadable


def python_files(roots: tuple[str, ...]) -> list[str]:
    return sorted(str(p) for root in roots if Path(root).exists() for p in Path(root).rglob("*.py") if p.is_file())


def module_name(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def parse(path: str) -> ast.Module | None:
    try:
        return ast.parse(Path(path).read_text(), filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return None


def imported_files(path: str, tree: ast.Module, modules: dict[str, str], paths: set[str]) -> set[str]:
    """Files `path` imports, or names as a dotted module path in a string literal."""
    package = module_name(path) if path.endswith("__init__.py") else module_name(path).rpartition(".")[0]

    def resolve(name: str) -> set[str]:
        parts = name.split(".")
        found = {modules[m] for m in (".".join(parts[: i + 1]) for i in range(len(parts))) if m in modules}
        sibling = str(Path(path).parent / f"{parts[0]}.py")
        if not found and sibling in paths:
            found.add(sibling)
        return found

    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets |= resolve(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            if base:
                targets |= resolve(base)
            for alias in node.names:
                if f"{base}.{alias.name}" in modules:
                    targets.add(modules[f"{base}.{alias.name}"])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "." in node.value:
            for name in DOTTED_NAME.findall(node.value):
                if name.split(".")[0] in SOURCE_ROOTS:
                    targets |= resolve(name)
    targets.discard(path)
    return targets


def model_name_pattern(stem: str) -> re.Pattern:
    """A model name used whole or as the base of a variant (`-5layer`, `_lora`), not as a longer name's prefix."""
    return re.compile(rf"(?<![\w.-]){re.escape(stem)}(?![\w.])")


def literal_names_file(literal: str, holder: str, path: str) -> bool:
    """`literal` in `holder` names `path`: the whole path, a path suffix, or a bare file name next to `holder`."""
    if path in literal:
        return True
    if "/" in literal:
        return literal.endswith(".py") and f"/{path}".endswith(f"/{literal.lstrip('/')}")
    return literal == Path(path).name and Path(holder).parent == Path(path).parent


def string_literals(tree: ast.Module) -> list[str]:
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def source_graph(changed: list[str]) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    """Each source file's imported files, text and string literals; deleted modules stay importable by name."""
    files = python_files(SOURCE_ROOTS) + [path for path in ENTRY_POINTS if Path(path).exists()]
    modules = {module_name(path): path for path in files + [p for p in changed if p.endswith(".py")]}
    paths = set(modules.values())
    trees = {path: tree for path in files if (tree := parse(path)) is not None}
    imports = {path: imported_files(path, tree, modules, paths) for path, tree in trees.items()}
    texts = {path: Path(path).read_text() for path in trees}
    literals = {path: string_literals(tree) for path, tree in trees.items()}
    return imports, texts, literals


@dataclass(frozen=True)
class Hub:
    gpu_tests: int  # enabled CUDA tests its dependents reach within the depth bound
    reached_from: str  # "changed", or the hop and file the trace met it from


@dataclass(frozen=True)
class Trace:
    reached: dict[str, str]  # registered test -> the first changed or intermediate file it was reached from
    hubs: dict[str, Hub]
    unfollowed: list[str]  # importers of the last hop, left at the depth bound
    outside_trace: list[str]
    on_training_path: list[str]
    unreferenced: list[str]


def trace(changed: list[str], registrations: dict[str, list[CIRegistry]], *, depth: int, hub_limit: int) -> Trace:
    gpu_tests = {
        path for path, rs in registrations.items() if any(r.backend == HWBackend.CUDA and not r.disabled for r in rs)
    }
    imports, texts, literals = source_graph(changed)
    importers: dict[str, set[str]] = {}
    for path, targets in imports.items():
        for target in targets:
            importers.setdefault(target, set()).add(path)
    run_by_training, frontier = set(), [path for path in ENTRY_POINTS if path in imports]
    while frontier:
        path = frontier.pop()
        if path not in run_by_training:
            run_by_training.add(path)
            frontier.extend(imports.get(path, ()))

    py_literals = {path: [v for v in values if ".py" in v] for path, values in literals.items()}

    @functools.cache
    def referrers(path: str) -> frozenset[str]:
        if not path.endswith(".py"):
            return frozenset(other for other, text in texts.items() if path in text)
        refs = importers.get(path, set()) | {
            other for other, values in py_literals.items() if any(literal_names_file(v, other, path) for v in values)
        }
        if Path(path).name == "conftest.py":
            refs |= {test for test in registrations if test.startswith(f"{Path(path).parent}/")}
        if path.startswith("scripts/models/"):
            pattern = model_name_pattern(Path(path).stem)
            refs |= {other for other, values in literals.items() if any(pattern.search(v) for v in values)}
        return frozenset(refs - {path})

    @functools.cache
    def gpu_reach(path: str) -> int:
        found, frontier, seen = set(), {path}, {path}
        for _ in range(depth + 1):
            frontier = {ref for node in frontier for ref in referrers(node)} - seen
            seen |= frontier
            found |= frontier & gpu_tests
            frontier -= registrations.keys()
        return len(found)

    reached = {path: "changed" for path in changed if path in registrations}
    frontier = [
        path for path in changed if path not in registrations and (path.endswith(".py") or not known_no_gpu_path(path))
    ]
    seen = set(frontier)
    parents: dict[str, str] = {}
    hubs: dict[str, Hub] = {}
    unfollowed: set[str] = set()
    unreferenced = []
    for hop in range(depth + 1):
        next_frontier = []
        for path in frontier:
            if hop == 0 and path.endswith(".py") and not referrers(path):
                unreferenced.append(path)
            reach = gpu_reach(path)
            if reach > hub_limit:
                hubs[path] = Hub(reach, f"hop {hop} from {parents[path]}" if path in parents else "changed")
                continue
            for ref in sorted(referrers(path)):
                if ref in registrations:
                    reached.setdefault(ref, f"{path} (hop {hop})")
                elif ref in seen:
                    continue
                elif hop < depth:
                    seen.add(ref)
                    parents[ref] = path
                    next_frontier.append(ref)
                else:
                    unfollowed.add(ref)
        frontier = next_frontier

    return Trace(
        reached=reached,
        hubs=hubs,
        unfollowed=sorted(unfollowed),
        outside_trace=sorted(
            path
            for path in changed
            if path not in registrations
            and not known_no_gpu_path(path)
            and not (path.endswith(".py") and in_import_graph(path))
        ),
        on_training_path=sorted(set(changed) & run_by_training),
        unreferenced=sorted(unreferenced),
    )


def rerun_targets(result: Trace, registrations: dict[str, list[CIRegistry]]) -> list[str]:
    """Reached tests a PR run would skip without a label: enabled CUDA tests and labeled CPU tests."""
    return sorted(
        test
        for test in result.reached
        if any(
            not r.disabled and (r.backend == HWBackend.CUDA or (r.backend == HWBackend.CPU and r.labels))
            for r in registrations[test]
        )
    )


def selected_by_label(registration: CIRegistry, label: str) -> bool:
    """Whether `run-ci-<label>` selects this registration on a PR (regular cadence, so never a nightly one)."""
    return label in registration.labels and registration_matches_selection(
        registration.labels, registration.nightly, admit_nightly_tests=False, include_labels={label}
    )


def label_coverage(targets: list[str], registrations: dict[str, list[CIRegistry]]) -> list[tuple[str, int, int]]:
    """(label, reached tests it selects, all tests it selects) for each label that selects a target."""

    def selected(tests: Iterable[str]) -> Counter:
        return Counter(
            label
            for test in tests
            for r in registrations[test]
            if not r.disabled and r.backend != HWBackend.ROCM
            for label in r.labels
            if selected_by_label(r, label)
        )

    reached, total = selected(targets), selected(registrations)
    return [(label, count, total[label]) for label, count in reached.most_common()]


def nightly_only(targets: list[str], registrations: dict[str, list[CIRegistry]]) -> list[str]:
    return [
        test
        for test in targets
        if all(r.nightly for r in registrations[test] if not r.disabled and r.backend != HWBackend.ROCM)
    ]


def print_report(
    result: Trace,
    registrations: dict[str, list[CIRegistry]],
    unreadable: dict[str, str],
    *,
    base: str,
    n_changed: int,
    depth: int,
    hub_limit: int,
) -> None:
    rows = [(test, r) for test in sorted(result.reached) for r in registrations[test]]
    print(f"{n_changed} changed files vs {base}; {len(rows)} registered tests reached\n")
    print("| test | backend | suite | labels | hardware | nightly | est_time | disabled | first reached via |")
    print("|---|---|---|---|---|---|---|---|---|")
    for test, r in rows:
        print(
            f"| {test} | {r.backend.name} | {r.suite} | {','.join(r.labels)} | {','.join(r.hardware)} "
            f"| {'yes' if r.nightly else ''} | {r.est_time:.0f} | {r.disabled or ''} | {result.reached[test]} |"
        )

    targets = rerun_targets(result, registrations)
    print(f"\n/rerun-test comments for the {len(targets)} enabled CUDA and labeled CPU tests (one comment each):")
    for test in targets:
        print(f"/rerun-test {test}")

    print("\nDomain labels that select those tests on a PR (reached / all tests the label selects):")
    for label, reached, total in label_coverage(targets, registrations):
        print(f"  run-ci-{label}: {reached} / {total}")

    unfollowed = result.unfollowed
    if len(unfollowed) > MAX_UNFOLLOWED_LISTED:
        unfollowed = [*unfollowed[:MAX_UNFOLLOWED_LISTED], f"... {len(unfollowed) - MAX_UNFOLLOWED_LISTED} more"]
    sections = [
        (
            "Nightly-only targets (no PR label selects them; keep their /rerun-test lines, which bypass the cadence"
            " gate)",
            nightly_only(targets, registrations),
        ),
        (
            f"Hubs not followed (each reaches more than {hub_limit} GPU tests within {depth} hops; needs domain labels"
            " unless CI does not run the path to it)",
            [
                f"{path}: reaches {hub.gpu_tests} GPU tests ({hub.reached_from})"
                for path, hub in sorted(result.hubs.items())
            ],
        ),
        (
            f"Trace stopped at --depth {depth}; these importers of the last hop were not followed (rerun with a larger"
            " --depth to see where they lead)",
            unfollowed,
        ),
        ("Changed paths that affect GPU runtime outside the import trace (need label-based CI)", result.outside_trace),
        (
            "Changed files the training entry points load (every e2e training run executes them; the trace stops at"
            " the launcher hub, so add the domain labels of the training paths they change)",
            result.on_training_path,
        ),
        (
            "Changed Python files nothing imports or names (entry points, or loaded some other way; check)",
            result.unreferenced,
        ),
        (
            "Test files whose CI registration cannot be read (not traced as tests; fix before running CI)",
            [f"{path}: {reason}" for path, reason in sorted(unreadable.items())],
        ),
    ]
    for title, lines in sections:
        if lines:
            print(f"\n{title}:")
            for line in lines:
                print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--depth", type=int, default=3, help="importer hops to follow")
    parser.add_argument(
        "--hub-limit", type=int, default=40, help="a file reaching more enabled CUDA tests than this is a hub"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    changed = changed_files(args.base)
    registrations, unreadable = load_registrations(discover_ci_files())
    result = trace(changed, registrations, depth=args.depth, hub_limit=args.hub_limit)
    print_report(
        result,
        registrations,
        unreadable,
        base=args.base,
        n_changed=len(changed),
        depth=args.depth,
        hub_limit=args.hub_limit,
    )


if __name__ == "__main__":
    main()
