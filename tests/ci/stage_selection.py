"""Select PR GPU stages from changed paths and the resolved test policy."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tests.ci.ci_policy import RunPolicy, registration_matches_selection, strip_run_ci_prefix
from tests.ci.ci_register import CIRegistry
from tests.ci.labels import KNOWN_LABELS

# Only stages with jobs in the Miles PR workflows belong here; the external
# MI350 nightly suites are not local runner-allocation targets.
PR_GPU_STAGES = frozenset(
    {
        "stage-b-2-gpu-h200",
        "stage-c-2-gpu-h200",
        "stage-c-4-gpu-h200",
        "stage-c-8-gpu-h100",
        "stage-c-8-gpu-h200",
        "stage-c-4-gpu-mi350",
    }
)

_BROAD_PR_SCOPES = frozenset({"nightly", "run-ci-all", "run-ci-image"})
# Keep this allowlist to paths proven not to affect GPU runtime. Every
# unmatched path deliberately fans out to all local GPU stages.
_NO_GPU_PATHS = frozenset({".pre-commit-config.yaml", "scripts/tools/sync_example_docs.py"})
_NO_GPU_PREFIXES = ("docs/",)
_NO_GPU_SUFFIXES = (".md", ".mdx")
_STATUS = re.compile(r"^(?:[ADMTUXB]|[RC][0-9]{1,3})$")


@dataclass(frozen=True)
class ChangedFile:
    status: str
    paths: tuple[str, ...]


def read_changed_files(path: str) -> tuple[ChangedFile, ...] | None:
    """Parse `git diff --name-status -z`; return None when it is unusable."""
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if not data or not data.endswith(b"\0"):
        return None

    try:
        fields = [field.decode("utf-8") for field in data[:-1].split(b"\0")]
    except UnicodeDecodeError:
        return None

    changed_files: list[ChangedFile] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if _STATUS.fullmatch(status) is None:
            return None

        path_count = 2 if status[0] in {"R", "C"} else 1
        paths = tuple(fields[index : index + path_count])
        index += path_count
        if len(paths) != path_count or any(not changed_path for changed_path in paths):
            return None
        changed_files.append(ChangedFile(status=status, paths=paths))

    return tuple(changed_files) or None


def _runnable_stages(registrations: Iterable[CIRegistry], run_policy: RunPolicy) -> set[str]:
    return {
        registration.suite
        for registration in registrations
        if registration.suite in PR_GPU_STAGES
        and registration.disabled is None
        and registration_matches_selection(
            registration.labels,
            registration.nightly,
            admit_nightly_tests=run_policy.admit_nightly_tests,
            include_labels=run_policy.include_labels,
        )
    }


def _explicit_scope_stages(
    registrations: Iterable[CIRegistry],
    run_policy: RunPolicy,
    raw_labels: Iterable[str],
    runnable_stages: set[str],
) -> set[str]:
    raw_labels = set(raw_labels)
    if raw_labels & _BROAD_PR_SCOPES:
        return set(runnable_stages)

    requested_labels = strip_run_ci_prefix(raw_labels) & set(KNOWN_LABELS)
    return {
        registration.suite
        for registration in registrations
        if registration.suite in PR_GPU_STAGES
        and registration.disabled is None
        and (not registration.nightly or run_policy.admit_nightly_tests)
        and bool(set(registration.labels) & requested_labels)
    }


def _known_no_gpu_path(path: str) -> bool:
    return path in _NO_GPU_PATHS or path.startswith(_NO_GPU_PREFIXES) or path.endswith(_NO_GPU_SUFFIXES)


def _affected_stages(changed_files: Iterable[ChangedFile], registrations: Iterable[CIRegistry]) -> set[str]:
    stages_by_file: dict[str, set[str]] = {}
    registered_files: set[str] = set()
    for registration in registrations:
        registered_files.add(registration.filename)
        if registration.suite in PR_GPU_STAGES:
            stages_by_file.setdefault(registration.filename, set()).add(registration.suite)

    affected: set[str] = set()
    for changed_file in changed_files:
        for path in changed_file.paths:
            if path in registered_files:
                affected.update(stages_by_file.get(path, ()))
            elif _known_no_gpu_path(path):
                continue
            else:
                return set(PR_GPU_STAGES)
    return affected


def select_skipped_gpu_stages(
    *,
    event_name: str,
    changed_files: tuple[ChangedFile, ...] | None,
    registrations: Iterable[CIRegistry],
    run_policy: RunPolicy,
    raw_labels: Iterable[str],
) -> tuple[str, ...]:
    """Return local GPU stage IDs that a workflow can safely skip."""
    if event_name != "pull_request" or changed_files is None:
        return ()

    registrations = tuple(registrations)
    runnable = _runnable_stages(registrations, run_policy)
    affected = _affected_stages(changed_files, registrations)
    affected.update(_explicit_scope_stages(registrations, run_policy, raw_labels, runnable))
    return tuple(sorted(PR_GPU_STAGES - (runnable & affected)))
