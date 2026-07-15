"""Resolve CI cadence, scope, and fast-fail policy from explicit inputs."""

import json
import os
import re
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from tests.ci.labels import KNOWN_LABELS

_RUN_CI_PREFIX = "run-ci-"
_WORKFLOW_ONLY_LABELS = {"nightly", "bypass-fastfail"}
_SAFE_RUN_CI_LABEL = re.compile(r"^run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*$")

REGULAR_CADENCE = "regular"
NIGHTLY_CADENCE = "nightly"
CI_CADENCES = frozenset({REGULAR_CADENCE, NIGHTLY_CADENCE})

# A scheduled trigger has no policy by itself. Each configured cron must map
# explicitly so a future cadence cannot silently inherit nightly behavior.
SCHEDULE_POLICIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "0 15 * * *": (NIGHTLY_CADENCE, ()),
}


@dataclass(frozen=True)
class RunPolicy:
    cadence: str
    include_labels: frozenset[str]
    bypass_fastfail: bool

    @property
    def is_nightly(self) -> bool:
        return self.cadence == NIGHTLY_CADENCE


@dataclass(frozen=True)
class WorkflowPolicy:
    cadence: str
    raw_labels: tuple[str, ...]
    bypass_fastfail: bool


def strip_run_ci_prefix(raw_labels: Iterable[str]) -> set[str]:
    """Strip the `run-ci-` prefix from each PR-side label.

    Inputs are the canonical PR-side CI label names forwarded by the workflow
    (e.g. `["run-ci-megatron", "nightly"]`). Empty input yields an empty set.
    Known workflow-only labels (`_WORKFLOW_ONLY_LABELS`) are consumed
    elsewhere and skipped silently; any other item missing the `run-ci-`
    prefix is skipped after a `warnings.warn(...)`, because silently
    including it would risk matching the wrong domain label (e.g. bare
    `"megatron"` colliding with a test's domain label by accident).
    """
    stripped: set[str] = set()
    for raw in raw_labels:
        if not raw or raw in _WORKFLOW_ONLY_LABELS:
            continue
        if raw.startswith(_RUN_CI_PREFIX):
            stripped.add(raw[len(_RUN_CI_PREFIX) :])
        else:
            warnings.warn(
                f"--labels entry {raw!r} is missing the expected {_RUN_CI_PREFIX!r} "
                f"prefix; ignoring. Domain labels must be raw `run-ci-<X>` strings.",
                stacklevel=2,
            )
    return stripped


def resolve_policy(cadence: str, raw_labels: set[str]) -> RunPolicy:
    """Resolve selection and within-stage failure behavior from explicit inputs.

    The workflow adapter resolves trigger-specific facts into a cadence and
    raw labels; this function never infers policy from a GitHub event name. A
    test runs iff it is cadence-eligible and declares no labels (always-run)
    or any of its labels is in the effective include set.

    Broad scopes are large include sets: `run-ci-all` includes every registered
    label, nightly cadence everything except `ft-long`, and `run-ci-image`
    everything except `ft-short` and `ft-long`. Branch order encodes the
    precedence `run-ci-all` > nightly > `run-ci-image`.

    Explicitly requested `run-ci-<x>` labels are unioned in last, so an
    explicit request always wins over a scope subtraction. A subtraction is
    not a per-test veto: a test carrying a subtracted label still runs when
    another of its labels is included.
    """
    if cadence not in CI_CADENCES:
        raise ValueError(f"Unknown CI cadence {cadence!r}; expected one of {sorted(CI_CADENCES)}")
    if "nightly" in raw_labels and cadence != NIGHTLY_CADENCE:
        raise ValueError("The nightly workflow label requires cadence='nightly'")

    requested = strip_run_ci_prefix(raw_labels) & set(KNOWN_LABELS)
    if "run-ci-all" in raw_labels:
        scope = set(KNOWN_LABELS)
    elif cadence == NIGHTLY_CADENCE:
        scope = set(KNOWN_LABELS) - {"ft-long"}
    elif "run-ci-image" in raw_labels:
        scope = set(KNOWN_LABELS) - {"ft-short", "ft-long"}
    else:
        scope = set()
    return RunPolicy(
        cadence=cadence,
        include_labels=frozenset(scope | requested),
        bypass_fastfail=cadence == NIGHTLY_CADENCE or "bypass-fastfail" in raw_labels,
    )


def _canonical_pr_labels(pr_labels_json: str) -> tuple[str, ...]:
    try:
        labels = json.loads(pr_labels_json)
    except json.JSONDecodeError as exc:
        raise ValueError("PR labels were not a JSON string array") from exc
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("PR labels were not a JSON string array")

    return tuple(
        label for label in labels if label in _WORKFLOW_ONLY_LABELS or _SAFE_RUN_CI_LABEL.fullmatch(label) is not None
    )


def resolve_workflow_inputs(event_name: str, schedule: str, pr_labels_json: str) -> WorkflowPolicy:
    """Adapt GitHub trigger facts to the workflow's stable policy outputs."""
    if event_name == "pull_request":
        raw_labels = _canonical_pr_labels(pr_labels_json)
        cadence = NIGHTLY_CADENCE if "nightly" in raw_labels else REGULAR_CADENCE
    elif event_name == "schedule":
        try:
            cadence, raw_labels = SCHEDULE_POLICIES[schedule]
        except KeyError as exc:
            raise ValueError(f"No CI policy is defined for schedule: {schedule}") from exc
    elif event_name == "workflow_dispatch":
        cadence, raw_labels = REGULAR_CADENCE, ()
    else:
        raise ValueError(f"Unsupported PR Test trigger: {event_name}")

    run_policy = resolve_policy(cadence, set(raw_labels))
    return WorkflowPolicy(
        cadence=cadence,
        raw_labels=raw_labels,
        bypass_fastfail=run_policy.bypass_fastfail,
    )


def _write_github_outputs(policy: WorkflowPolicy, output_path: str) -> None:
    raw_labels = " ".join(policy.raw_labels)
    bypass_fastfail = str(policy.bypass_fastfail).lower()
    with open(output_path, "a", encoding="utf-8") as output:
        output.write(f"cadence={policy.cadence}\n")
        output.write(f"raw_labels={raw_labels}\n")
        output.write(f"bypass_fastfail={bypass_fastfail}\n")
    print(f"Resolved CI policy: cadence={policy.cadence} labels=[{raw_labels}] bypass_fastfail={bypass_fastfail}")


def main() -> int:
    try:
        policy = resolve_workflow_inputs(
            event_name=os.environ["EVENT_NAME"],
            schedule=os.environ.get("SCHEDULE", ""),
            pr_labels_json=os.environ.get("PR_LABELS_JSON", ""),
        )
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    _write_github_outputs(policy, os.environ["GITHUB_OUTPUT"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
