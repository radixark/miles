"""Contract tests for the shared CI policy and GitHub trigger adapter."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.ci.ci_policy import NIGHTLY_CADENCE, REGULAR_CADENCE, resolve_policy, resolve_workflow_inputs
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize(
    ("event_name", "schedule", "labels_json", "cadence", "raw_labels", "bypass_fastfail"),
    [
        ("pull_request", "", "[]", REGULAR_CADENCE, (), False),
        (
            "pull_request",
            "",
            json.dumps(["run-ci-megatron", "bypass-fastfail"]),
            REGULAR_CADENCE,
            ("run-ci-megatron", "bypass-fastfail"),
            True,
        ),
        (
            "pull_request",
            "",
            json.dumps(
                [
                    "ignored",
                    "run-ci-megatron",
                    "nightly",
                    "run-ci-megatron",
                    "run-ci-a_B.c-d",
                    "run-ci-/unsafe",
                ]
            ),
            NIGHTLY_CADENCE,
            ("run-ci-megatron", "nightly", "run-ci-megatron", "run-ci-a_B.c-d"),
            True,
        ),
        ("schedule", "0 15 * * *", "not JSON", NIGHTLY_CADENCE, (), True),
        ("workflow_dispatch", "", "not JSON", REGULAR_CADENCE, (), False),
    ],
)
def test_trigger_facts_resolve_to_stable_workflow_outputs(
    event_name,
    schedule,
    labels_json,
    cadence,
    raw_labels,
    bypass_fastfail,
):
    policy = resolve_workflow_inputs(event_name, schedule, labels_json)

    assert policy.cadence == cadence
    assert policy.raw_labels == raw_labels
    assert policy.bypass_fastfail is bypass_fastfail


@pytest.mark.parametrize("labels_json", ["{", "{}", "null", '["run-ci-megatron", 1]'])
def test_pull_request_rejects_non_string_array_labels(labels_json):
    with pytest.raises(ValueError, match="PR labels were not a JSON string array"):
        resolve_workflow_inputs("pull_request", "", labels_json)


def test_unknown_schedule_is_not_assumed_to_be_nightly():
    with pytest.raises(ValueError, match=r"No CI policy is defined for schedule: 0 0 \* \* 0"):
        resolve_workflow_inputs("schedule", "0 0 * * 0", "[]")


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError, match="Unsupported PR Test trigger: push"):
        resolve_workflow_inputs("push", "", "[]")


def test_nightly_label_and_nightly_schedule_share_the_same_run_policy():
    labeled = resolve_workflow_inputs("pull_request", "", '["nightly"]')
    scheduled = resolve_workflow_inputs("schedule", "0 15 * * *", "not JSON")

    assert resolve_policy(labeled.cadence, set(labeled.raw_labels)) == resolve_policy(
        scheduled.cadence, set(scheduled.raw_labels)
    )


@pytest.mark.parametrize(
    ("labels_json", "expected_outputs"),
    [
        (
            "[]",
            "existing=value\ncadence=regular\nraw_labels=\nbypass_fastfail=false\n",
        ),
        (
            '["run-ci-megatron", "nightly", "run-ci-megatron", "ignored"]',
            "existing=value\ncadence=nightly\n"
            "raw_labels=run-ci-megatron nightly run-ci-megatron\n"
            "bypass_fastfail=true\n",
        ),
    ],
)
def test_cli_appends_exact_github_outputs(tmp_path, labels_json, expected_outputs):
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    output_path.write_text("existing=value\n")
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "pull_request",
            "SCHEDULE": "",
            "PR_LABELS_JSON": labels_json,
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.read_text() == expected_outputs
    assert "Resolved CI policy:" in result.stdout


def test_cli_fails_for_unknown_schedule(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "schedule",
            "SCHEDULE": "0 0 * * 0",
            "PR_LABELS_JSON": "not JSON",
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "::error::No CI policy is defined for schedule: 0 0 * * 0" in result.stderr
    assert not output_path.exists()
