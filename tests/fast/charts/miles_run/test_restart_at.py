import json
from typing import Any

from tests.fast.charts.utils import named_object, render_run, requires_helm, with_object_names

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION

ORCHESTRATOR = "myrun-miles-run-orchestrator"
ROLLOUT_EXECUTOR = "myrun-miles-run-rollout-executor"
ROUTER = "myrun-miles-run-router"
STAMP = "2026-08-12T09:00:00+00:00"

WORKERS = [
    {"name": "rollout-executor", "command": ["python", "-m", "serve"]},
    {"name": "router", "command": ["python", "-m", "router"]},
]


def _render(*args: str, workers: list[dict[str, Any]] = WORKERS) -> list[dict[str, Any]]:
    return render_run("--set-json", f"run.staticWorkers={json.dumps(with_object_names(workers))}", *args)


def _pod_annotations(objects: list[dict[str, Any]], name: str) -> dict[str, str]:
    return named_object(objects, "StatefulSet", name)["spec"]["template"]["metadata"].get("annotations", {})


@requires_helm
class TestRestartAtAnnotation:
    def test_an_ordinary_launch_stamps_no_annotation(self):
        """Stamping a value on every launch would roll a live run's pods for nothing."""
        objects = _render()

        assert _pod_annotations(objects, ORCHESTRATOR) == {}
        assert _pod_annotations(objects, ROLLOUT_EXECUTOR) == {}

    def test_the_orchestrator_carries_the_stamp_it_is_given(self):
        """The pod template only changes, and the StatefulSet only rolls, because of this annotation."""
        objects = _render("--set", f"run.orchestrator.restartAt={STAMP}")

        assert _pod_annotations(objects, ORCHESTRATOR) == {RESTART_AT_ANNOTATION: STAMP}

    def test_only_the_static_worker_that_is_given_a_stamp_carries_one(self):
        """The rollout executor is the second component a hot restart replaces; every other pod keeps running."""
        objects = _render(workers=[{**WORKERS[0], "restartAt": STAMP}, WORKERS[1]])

        assert _pod_annotations(objects, ROLLOUT_EXECUTOR) == {RESTART_AT_ANNOTATION: STAMP}
        assert _pod_annotations(objects, ROUTER) == {}

    def test_the_stamp_does_not_reach_the_statefulset_metadata(self):
        """Only a pod template change rolls the pods, so the annotation has to sit on the template."""
        objects = _render("--set", f"run.orchestrator.restartAt={STAMP}")

        assert RESTART_AT_ANNOTATION not in named_object(objects, "StatefulSet", ORCHESTRATOR)["metadata"].get(
            "annotations", {}
        )
