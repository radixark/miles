import copy
from collections.abc import Callable
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher import manifest_diff
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest


def _leader_worker_set(replicas: int = 2) -> dict[str, Any]:
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": {"name": "myrun-miles-run-engine"},
        "spec": {
            "replicas": replicas,
            "leaderWorkerTemplate": {
                "workerTemplate": {
                    "spec": {
                        "containers": [{"name": "worker", "image": "miles:dev", "command": ["python", "-m", "engine"]}]
                    }
                }
            },
        },
    }


def _config_map() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "myrun-miles-run-values"},
        "data": {"values.yaml": "run: {}\n"},
    }


def _stateful_set() -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": "myrun-miles-run-orchestrator"},
        "spec": {
            "replicas": 1,
            "template": {"spec": {"containers": [{"name": "orchestrator", "image": "miles:dev"}]}},
        },
    }


def _manifest(objects: list[dict[str, Any]]) -> Manifest:
    return Manifest.parse("---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in objects))


def _objects() -> list[dict[str, Any]]:
    return [_leader_worker_set(), _config_map(), _stateful_set()]


def _manifest_after(mutate: Callable[[list[dict[str, Any]]], None]) -> Manifest:
    objects = copy.deepcopy(_objects())
    mutate(objects)
    return _manifest(objects)


def _worker_container(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return objects[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]


class TestManifestScaling:
    def test_allows_a_pool_that_only_grew(self):
        """Chart templates render the live run's own manifests, and only the cell count may move under it."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        assert diff.is_allowed

    def test_reports_the_replica_change_it_will_apply(self):
        """An accepted upgrade still restarts nothing but adds pods, and the user must see how many."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        assert diff.scaled == ["LeaderWorkerSet/myrun-miles-run-engine: replicas 2 -> 6"]

    def test_refuses_a_configmap_whose_data_changed(self):
        """The only configmap this chart renders holds the job that uninstalls the run, so its content is the run's."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: objects[1]["data"].update({"values.yaml": "run: {id: x}\n"})),
        )

        assert diff.changed == ["ConfigMap/myrun-miles-run-values: data.values.yaml"]

    def test_says_so_when_the_rendered_manifests_are_identical(self):
        """Relaunching the same run id is how users check on a run, and it must read as a no-op."""
        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=_manifest(_objects()))

        assert "nothing to change" in diff.summarize_scaling()


class TestManifestRefusals:
    def test_refuses_a_changed_container_image(self):
        """A new image restarts every pod of a live run, which is the restart this check exists to stop."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        assert not diff.is_allowed
        assert diff.changed == [
            "LeaderWorkerSet/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].image"
        ]

    def test_refuses_a_changed_container_command(self):
        """A template-only edit to the launch command would relaunch the pool_id as a different experiment."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(
                lambda objects: _worker_container(objects).update(command=["python", "-m", "other"])
            ),
        )

        assert not diff.is_allowed
        assert diff.changed == [
            "LeaderWorkerSet/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].command.[2]"
        ]

    def test_refuses_an_object_the_upgrade_would_add(self):
        """A workload appearing mid-run was never part of the run being upgraded."""
        second = _leader_worker_set() | {"metadata": {"name": "second"}}
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects.append(second))
        )

        assert not diff.is_allowed
        assert diff.added == ["LeaderWorkerSet/second"]

    def test_refuses_an_object_the_upgrade_would_delete(self):
        """Upgrading would remove the orchestrator, and with it the run it is driving."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects.pop())
        )

        assert not diff.is_allowed
        assert diff.removed == ["StatefulSet/myrun-miles-run-orchestrator"]

    def test_refuses_replicas_moving_on_a_kind_that_does_not_scale(self):
        """Only a LeaderWorkerSet adds cells without touching the pods it already has."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[2]["spec"].update(replicas=2))
        )

        assert not diff.is_allowed
        assert diff.changed == ["StatefulSet/myrun-miles-run-orchestrator: spec.replicas"]

    def test_refuses_a_data_block_moving_on_a_kind_that_is_not_a_configmap(self):
        """A Secret's data is mounted into running pods, so rewriting it is not a free change."""
        secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "creds"}, "data": {"token": "YQ=="}}
        before = _manifest([*_objects(), secret])
        after = _manifest([*_objects(), secret | {"data": {"token": "Yg=="}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.changed == ["Secret/creds: data.token"]

    def test_names_the_field_it_refused(self):
        """A refusal the user cannot locate just makes them reach for --skip-upgrade-check."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        assert "containers.[0].image" in diff.describe()

    def test_keeps_the_replica_change_alongside_the_refusal(self):
        """The user asked for both, and reporting only the refusal hides half of what they asked for."""

        def grow_and_repoint(objects: list[dict[str, Any]]) -> None:
            objects[0]["spec"].update(replicas=6)
            _worker_container(objects).update(image="miles:other")

        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=_manifest_after(grow_and_repoint))

        assert not diff.is_allowed
        assert diff.scaled == ["LeaderWorkerSet/myrun-miles-run-engine: replicas 2 -> 6"]
