import copy
from collections.abc import Callable
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher import manifest_diff
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest

NAMESPACE = "rl"


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
    return Manifest.parse(
        "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in objects), namespace=NAMESPACE
    )


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

        assert diff.scaled == [
            "leaderworkerset.x-k8s.io/v1/LeaderWorkerSet/rl/myrun-miles-run-engine: replicas 2 -> 6"
        ]

    def test_refuses_a_configmap_whose_data_changed(self):
        """The only configmap this chart renders holds the job that uninstalls the run, so its content is the run's."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: objects[1]["data"].update({"values.yaml": "run: {id: x}\n"})),
        )

        assert diff.changed == ["v1/ConfigMap/rl/myrun-miles-run-values: data.values.yaml"]

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
            "leaderworkerset.x-k8s.io/v1/LeaderWorkerSet/rl/myrun-miles-run-engine: "
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
            "leaderworkerset.x-k8s.io/v1/LeaderWorkerSet/rl/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].command.[2]"
        ]

    def test_refuses_an_object_the_upgrade_would_add(self):
        """A workload appearing mid-run was never part of the run being upgraded."""
        second = _leader_worker_set() | {"metadata": {"name": "second"}}
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects.append(second))
        )

        assert not diff.is_allowed
        assert diff.added == ["leaderworkerset.x-k8s.io/v1/LeaderWorkerSet/rl/second"]

    def test_refuses_an_object_the_upgrade_would_delete(self):
        """Upgrading would remove the orchestrator, and with it the run it is driving."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects.pop())
        )

        assert not diff.is_allowed
        assert diff.removed == ["apps/v1/StatefulSet/rl/myrun-miles-run-orchestrator"]

    def test_refuses_replicas_moving_on_a_kind_that_does_not_scale(self):
        """Only a LeaderWorkerSet adds cells without touching the pods it already has."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[2]["spec"].update(replicas=2))
        )

        assert not diff.is_allowed
        assert diff.changed == ["apps/v1/StatefulSet/rl/myrun-miles-run-orchestrator: spec.replicas"]

    def test_refuses_a_data_block_moving_on_a_kind_that_is_not_a_configmap(self):
        """A Secret's data is mounted into running pods, so rewriting it is not a free change."""
        secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "creds"}, "data": {"token": "YQ=="}}
        before = _manifest([*_objects(), secret])
        after = _manifest([*_objects(), secret | {"data": {"token": "Yg=="}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.changed == ["v1/Secret/rl/creds: data.token"]

    def test_refuses_a_change_to_an_object_another_namespace_shares_a_name_with(self):
        """A release may hold both, and folding them would let this edit through as no change at all."""
        here = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}, "spec": {"clusterIP": "None"}}
        elsewhere = {**here, "metadata": {"name": "engine", "namespace": "other"}}
        before = _manifest([*_objects(), here, elsewhere])
        after = _manifest([*_objects(), here, {**elsewhere, "spec": {"clusterIP": "10.0.0.1"}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.changed == ["v1/Service/other/engine: spec.clusterIP"]

    def test_refuses_a_change_to_an_object_another_api_group_shares_a_name_with(self):
        """A crd of the same kind and name is a second object, and only apiVersion tells the two apart."""
        builtin = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "engine"},
            "spec": {"clusterIP": "None"},
        }
        crd = {**builtin, "apiVersion": "example.com/v1"}
        before = _manifest([*_objects(), builtin, crd])
        after = _manifest([*_objects(), builtin, {**crd, "spec": {"clusterIP": "10.0.0.1"}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.changed == ["example.com/v1/Service/rl/engine: spec.clusterIP"]

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
        assert diff.scaled == [
            "leaderworkerset.x-k8s.io/v1/LeaderWorkerSet/rl/myrun-miles-run-engine: replicas 2 -> 6"
        ]


def _foreign_leader_worker_set(replicas: int = 2, api_version: str = "example.com/v1") -> dict[str, Any]:
    return {
        "apiVersion": api_version,
        "kind": "LeaderWorkerSet",
        "metadata": {"name": "someone-elses-leaderworkerset"},
        "spec": {"replicas": replicas},
    }


_FOREIGN_KEY = ManifestObjectKey(kind="LeaderWorkerSet", name="someone-elses-leaderworkerset")


class TestScalingIsLimitedToTheSupportedApi:
    def test_refuses_a_replica_change_on_a_leaderworkerset_of_another_api(self):
        """The exemption is written for the api this launcher drives, and nothing else answers for it."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
        )

        assert not diff.is_allowed
        assert diff.disallowed_changed == [
            "example.com/v1/LeaderWorkerSet/rl/someone-elses-leaderworkerset: spec.replicas"
        ]

    def test_reports_no_scaling_for_a_leaderworkerset_of_another_api(self):
        """Reporting it as scaling is what let the change through the restart gate without a word."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
        )

        assert diff.allowed_changed == []

    def test_counts_a_foreign_replica_change_as_a_rebuild(self):
        """Only the supported api is known to add cells without restarting what is already running."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
        )

        assert diff.rebuilds(key=_FOREIGN_KEY)

    def test_refuses_a_replica_change_on_an_unsupported_version_of_the_same_api_group(self):
        """A future or older version of the api can size a set by something other than spec.replicas."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set(api_version="leaderworkerset.x-k8s.io/v1alpha1")]),
            after=_manifest([_foreign_leader_worker_set(replicas=6, api_version="leaderworkerset.x-k8s.io/v1alpha1")]),
        )

        assert not diff.is_allowed

    def test_still_scales_the_supported_api(self):
        """The whole point of the exemption is that this run's own engine pool may grow."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set(api_version="leaderworkerset.x-k8s.io/v1")]),
            after=_manifest([_foreign_leader_worker_set(replicas=6, api_version="leaderworkerset.x-k8s.io/v1")]),
        )

        assert diff.is_allowed
        assert diff.allowed_changed == [
            "leaderworkerset.x-k8s.io/v1/LeaderWorkerSet/rl/someone-elses-leaderworkerset: replicas 2 -> 6"
        ]

    def test_a_whitelisted_foreign_object_is_still_allowed_to_change(self):
        """A hot restart names the objects it may rebuild, and that escape hatch is api-agnostic."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
            allow_diff_object_keys=frozenset({_FOREIGN_KEY}),
        )

        assert diff.is_allowed
