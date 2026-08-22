from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    STATEFUL_SET_KIND,
    Manifest,
    ManifestObjectKey,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"
NAMESPACE = "rl"


def _parse(rendered: str) -> Manifest:
    return Manifest.parse(rendered, namespace=NAMESPACE)


def _rendered(*documents: dict) -> str:
    return "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in documents)


def _stateful_set(
    *,
    name: str = ORCHESTRATOR,
    command: list[str] | None = None,
    annotations: dict[str, Any] | None = None,
    namespace: str | None = None,
) -> dict:
    container = {"name": "orchestrator", "image": "miles:dev"}
    if command is not None:
        container["command"] = command
    template: dict[str, Any] = {"spec": {"containers": [container]}}
    if annotations is not None:
        template["metadata"] = {"annotations": annotations}
    metadata: dict[str, Any] = {"name": name}
    if namespace is not None:
        metadata["namespace"] = namespace
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": metadata,
        "spec": {"replicas": 1, "template": template},
    }


class TestParse:
    def test_identifies_an_object_the_way_the_api_server_does(self):
        """Kind and name alone name several distinct objects, and a run may install more than one of them."""
        manifest = _parse(_rendered(_stateful_set()))

        assert list(manifest.by_identity) == [("apps/v1", "StatefulSet", NAMESPACE, ORCHESTRATOR)]

    def test_skips_the_empty_documents_helm_leaves_behind(self):
        """A template whose guard is off renders to nothing, and yaml reads that as a None document."""
        manifest = _parse("---\n---\n" + _rendered(_stateful_set()))

        assert len(manifest.objects) == 1

    def test_reads_the_replica_count_a_resize_moves(self):
        """The upgrade check compares this number, and a kind that has none must not read as zero."""
        manifest = _parse(_rendered(_stateful_set(), {"kind": "ConfigMap", "metadata": {"name": "values"}}))

        assert manifest.by_identity[("apps/v1", "StatefulSet", NAMESPACE, ORCHESTRATOR)].replicas == 1
        assert manifest.by_identity[("", "ConfigMap", NAMESPACE, "values")].replicas is None


class TestIdentity:
    def test_an_object_of_another_api_group_is_not_the_same_object(self):
        """A crd may share a kind and a name with a built-in, and folding them hides one of the two."""
        mine = {"apiVersion": "example.com/v1", "kind": "Service", "metadata": {"name": "engine"}}
        builtin = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}

        assert len(_parse(_rendered(mine, builtin)).by_identity) == 2

    def test_an_object_of_another_namespace_is_not_the_same_object(self):
        """An object may name any namespace, and two that do are two objects the cluster keeps apart."""
        here = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}
        elsewhere = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine", "namespace": "other"}}

        assert len(_parse(_rendered(here, elsewhere)).by_identity) == 2

    def test_naming_this_release_own_namespace_is_the_same_object(self):
        """Spelling out the namespace the object lands in anyway must not read as a second object."""
        implicit = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}
        explicit = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine", "namespace": NAMESPACE}}

        assert _parse(_rendered(implicit)).by_identity.keys() == _parse(_rendered(explicit)).by_identity.keys()

    def test_two_objects_of_one_identity_are_refused(self):
        """Keeping only the last of them would let a relaunch change the other one without being noticed."""
        service = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}

        with pytest.raises(AssertionError, match="v1/Service"):
            assert _parse(_rendered(service, service)).by_identity


class TestFlagValue:
    def test_reads_what_the_installed_release_was_told(self):
        """The pod command line is the only place a launch can read back what the one before it decided."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert (
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") == "/runs/a.state"
        )

    def test_reads_only_the_object_it_was_asked_about(self):
        """Every pod of a run is launched by the same image, so a flag found elsewhere means something else."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.flag_value("--state-file", stateful_set="another-run", container="orchestrator") is None

    def test_reads_only_the_container_it_was_asked_about(self):
        """One pod can carry a sidecar, and its command line answers for the sidecar only."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="worker") is None

    def test_refuses_a_flag_the_command_leaves_unanswered(self):
        """Reading past the end would crash with an index, which says nothing about which release is malformed."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file"])))

        with pytest.raises(AssertionError, match="takes a value"):
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator")


class TestStateFile:
    def test_finds_the_file_the_installed_orchestrator_already_writes(self):
        """Re-attaching means waiting on the verdict of the launch that is running, not opening a second one."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert str(manifest.state_file(stateful_set=ORCHESTRATOR, container="orchestrator")) == "/runs/a.state"

    def test_names_nothing_when_no_container_of_that_name_carries_the_flag(self):
        """A release installed without an orchestrator has no verdict to inherit."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "-m", "something"])))

        assert manifest.state_file(stateful_set=ORCHESTRATOR, container="orchestrator") is None


class TestKindsItDoesNotModel:
    def test_carries_every_kind_this_chart_renders_through_untouched(self):
        """Only replicas and a pod template are read; the rest of a spec still has to reach the diff verbatim."""
        documents = [
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "engine"},
                "spec": {"clusterIP": "None", "ports": [{"port": 30000, "name": "primary"}]},
            },
            {
                "apiVersion": "leaderworkerset.x-k8s.io/v1",
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "engine"},
                "spec": {
                    "replicas": 2,
                    "leaderWorkerTemplate": {"size": 2, "workerTemplate": {"spec": {"containers": []}}},
                },
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {"name": "pairing"},
                "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "uninstall"},
                "spec": {
                    "completions": 1,
                    "template": {"spec": {"containers": [{"name": "helm", "image": "miles:dev"}]}},
                },
            },
        ]

        manifest = _parse(_rendered(*documents))

        assert [described.body for described in manifest.objects] == documents

    def test_reads_no_replicas_off_a_kind_that_has_none(self):
        """A Service and a Role never scale, and inventing a count for them would read as a resize."""
        manifest = _parse(
            _rendered({"kind": "Service", "metadata": {"name": "engine"}, "spec": {"clusterIP": "None"}})
        )

        assert manifest.objects[0].replicas is None

    def test_finds_no_container_in_a_workload_it_does_not_model(self):
        """The state file flag means the orchestrator's container, and every other pod runs the same image."""
        manifest = _parse(
            _rendered(
                {
                    "apiVersion": "leaderworkerset.x-k8s.io/v1",
                    "kind": "LeaderWorkerSet",
                    "metadata": {"name": "engine"},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "orchestrator",
                                            "command": ["python", "--state-file", "/runs/a.state"],
                                        }
                                    ]
                                }
                            }
                        }
                    },
                }
            )
        )

        assert manifest.state_file(stateful_set="engine", container="orchestrator") is None


_STAMP = "2026-08-12T09:00:00+00:00"
_STAMP_WITH_MICROSECONDS = "2026-08-12T09:00:00.123456+00:00"


class TestTheRestartStamp:
    def test_a_manifest_that_was_never_hot_restarted_carries_none(self):
        """Inventing a stamp would roll the pods of every run on its first ordinary relaunch."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator")))

        assert manifest.restart_at(object_name="orchestrator") is None

    def test_each_object_is_asked_for_its_own_stamp(self):
        """A pool that never got the stamp must not be rendered with it and turned into a refused diff."""
        manifest = _parse(
            _rendered(
                _stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: _STAMP}),
                _stateful_set(name="rollout-executor"),
            )
        )

        assert manifest.restart_at(object_name="orchestrator") == _STAMP
        assert manifest.restart_at(object_name="rollout-executor") is None

    def test_the_stamp_is_read_off_the_stateful_set_and_not_off_a_service_of_the_same_name(self):
        """The chart renders the headless Service first, and reading that one always answers None."""
        manifest = _parse(
            _rendered(
                {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "orchestrator"}},
                _stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: _STAMP}),
            )
        )

        assert manifest.restart_at(object_name="orchestrator") == _STAMP

    @pytest.mark.parametrize("stamp", [_STAMP, _STAMP_WITH_MICROSECONDS])
    def test_a_stamp_is_read_back_exactly_as_it_was_written(self, stamp: str):
        """A stamp that comes back in another spelling is a pod template diff the gate refuses forever."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: stamp})))

        assert manifest.restart_at(object_name="orchestrator") == stamp

    def test_a_stamp_that_is_no_timestamp_is_carried_as_it_stands(self):
        """This launch never has to read a stamp, only to render back the one it found."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: "soon"})))

        assert manifest.restart_at(object_name="orchestrator") == "soon"


class TestLookingAnObjectUpByItsKey:
    def test_a_name_shared_by_two_kinds_answers_the_kind_that_was_asked_for(self):
        """The chart renders a Service beside every StatefulSet, and they share the object's name."""
        manifest = _parse(
            _rendered(
                {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "orchestrator"}},
                _stateful_set(name="orchestrator", command=["python", "train.py"]),
            )
        )

        found = manifest.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name="orchestrator"))

        assert found is not None and found.kind == STATEFUL_SET_KIND

    def test_a_key_no_object_carries_answers_nothing(self):
        """A release that carries no such object is the ordinary answer, not a failure."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator")))

        assert manifest.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name="engine")) is None

    def test_a_key_two_objects_share_stops_the_launch(self):
        """Answering for whichever rendered first would gate the launch on an object nobody named."""
        manifest = _parse(
            _rendered(
                _stateful_set(name="orchestrator", namespace="rl"),
                _stateful_set(name="orchestrator", namespace="other"),
            )
        )

        with pytest.raises(AssertionError, match="objects keyed"):
            manifest.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name="orchestrator"))


class TestAnnotationsThatAreNotStrings:
    def test_an_object_carrying_a_non_string_annotation_still_parses(self):
        """A user's own annotation must not make the launcher fail to read the release it is upgrading."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator", annotations={"replicas": 3})))

        assert manifest.restart_at(object_name="orchestrator") is None
