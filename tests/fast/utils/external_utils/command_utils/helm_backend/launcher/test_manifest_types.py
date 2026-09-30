from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    STATEFUL_SET_KIND,
    GeneralManifestObject,
    Manifest,
    ManifestObjectKey,
    PodWorkloadObject,
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

        assert list(manifest.by_identity) == [("apps", "StatefulSet", NAMESPACE, ORCHESTRATOR)]

    def test_skips_the_empty_documents_helm_leaves_behind(self):
        """A template whose guard is off renders to nothing, and yaml reads that as a None document."""
        manifest = _parse("---\n---\n" + _rendered(_stateful_set()))

        assert len(manifest.objects) == 1

    def test_reads_the_replica_count_a_resize_moves(self):
        """The upgrade check compares this number, and a kind that carries no pod must not answer for it at all."""
        manifest = _parse(_rendered(_stateful_set(), {"kind": "ConfigMap", "metadata": {"name": "values"}}))

        stateful_set = manifest.by_identity[("apps", "StatefulSet", NAMESPACE, ORCHESTRATOR)]
        config_map = manifest.by_identity[("", "ConfigMap", NAMESPACE, "values")]

        assert isinstance(stateful_set, PodWorkloadObject) and stateful_set.replicas == 1
        assert isinstance(config_map, GeneralManifestObject)


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

        with pytest.raises(AssertionError, match="Service/rl/engine"):
            assert _parse(_rendered(service, service)).by_identity

    def test_two_served_versions_of_one_object_are_refused(self):
        """A crd answers on every version it serves, and the two spellings are one object to apply."""
        served = {"apiVersion": "example.com/v1", "kind": "Engine", "metadata": {"name": "engine"}}
        aliased = {**served, "apiVersion": "example.com/v1beta1"}

        with pytest.raises(AssertionError, match="example.com/Engine/rl/engine"):
            assert _parse(_rendered(served, aliased)).by_identity


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

    def test_refuses_two_containers_with_the_requested_name(self) -> None:
        """Duplicate container names make the requested command line ambiguous."""
        stateful_set = _stateful_set(command=["python", "--state-file", "/runs/a.state"])
        stateful_set["spec"]["template"]["spec"]["containers"].append(
            {"name": "orchestrator", "command": ["python", "--state-file", "/runs/b.state"]}
        )
        manifest = _parse(_rendered(stateful_set))

        with pytest.raises(AssertionError, match="declares 2 containers named 'orchestrator'"):
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator")

    def test_refuses_two_stateful_sets_with_the_requested_name(self) -> None:
        """A same-named StatefulSet in another namespace must not make the release ambiguous."""
        here = _stateful_set(command=["python", "--state-file", "/runs/a.state"])
        here["metadata"]["namespace"] = NAMESPACE
        elsewhere = _stateful_set(command=["python", "--state-file", "/runs/b.state"])
        elsewhere["metadata"]["namespace"] = "other"
        manifest = _parse(_rendered(here, elsewhere))

        assert (
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") == "/runs/a.state"
        )


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

        assert manifest.pod_workloads == []

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

    def test_keeps_a_custom_resource_whose_spec_is_shaped_like_no_pod(self):
        """An extra manifest is any object the user hands the chart, and validating it as a workload rejects it."""
        document = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PrometheusRule",
            "metadata": {"name": "alerts"},
            "spec": {"groups": [{"name": "run", "rules": [{"alert": "Down", "expr": "up == 0"}]}]},
        }

        manifest = _parse(_rendered(document))

        described = manifest.objects[0]

        assert not isinstance(described, PodWorkloadObject)
        assert described.spec == document["spec"]
        assert described.body == document

    def test_decodes_a_workload_kind_as_the_object_that_carries_a_pod(self):
        """Only the variant that models a pod may answer for replicas and containers, so the kind has to pick it."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "train.py"])))

        described = manifest.objects[0]

        assert isinstance(described, PodWorkloadObject)
        assert described.replicas == 1
        assert [found.name for found in described.containers] == ["orchestrator"]


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


class TestWhichStatefulSetAFlagIsReadOff:
    def test_a_stateful_set_of_the_same_name_in_another_namespace_does_not_join_the_match(self):
        """Two of them tripped the at-most-one assert, which aborted every relaunch of the run."""
        manifest = _parse(
            _rendered(
                _stateful_set(command=["python", "--state-file", "/runs/a.state"]),
                _stateful_set(command=["python", "--state-file", "/runs/b.state"], namespace="other"),
            )
        )

        assert (
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") == "/runs/a.state"
        )

    def test_a_stateful_set_of_another_api_group_answers_for_nothing(self):
        """A crd may share the kind and the name, and its command line says nothing about this release."""
        foreign = _stateful_set(command=["python", "--state-file", "/runs/foreign.state"])
        foreign["apiVersion"] = "example.com/v1"

        manifest = _parse(_rendered(foreign))

        assert manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") is None

    def test_the_stateful_set_this_release_installed_is_still_the_one_read(self):
        """Narrowing the match must not stop the ordinary release from being found at all."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert str(manifest.state_file(stateful_set=ORCHESTRATOR, container="orchestrator")) == "/runs/a.state"


class TestHowAFlagMayBeSpelled:
    def test_a_flag_joined_to_its_value_by_an_equals_sign_is_read(self):
        """The uuid is passed this way, and reading it as absent made the relaunch mint a new one."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid=abc123"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") == "abc123"

    def test_a_flag_separated_from_its_value_is_still_read(self):
        """Both spellings reach the container the same way, and the launcher has to read either."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid", "abc123"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") == "abc123"

    def test_the_last_of_two_spellings_of_one_flag_wins(self):
        """argparse reads the last occurrence, so anything else answers with a value the run never used."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid", "first", "--run-uuid=second"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") == "second"

    def test_a_flag_the_command_never_names_is_still_absent(self):
        """A release installed without the flag has nothing for the relaunch to inherit."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") is None

    def test_a_longer_flag_that_starts_with_the_one_asked_for_is_not_read(self):
        """--run-uuid-file is another argument, and answering with its value would resume the wrong run."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid-file=/runs/uuid"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") is None

    def test_a_command_ending_with_the_bare_flag_still_stops_the_launch(self):
        """Reading past the end says nothing about which release is malformed."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid"])))

        with pytest.raises(AssertionError, match="takes a value"):
            manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator")


def _stateful_set_with_null_annotations(name: str = ORCHESTRATOR) -> dict:
    document = _stateful_set(name=name)
    document["spec"]["template"]["metadata"] = {"annotations": None}
    return document


class TestAPodTemplateWhoseAnnotationsAreNull:
    def test_a_release_carrying_one_still_parses(self):
        """An extra manifest written this way used to take the whole relaunch down in Manifest.parse."""
        manifest = _parse(_rendered(_stateful_set_with_null_annotations()))

        assert [described.metadata.name for described in manifest.objects] == [ORCHESTRATOR]

    def test_it_reads_as_an_object_that_carries_no_restart_stamp(self):
        """An empty annotations block and a null one say the same thing: this object was never hot restarted."""
        manifest = _parse(_rendered(_stateful_set_with_null_annotations()))

        assert manifest.restart_at(object_name=ORCHESTRATOR) is None

    def test_the_null_block_reaches_the_diff_as_the_empty_block_it_was_read_as(self):
        """What the launch compares is the parsed object, so the normalization has to be visible there too."""
        manifest = _parse(_rendered(_stateful_set_with_null_annotations()))

        assert manifest.objects[0].body["spec"]["template"]["metadata"] == {"annotations": {}}

    def test_an_object_that_does_carry_annotations_still_reads_them(self):
        """Normalizing the null must not swallow the stamp a hot restart writes into the same block."""
        manifest = _parse(_rendered(_stateful_set(annotations={RESTART_AT_ANNOTATION: _STAMP})))

        assert manifest.restart_at(object_name=ORCHESTRATOR) == _STAMP
