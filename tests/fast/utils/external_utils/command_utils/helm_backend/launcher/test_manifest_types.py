import pytest
import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest

ORCHESTRATOR = "myrun-miles-run-orchestrator"
NAMESPACE = "rl"


def _parse(rendered: str) -> Manifest:
    return Manifest.parse(rendered, namespace=NAMESPACE)


def _rendered(*documents: dict) -> str:
    return "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in documents)


def _stateful_set(*, name: str = ORCHESTRATOR, command: list[str] | None = None) -> dict:
    container = {"name": "orchestrator", "image": "miles:dev"}
    if command is not None:
        container["command"] = command
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": name},
        "spec": {"replicas": 1, "template": {"spec": {"containers": [container]}}},
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
