import copy
from collections.abc import Callable
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend import elastic


def _values() -> dict[str, Any]:
    return {
        "infra": {
            "image": "miles:dev",
            "sharedStorage": {"claimName": "cluster-storage"},
        },
        "adhoc": {"enabled": False, "name": "", "completions": 1},
        "run": {
            "id": "260101-000000-000",
            "orchestrator": {"command": ["python", "train.py", "--x", "1"]},
            "staticWorkers": [{"name": "inference-router-0", "command": ["python", "-m", "router"]}],
            "inferenceEngines": [
                {"name": "inference-engine-0-0", "command": ["python", "-m", "engine"], "replicas": 2, "size": 4}
            ],
            "trainers": [{"name": "trainer-actor", "command": ["python", "-m", "serve"], "replicas": 2, "size": 2}],
        },
    }


def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    values = copy.deepcopy(_values())
    mutate(values)
    return values


class TestScaling:
    def test_allows_growing_an_engine_pool(self):
        """Adding engines to a live run is the whole point of relaunching a run id."""
        diff = elastic.diff_values(
            _values(), _mutated(lambda values: values["run"]["inferenceEngines"][0].update(replicas=6))
        )

        assert diff.is_allowed

    def test_reports_the_engine_replica_change_it_will_apply(self):
        """An accepted upgrade still changes a live run, so it must not happen silently."""
        diff = elastic.diff_values(
            _values(), _mutated(lambda values: values["run"]["inferenceEngines"][0].update(replicas=6))
        )

        assert diff.scaled == ["run.inferenceEngines.[0].replicas: 2 -> 6"]

    def test_allows_growing_a_trainer_pool(self):
        """Trainers heal and grow per data-parallel group exactly as engines do."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["run"]["trainers"][0].update(replicas=5)))

        assert diff.is_allowed
        assert diff.scaled == ["run.trainers.[0].replicas: 2 -> 5"]

    def test_allows_growing_a_pool_that_started_with_one_cell(self):
        """A run that began on one engine is exactly the run a user grows once it looks healthy."""
        before = _mutated(lambda values: values["run"]["inferenceEngines"][0].update(replicas=1))

        diff = elastic.diff_values(
            before, _mutated(lambda values: values["run"]["inferenceEngines"][0].update(replicas=4))
        )

        assert diff.is_allowed
        assert diff.scaled == ["run.inferenceEngines.[0].replicas: 1 -> 4"]

    def test_allows_shrinking_a_pool_back_to_one_cell(self):
        """Giving gpus back without ending the run is the other half of elastic scaling."""
        after = _mutated(lambda values: values["run"]["inferenceEngines"][0].update(replicas=1))

        diff = elastic.diff_values(_values(), after)

        assert diff.is_allowed
        assert diff.scaled == ["run.inferenceEngines.[0].replicas: 2 -> 1"]

    def test_allows_shrinking_a_trainer_pool_back_to_one_cell(self):
        """Trainers scale down the same way, and a rejected upgrade would strand the run at its old size."""
        after = _mutated(lambda values: values["run"]["trainers"][0].update(replicas=1))

        diff = elastic.diff_values(_values(), after)

        assert diff.is_allowed
        assert diff.scaled == ["run.trainers.[0].replicas: 2 -> 1"]

    def test_says_so_when_a_relaunch_changes_nothing(self):
        """Rerunning the same command is a common way to check on a run, and must read as a no-op."""
        assert "nothing to change" in elastic.diff_values(_values(), _values()).summarize_scaling()


class TestRefusals:
    def test_refuses_a_changed_infra_value(self):
        """A new image restarts every pod, killing the experiment the user meant to grow."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["infra"].update(image="miles:other")))

        assert not diff.is_allowed
        assert diff.changed == ["infra.image"]

    def test_refuses_a_changed_nested_infra_value(self):
        """Remounting the shared storage under a live run loses every checkpoint path it holds open."""
        diff = elastic.diff_values(
            _values(), _mutated(lambda values: values["infra"]["sharedStorage"].update(claimName="other"))
        )

        assert diff.changed == ["infra.sharedStorage.claimName"]

    def test_refuses_an_adhoc_section_that_woke_up(self):
        """An upgrade rendering an adhoc Job would run that adhoc command against a run already training."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["adhoc"].update(enabled=True)))

        assert diff.changed == ["adhoc.enabled"]

    def test_refuses_a_changed_adhoc_detail(self):
        """Everything outside the two replica counts is part of what the run already is."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["adhoc"].update(completions=4)))

        assert not diff.is_allowed

    def test_refuses_a_changed_run_id(self):
        """A different id is a different run, and upgrading in place would hide that entirely."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["run"].update(id="260101-000000-001")))

        assert diff.changed == ["run.id"]

    def test_refuses_a_changed_orchestrator_command(self):
        """The training arguments are the experiment, so changing them mid-run invalidates the results."""
        diff = elastic.diff_values(
            _values(), _mutated(lambda values: values["run"]["orchestrator"].update(command=["python", "other.py"]))
        )

        assert diff.changed == ["run.orchestrator.command"]

    def test_refuses_a_changed_pool_command(self):
        """An engine relaunched with different flags is a different engine, however many of them there are."""
        diff = elastic.diff_values(
            _values(),
            _mutated(lambda values: values["run"]["inferenceEngines"][0].update(command=["python", "-m", "other"])),
        )

        assert diff.changed == ["run.inferenceEngines.[0].command.[2]"]

    def test_refuses_a_resized_pool(self):
        """A different group size is a different parallelism, so it is a different run."""
        diff = elastic.diff_values(
            _values(), _mutated(lambda values: values["run"]["inferenceEngines"][0].update(size=8))
        )

        assert diff.changed == ["run.inferenceEngines.[0].size"]

    def test_refuses_a_new_pool(self):
        """A run gaining a workload mid-flight was never launched as the run being upgraded."""
        diff = elastic.diff_values(
            _values(),
            _mutated(lambda values: values["run"]["inferenceEngines"].append({"name": "second", "replicas": 1})),
        )

        assert diff.changed == ["run.inferenceEngines"]

    def test_refuses_a_disappearing_pool(self):
        """Upgrading would delete it, and with it whatever it was serving."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["run"]["staticWorkers"].clear()))

        assert diff.changed == ["run.staticWorkers"]

    def test_refuses_a_key_the_installed_values_never_had(self):
        """A value appearing out of nowhere changes the rendered manifests just as an edited one does."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["run"].update(extra="x")))

        assert diff.changed == ["run.extra"]

    def test_names_the_path_it_refused(self):
        """A refusal the user cannot act on just makes them reach for force."""
        diff = elastic.diff_values(_values(), _mutated(lambda values: values["infra"].update(image="miles:other")))

        assert "infra.image" in diff.describe()

    def test_keeps_the_replica_change_alongside_the_refusal(self):
        """The user asked for both, and reporting only the refusal hides half of what they asked for."""

        def grow_and_repoint(values: dict[str, Any]) -> None:
            values["run"]["inferenceEngines"][0].update(replicas=6)
            values["infra"].update(image="miles:other")

        diff = elastic.diff_values(_values(), _mutated(grow_and_repoint))

        assert not diff.is_allowed
        assert diff.scaled == ["run.inferenceEngines.[0].replicas: 2 -> 6"]


class TestMissingSides:
    def test_treats_a_release_with_no_recorded_values_as_empty(self):
        """helm reports null for a release installed without values, and that must not crash the check."""
        diff = elastic.diff_values(None, _values())

        assert not diff.is_allowed

    def test_agrees_with_itself_when_both_sides_are_absent(self):
        """Two empty value sets differ in nothing, so nothing may be refused."""
        assert elastic.diff_values(None, None).is_allowed


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


def _rendered(objects: list[dict[str, Any]]) -> str:
    return "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in objects)


def _objects() -> list[dict[str, Any]]:
    return [_leader_worker_set(), _config_map(), _stateful_set()]


def _rendered_after(mutate: Callable[[list[dict[str, Any]]], None]) -> str:
    objects = copy.deepcopy(_objects())
    mutate(objects)
    return _rendered(objects)


def _worker_container(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return objects[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]


class TestManifestScaling:
    def test_allows_a_pool_that_only_grew(self):
        """Chart templates render the live run's own manifests, and only the cell count may move under it."""
        diff = elastic.diff_manifests(
            _rendered(_objects()), _rendered_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        assert diff.is_allowed

    def test_reports_the_replica_change_it_will_apply(self):
        """An accepted upgrade still restarts nothing but adds pods, and the user must see how many."""
        diff = elastic.diff_manifests(
            _rendered(_objects()), _rendered_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        assert diff.scaled == ["LeaderWorkerSet/myrun-miles-run-engine: replicas 2 -> 6"]

    def test_allows_a_configmap_whose_data_changed(self):
        """The run's values configmap is rewritten on every relaunch and no pod is restarted by it."""
        diff = elastic.diff_manifests(
            _rendered(_objects()),
            _rendered_after(lambda objects: objects[1]["data"].update({"values.yaml": "run: {id: x}\n"})),
        )

        assert diff.is_allowed

    def test_says_so_when_the_rendered_manifests_are_identical(self):
        """Relaunching the same run id is how users check on a run, and it must read as a no-op."""
        diff = elastic.diff_manifests(_rendered(_objects()), _rendered(_objects()))

        assert "nothing to change" in diff.summarize_scaling()


class TestManifestRefusals:
    def test_refuses_a_changed_container_image(self):
        """A new image restarts every pod of a live run, which is the restart this check exists to stop."""
        diff = elastic.diff_manifests(
            _rendered(_objects()),
            _rendered_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        assert not diff.is_allowed
        assert diff.changed == [
            "LeaderWorkerSet/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].image"
        ]

    def test_refuses_a_changed_container_command(self):
        """A template-only edit to the launch command would relaunch the pool_id as a different experiment."""
        diff = elastic.diff_manifests(
            _rendered(_objects()),
            _rendered_after(lambda objects: _worker_container(objects).update(command=["python", "-m", "other"])),
        )

        assert not diff.is_allowed
        assert diff.changed == [
            "LeaderWorkerSet/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].command.[2]"
        ]

    def test_refuses_an_object_the_upgrade_would_add(self):
        """A workload appearing mid-run was never part of the run being upgraded."""
        second = _leader_worker_set() | {"metadata": {"name": "second"}}
        diff = elastic.diff_manifests(_rendered(_objects()), _rendered_after(lambda objects: objects.append(second)))

        assert not diff.is_allowed
        assert diff.added == ["LeaderWorkerSet/second"]

    def test_refuses_an_object_the_upgrade_would_delete(self):
        """Upgrading would remove the orchestrator, and with it the run it is driving."""
        diff = elastic.diff_manifests(_rendered(_objects()), _rendered_after(lambda objects: objects.pop()))

        assert not diff.is_allowed
        assert diff.removed == ["StatefulSet/myrun-miles-run-orchestrator"]

    def test_refuses_replicas_moving_on_a_kind_that_does_not_scale(self):
        """Only a LeaderWorkerSet adds cells without touching the pods it already has."""
        diff = elastic.diff_manifests(
            _rendered(_objects()), _rendered_after(lambda objects: objects[2]["spec"].update(replicas=2))
        )

        assert not diff.is_allowed
        assert diff.changed == ["StatefulSet/myrun-miles-run-orchestrator: spec.replicas"]

    def test_refuses_a_data_block_moving_on_a_kind_that_is_not_a_configmap(self):
        """A Secret's data is mounted into running pods, so rewriting it is not a free change."""
        secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "creds"}, "data": {"token": "YQ=="}}
        before = _rendered([*_objects(), secret])
        after = _rendered([*_objects(), secret | {"data": {"token": "Yg=="}}])

        diff = elastic.diff_manifests(before, after)

        assert not diff.is_allowed
        assert diff.changed == ["Secret/creds: data.token"]

    def test_names_the_field_it_refused(self):
        """A refusal the user cannot locate just makes them reach for force."""
        diff = elastic.diff_manifests(
            _rendered(_objects()),
            _rendered_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        assert "containers.[0].image" in diff.describe()

    def test_keeps_the_replica_change_alongside_the_refusal(self):
        """The user asked for both, and reporting only the refusal hides half of what they asked for."""

        def grow_and_repoint(objects: list[dict[str, Any]]) -> None:
            objects[0]["spec"].update(replicas=6)
            _worker_container(objects).update(image="miles:other")

        diff = elastic.diff_manifests(_rendered(_objects()), _rendered_after(grow_and_repoint))

        assert not diff.is_allowed
        assert diff.scaled == ["LeaderWorkerSet/myrun-miles-run-engine: replicas 2 -> 6"]


class TestManifestOf:
    def test_takes_the_manifest_out_of_a_dry_run(self):
        """helm prints the release summary and the notes around it, and neither is part of the objects."""
        output = (
            "NAME: myrun\nLAST DEPLOYED: today\nHOOKS:\nMANIFEST:\n"
            f"{_rendered([_config_map()])}\nNOTES:\nwatch your run with kubectl\n"
        )

        assert list(yaml.safe_load_all(elastic.manifest_of(output))) == [_config_map()]

    def test_keeps_the_whole_manifest_when_helm_prints_no_notes(self):
        """A chart without NOTES.txt still renders objects, and cutting at a missing marker would drop them."""
        output = f"NAME: myrun\nMANIFEST:\n{_rendered([_config_map()])}"

        assert list(yaml.safe_load_all(elastic.manifest_of(output))) == [_config_map()]

    def test_passes_a_bare_manifest_straight_through(self):
        """helm get manifest prints the objects alone, and the same parser reads both sides of the diff."""
        rendered = _rendered([_config_map()])

        assert elastic.manifest_of(rendered) == rendered
