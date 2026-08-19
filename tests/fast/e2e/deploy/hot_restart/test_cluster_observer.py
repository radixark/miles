import threading
import time

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import cluster_observer as cluster_module
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    LEADER_WORKER_SET_KIND,
    POD_KIND,
    STATEFUL_SET_KIND,
    ClusterObserver,
    ClusterSnapshot,
    PodFact,
    compute_trainer_rpc_url,
    parse_pod_facts,
    parse_workload_facts,
)
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    NAMESPACE,
    ORCHESTRATOR,
    RELEASE,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION


class TestComputeTrainerRpcUrl:
    def test_the_trainer_is_asked_for_its_boot_uuid_at_its_own_pod(self):
        """A trainer that did restart answers the same url with a different boot uuid."""
        url = compute_trainer_rpc_url(release=RELEASE, namespace=NAMESPACE, trainer_id="actor")

        assert url.startswith("http://")
        assert NAMESPACE in url
        assert url.endswith("/v1/health")


class TestParsePodFacts:
    def test_a_pod_is_identified_by_the_uid_it_was_created_with(self):
        """A replaced pod keeps its name, so only the uid tells it apart from the one that survived."""
        payload = {
            "items": [
                {
                    "metadata": {"name": "b", "uid": "uid-b"},
                    "status": {"containerStatuses": [{"restartCount": 1}, {"restartCount": 2}]},
                },
                {"metadata": {"name": "a", "uid": "uid-a"}, "status": {}},
            ]
        }

        assert parse_pod_facts(payload) == (
            PodFact(name="a", uid="uid-a", restart_count=0),
            PodFact(name="b", uid="uid-b", restart_count=3),
        )


class TestParseWorkloadFacts:
    def test_the_generation_and_the_stamp_of_each_statefulset_are_read(self):
        """A StatefulSet records both its generation and the canonical content of its pod template."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ORCHESTRATOR, "generation": 2},
                    "spec": {"template": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}}},
                },
                {"metadata": {"name": TRAINER, "generation": 1}, "spec": {"template": {"metadata": {}}}},
            ]
        }

        orchestrator, trainer = parse_workload_facts(payload, kind=STATEFUL_SET_KIND)

        assert (
            orchestrator.kind,
            orchestrator.name,
            orchestrator.generation,
            orchestrator.restart_at,
        ) == (STATEFUL_SET_KIND, ORCHESTRATOR, 2, "t1")
        assert (trainer.kind, trainer.name, trainer.generation, trainer.restart_at) == (
            STATEFUL_SET_KIND,
            TRAINER,
            1,
            None,
        )
        assert len(orchestrator.pod_template_fingerprint) == 64
        assert orchestrator.pod_template_fingerprint != trainer.pod_template_fingerprint

    def test_a_leaderworkerset_carries_its_stamp_on_the_template_of_its_group(self):
        """The trainer cells and the engines of a run are leaderworkersets, not statefulsets."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ENGINE_POOL, "generation": 3},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}}
                        }
                    },
                }
            ]
        }

        [engine_pool] = parse_workload_facts(payload, kind=LEADER_WORKER_SET_KIND)

        assert (engine_pool.kind, engine_pool.name, engine_pool.generation, engine_pool.restart_at) == (
            LEADER_WORKER_SET_KIND,
            ENGINE_POOL,
            3,
            "t1",
        )
        assert len(engine_pool.pod_template_fingerprint) == 64

    def test_equivalent_template_key_orders_have_the_same_fingerprint(self):
        """Kubernetes may return object keys in another order without changing the pod template."""
        first = {
            "items": [
                {
                    "metadata": {"name": TRAINER, "generation": 1},
                    "spec": {"template": {"metadata": {"labels": {"a": "1", "b": "2"}}, "spec": {}}},
                }
            ]
        }
        second = {
            "items": [
                {
                    "metadata": {"generation": 2, "name": TRAINER},
                    "spec": {"template": {"spec": {}, "metadata": {"labels": {"b": "2", "a": "1"}}}},
                }
            ]
        }

        [before] = parse_workload_facts(first, kind=STATEFUL_SET_KIND)
        [after] = parse_workload_facts(second, kind=STATEFUL_SET_KIND)

        assert before.generation != after.generation
        assert before.pod_template_fingerprint == after.pod_template_fingerprint

    def test_an_object_stamped_twice_over_is_refused(self):
        """One stamp per replaced object is what makes counting the stamps count the take-overs."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ENGINE_POOL, "generation": 3},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "leaderTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}},
                            "workerTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t2"}}},
                        }
                    },
                }
            ]
        }

        with pytest.raises(AssertionError, match="restart stamps"):
            parse_workload_facts(payload, kind=LEADER_WORKER_SET_KIND)


def _observer() -> ClusterObserver:
    return ClusterObserver(release=RELEASE, namespace=NAMESPACE, trainer_id="actor")


class TestClusterObserver:
    def _install_reader(self, monkeypatch, snapshots: list[ClusterSnapshot]) -> None:
        remaining = list(snapshots)
        monkeypatch.setattr(cluster_module, "read_cluster_snapshot", lambda **_kwargs: remaining.pop(0))

    def test_a_read_that_could_not_see_the_whole_release_is_counted_not_recorded(self, monkeypatch):
        """A verdict read off two lucky observations of a run nobody could reach proves nothing."""
        observer = _observer()
        self._install_reader(
            monkeypatch,
            [
                cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER)], reads_missing=(POD_KIND,)),
                cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)]),
            ],
        )

        observer.observe_once()
        observer.observe_once()

        assert len(observer.snapshots) == 1
        assert (observer.attempts, observer.failures) == (2, 1)

    def test_a_pods_read_that_failed_is_a_failed_read_and_not_a_vanished_release(self, monkeypatch):
        """These used to be the same thing, which left the "reads missing" branch unreachable."""
        monkeypatch.setattr(
            cluster_module,
            "_read_objects",
            lambda **kwargs: None if kwargs["kind"] == POD_KIND else {"items": []},
        )
        monkeypatch.setattr(cluster_module, "read_boot_uuid", lambda _url: "boot-a")

        snapshot = cluster_module.read_cluster_snapshot(
            release=RELEASE, namespace=NAMESPACE, trainer_rpc_url="http://x"
        )

        assert snapshot.reads_missing == (POD_KIND,) and not snapshot.describes_whole_release

    def test_a_read_that_raised_leaves_the_run_being_watched_rather_than_ending_it(self, monkeypatch):
        """kubectl answers late now and then, and the observer is the one thread this run cannot lose."""
        observer = _observer()
        monkeypatch.setattr(cluster_module, "read_cluster_snapshot", _raise_boom)

        observer.observe_once_or_warn()

        assert observer.snapshots == []
        assert observer.failures == 1

    def test_a_release_being_uninstalled_is_not_recorded_as_a_run_losing_its_pods(self, monkeypatch):
        """The last observation happens while the run is torn down, and every pod is gone by then."""
        observer = _observer()
        self._install_reader(monkeypatch, [cluster_snapshot(pods=[], workloads=[])])

        observer.observe_once()

        assert observer.snapshots == []


def _raise_boom(**_kwargs) -> None:
    raise RuntimeError("kubectl said no")


def _wait_until_threads_left(count: int) -> None:
    deadline = time.monotonic() + 5.0
    while threading.active_count() > count and time.monotonic() < deadline:
        time.sleep(0.01)


class TestObservingTheClusterInTheBackground:
    def test_the_closing_snapshot_is_taken_after_the_body_returns(self, monkeypatch):
        """The run ends inside the body, and the frame that shows its last pods comes after."""
        observer = _observer()
        taken: list[str] = []
        monkeypatch.setattr(
            cluster_module.ClusterObserver, "observe_once_or_warn", lambda _self: taken.append("polled")
        )
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once", lambda _self: taken.append("closing"))

        with cluster_module.observing_cluster(observer, poll_interval_seconds=0.0):
            pass

        assert taken[-1] == "closing"

    def test_an_observer_still_mid_read_when_asked_to_stop_is_reported(self, monkeypatch):
        """Reading what it collected while it is still writing would race it."""
        release = threading.Event()
        monkeypatch.setattr(
            cluster_module.ClusterObserver, "observe_once_or_warn", lambda _self: release.wait(timeout=30.0)
        )
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once", lambda _self: None)
        monkeypatch.setattr(cluster_module, "JOIN_TIMEOUT_SECONDS", 0.05)
        before = threading.active_count()

        try:
            with pytest.raises(AssertionError, match="still reading the run"):
                with cluster_module.observing_cluster(_observer(), poll_interval_seconds=0.0):
                    pass
        finally:
            release.set()
            _wait_until_threads_left(before)

    def test_a_body_that_raised_does_not_leave_the_observer_running(self, monkeypatch):
        """A leaked poller keeps reading a release the next test is about to install over."""
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once", lambda _self: None)
        before = threading.active_count()

        with pytest.raises(_BodyFailed):
            with cluster_module.observing_cluster(_observer(), poll_interval_seconds=0.0):
                raise _BodyFailed

        _wait_until_threads_left(before)
        assert threading.active_count() == before


class _BodyFailed(Exception):
    pass
