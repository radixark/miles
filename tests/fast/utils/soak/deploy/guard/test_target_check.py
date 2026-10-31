import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import NAMESPACE, ORCHESTRATOR, RELEASE, ROLLOUT_EXECUTOR, TRAINER
from tests.fast.utils.soak.deploy.deploy_fakes import _deployment_target, _FakeWorkloadKubectl, _workload_item
from tests.utils.deploy.hot_restart.cluster_observer import LEADER_WORKER_SET_KIND, STATEFUL_SET_KIND
from tests.utils.soak.deploy.guard import target_check
from tests.utils.soak.deploy.guard.target_check import assert_workloads_unchanged

from miles.utils.test_utils.kubectl_reads import compute_release_selector
from miles.utils.workers.cell_operations.base import StaleFaultTargetError

_STAMPS = {ORCHESTRATOR: "t1", ROLLOUT_EXECUTOR: "t1", TRAINER: None}


def _install(monkeypatch: pytest.MonkeyPatch, **kwargs: list[dict]) -> _FakeWorkloadKubectl:
    kubectl = _FakeWorkloadKubectl(**kwargs)
    monkeypatch.setattr(target_check, "run_process", kubectl)
    return kubectl


def _observed_items() -> list[dict]:
    return [_workload_item(name, stamp=stamp) for name, stamp in _STAMPS.items() if name != TRAINER]


class TestAssertWorkloadsUnchanged:
    def test_the_same_uids_and_stamps_across_both_kinds_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """StatefulSets and LeaderWorkerSets together must match the observed generation exactly."""
        kubectl = _install(
            monkeypatch,
            stateful_sets=_observed_items(),
            leader_worker_sets=[_workload_item(TRAINER, stamp=None)],
        )

        assert_workloads_unchanged(_deployment_target(stamps=_STAMPS))

        assert [argv[2] for argv in kubectl.calls] == [STATEFUL_SET_KIND, LEADER_WORKER_SET_KIND]
        for argv in kubectl.calls:
            assert argv[argv.index("--namespace") + 1] == NAMESPACE
            assert argv[argv.index("--selector") + 1] == compute_release_selector(release=RELEASE)

    @pytest.mark.parametrize(
        ("stateful_sets", "trainer"),
        [
            pytest.param(
                [_workload_item(ORCHESTRATOR, stamp="t2"), _workload_item(ROLLOUT_EXECUTOR, stamp="t1")],
                _workload_item(TRAINER, stamp=None),
                id="restamped",
            ),
            pytest.param(
                [
                    _workload_item(ORCHESTRATOR, stamp="t1", uid="uid-new"),
                    _workload_item(ROLLOUT_EXECUTOR, stamp="t1"),
                ],
                _workload_item(TRAINER, stamp=None),
                id="replaced_uid",
            ),
            pytest.param(
                [_workload_item(ORCHESTRATOR, stamp="t1")], _workload_item(TRAINER, stamp=None), id="missing_workload"
            ),
            pytest.param(
                [
                    _workload_item(ORCHESTRATOR, stamp="t1"),
                    _workload_item(ROLLOUT_EXECUTOR, stamp="t1"),
                    _workload_item(f"{RELEASE}-extra", stamp=None),
                ],
                _workload_item(TRAINER, stamp=None),
                id="extra_workload",
            ),
            pytest.param(
                [_workload_item(ORCHESTRATOR, stamp="t1"), _workload_item(ROLLOUT_EXECUTOR, stamp="t1")],
                _workload_item(TRAINER, stamp="t1"),
                id="trainer_stamped",
            ),
        ],
    )
    def test_any_generation_change_is_stale(
        self, monkeypatch: pytest.MonkeyPatch, stateful_sets: list[dict], trainer: dict
    ) -> None:
        """A restamped, recreated, missing, extra or newly stamped workload means another take-over already ran."""
        _install(monkeypatch, stateful_sets=stateful_sets, leader_worker_sets=[trainer])

        with pytest.raises(StaleFaultTargetError, match="changed since observation"):
            assert_workloads_unchanged(_deployment_target(stamps=_STAMPS))

    def test_a_deleting_workload_is_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A workload being torn down cannot be upgraded in place."""
        items = [_workload_item(ORCHESTRATOR, stamp="t1", deleting=True), _workload_item(ROLLOUT_EXECUTOR, stamp="t1")]
        _install(monkeypatch, stateful_sets=items, leader_worker_sets=[_workload_item(TRAINER, stamp=None)])

        with pytest.raises(StaleFaultTargetError, match="deleting or ambiguous"):
            assert_workloads_unchanged(_deployment_target(stamps=_STAMPS))

    def test_a_name_listed_under_both_kinds_is_ambiguous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two objects with one name cannot be told apart by name-keyed stamps."""
        _install(
            monkeypatch,
            stateful_sets=_observed_items(),
            leader_worker_sets=[_workload_item(TRAINER, stamp=None), _workload_item(ORCHESTRATOR, stamp="t1")],
        )

        with pytest.raises(StaleFaultTargetError, match="deleting or ambiguous"):
            assert_workloads_unchanged(_deployment_target(stamps=_STAMPS))

    def test_a_workload_without_a_uid_is_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a uid a recreated workload is indistinguishable from the observed one."""
        items = [_workload_item(ORCHESTRATOR, stamp="t1", uid=""), _workload_item(ROLLOUT_EXECUTOR, stamp="t1")]
        _install(monkeypatch, stateful_sets=items, leader_worker_sets=[_workload_item(TRAINER, stamp=None)])

        with pytest.raises(StaleFaultTargetError, match="has no uid"):
            assert_workloads_unchanged(_deployment_target(stamps=_STAMPS))

    def test_an_empty_release_is_stale_even_against_an_empty_observation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No workloads at all is a gone release, never an unchanged one."""
        _install(monkeypatch, stateful_sets=[], leader_worker_sets=[])

        with pytest.raises(StaleFaultTargetError):
            assert_workloads_unchanged(_deployment_target(stamps={}))
