import subprocess

import httpx
import pytest
from tests.fast.utils.soak.soak_fakes import (
    _cell,
    _FakeCellApi,
    _FakeKubectl,
    _fault_target,
    _patch_http,
    _pod_json,
    _process_target,
)
from tests.utils.soak.ft import observers as observers_module
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.observers import CellObserver, create_cell_observer
from tests.utils.soak.ft.types import CellTarget

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.test_utils.fault_injector.actions.process import FailureMode
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.types import ClusterBackend

_BASE_URL = "http://api:18080"
_ACTOR_0 = compute_cell_id(pool_id="actor", cell_index=0)
_ACTOR_1 = compute_cell_id(pool_id="actor", cell_index=1)
_ROLLOUT_0 = compute_cell_id(pool_id="rollout", cell_index=0)


def _targets_by_identity(targets: list[CellTarget] | None) -> dict[str, CellTarget]:
    assert targets is not None
    return {target.identity: target for target in targets}


class TestCellObserverCells:
    async def test_cells_of_observed_types_become_targets_with_identity_incarnation_and_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each observed cell maps name, workers hash, liveness and readiness onto its target."""
        api = _FakeCellApi(
            [
                _cell(_ACTOR_0, cell_type="actor", workers_hash="h0"),
                _cell(_ROLLOUT_0, cell_type="rollout", workers_hash="h1", serving=TriState.FALSE),
                _cell("critic-000", cell_type="critic"),
            ]
        )
        _patch_http(monkeypatch, api)

        observation = await CellObserver(base_url=_BASE_URL, cell_types={"actor", "rollout"}).observe()

        assert observation.errors == {}
        targets = _targets_by_identity(observation.targets)
        assert set(targets) == {_ACTOR_0, _ROLLOUT_0}
        assert (targets[_ACTOR_0].kind, targets[_ACTOR_0].incarnation, targets[_ACTOR_0].ready) == (
            "actor",
            "h0",
            True,
        )
        rollout = targets[_ROLLOUT_0]
        assert (rollout.kind, rollout.incarnation, rollout.alive, rollout.ready) == ("rollout", "h1", True, False)
        assert all(target.fault_target is None and target.pods == [] for target in targets.values())

    @pytest.mark.parametrize("reply", [500, 404, {"items": [{"metadata": {}}]}])
    async def test_an_unreadable_cell_list_leaves_no_targets_and_records_the_error(
        self, monkeypatch: pytest.MonkeyPatch, reply: int | dict
    ) -> None:
        """A failed or malformed cell read is a failed observation, never an empty cluster."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor")])
        api.list_reply = reply
        _patch_http(monkeypatch, api)

        observation = await CellObserver(base_url=_BASE_URL, cell_types={"actor"}).observe()

        assert observation.targets is None
        assert set(observation.errors) == {"cells"}

    async def test_an_empty_cell_list_is_an_observed_empty_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A readable list without cells is a successful observation of zero targets."""
        _patch_http(monkeypatch, _FakeCellApi([]))

        observation = await CellObserver(base_url=_BASE_URL, cell_types={"actor"}).observe()

        assert observation.targets == []
        assert observation.errors == {}


class TestCellObserverFaultTargets:
    async def test_fault_targets_are_read_at_rank_zero_only_for_the_types_that_need_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only forms that address a process read its fault target, and they read rank 0."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor"), _cell(_ROLLOUT_0, cell_type="rollout")])
        api.fault_targets[_ACTOR_0] = _fault_target(_ACTOR_0)
        _patch_http(monkeypatch, api)

        observation = await CellObserver(
            base_url=_BASE_URL, cell_types={"actor", "rollout"}, fault_target_cell_types=frozenset({"actor"})
        ).observe()

        targets = _targets_by_identity(observation.targets)
        assert targets[_ACTOR_0].fault_target == _fault_target(_ACTOR_0)
        assert targets[_ROLLOUT_0].fault_target is None
        assert f"GET /api/v1/cells/{_ACTOR_0}/fault-target?sub_index=0" in api.paths
        assert not any(_ROLLOUT_0 in path for path in api.paths)

    @pytest.mark.parametrize(
        "reply",
        [
            _fault_target(_ACTOR_0, workers_hash="other"),
            _fault_target(_ACTOR_1),
            _fault_target(_ACTOR_0, rank=1),
            404,
            500,
        ],
    )
    async def test_a_fault_target_of_another_incarnation_cell_or_rank_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, reply: object
    ) -> None:
        """A reply that does not describe this cell's current rank-0 worker is an error, not a target."""
        api = _FakeCellApi([_cell(_ACTOR_0, cell_type="actor"), _cell(_ACTOR_1, cell_type="actor")])
        api.fault_targets = {_ACTOR_0: reply, _ACTOR_1: _fault_target(_ACTOR_1)}
        _patch_http(monkeypatch, api)

        observation = await CellObserver(
            base_url=_BASE_URL, cell_types={"actor"}, fault_target_cell_types=frozenset({"actor"})
        ).observe()

        targets = _targets_by_identity(observation.targets)
        assert set(observation.errors) == {f"fault_target:{_ACTOR_0}"}
        assert targets[_ACTOR_0].fault_target is None
        assert targets[_ACTOR_1].fault_target == _fault_target(_ACTOR_1)


class TestCellObserverPods:
    async def test_a_ray_observation_never_reads_pods(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a release there is no pod to attach and kubectl is never called."""
        _patch_http(monkeypatch, _FakeCellApi([_cell(_ACTOR_0, cell_type="actor")]))
        kubectl = _FakeKubectl(pods=[])
        monkeypatch.setattr(observers_module, "run_process", kubectl)

        observation = await CellObserver(base_url=_BASE_URL, cell_types={"actor"}).observe()

        assert kubectl.calls == []
        assert observation.targets[0].pods == []

    async def test_release_pods_are_attached_to_their_cells_and_unowned_pods_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pods are read by release selector and matched to cells through their pool and index labels."""
        _patch_http(
            monkeypatch, _FakeCellApi([_cell(_ACTOR_0, cell_type="actor"), _cell(_ACTOR_1, cell_type="actor")])
        )
        kubectl = _FakeKubectl(
            pods=[
                _pod_json("p-a0", pool_id="actor", cell_index=0),
                _pod_json("p-a1", pool_id="actor", cell_index=1),
                _pod_json("p-a9", pool_id="actor", cell_index=9),
                _pod_json("p-unlabelled", pool_id="actor", cell_index=None),
            ]
        )
        monkeypatch.setattr(observers_module, "run_process", kubectl)

        observation = await CellObserver(
            base_url=_BASE_URL, cell_types={"actor"}, namespace="rl", release="miles-run-all"
        ).observe()

        [get] = kubectl.calls
        assert get[:3] == ["kubectl", "get", "pods"]
        assert get[get.index("--namespace") + 1] == "rl"
        assert "miles-run-all" in get[get.index("--selector") + 1]
        targets = _targets_by_identity(observation.targets)
        [pod] = targets[_ACTOR_0].pods
        assert (pod.namespace, pod.release, pod.name, pod.uid) == ("rl", "miles-run-all", "p-a0", "uid-p-a0")
        assert [one.name for one in targets[_ACTOR_1].pods] == ["p-a1"]
        assert observation.errors == {}

    async def test_a_failed_pod_read_is_an_error_not_an_observation_without_pods(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kubectl failure keeps the cell targets but marks the observation incomplete."""
        _patch_http(monkeypatch, _FakeCellApi([_cell(_ACTOR_0, cell_type="actor")]))
        monkeypatch.setattr(
            observers_module,
            "run_process",
            _FakeKubectl(pods=[], get_error=subprocess.CalledProcessError(1, ["kubectl"])),
        )

        observation = await CellObserver(
            base_url=_BASE_URL, cell_types={"actor"}, namespace="rl", release="miles-run-all"
        ).observe()

        assert set(observation.errors) == {"pods"}
        assert observation.targets[0].pods == []

    async def test_a_release_without_a_namespace_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading pods across all namespaces could attach another run's pods."""
        _patch_http(monkeypatch, _FakeCellApi([_cell(_ACTOR_0, cell_type="actor")]))

        with pytest.raises(AssertionError, match="namespace"):
            await CellObserver(base_url=_BASE_URL, cell_types={"actor"}, release="miles-run-all").observe()


class TestCellObserverProcesses:
    async def test_matching_process_targets_are_attached_per_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each pod of a cell type with a pattern gets its sampled process target."""
        _patch_http(monkeypatch, _FakeCellApi([_cell(_ROLLOUT_0, cell_type="rollout")]))
        kubectl = _FakeKubectl(
            pods=[_pod_json("p-r0", pool_id="rollout", cell_index=0)],
            process_targets={("p-r0", "engine"): _process_target(pod_uid="uid-p-r0", pattern="sglang")},
        )
        monkeypatch.setattr(observers_module, "run_process", kubectl)

        observation = await CellObserver(
            base_url=_BASE_URL,
            cell_types={"rollout"},
            namespace="rl",
            release="miles-run-all",
            process_patterns_of_type={"rollout": {"engine": "sglang"}},
        ).observe()

        [pod] = observation.targets[0].pods
        assert pod.process_targets == {"engine": _process_target(pod_uid="uid-p-r0", pattern="sglang")}
        exec_call = kubectl.calls[1]
        assert exec_call[:2] == ["kubectl", "exec"]
        assert exec_call[-3:] == ["observe", "uid-p-r0", "sglang"]
        assert observation.errors == {}

    @pytest.mark.parametrize(
        "reply",
        [
            _process_target(pod_uid="uid-other", pattern="sglang"),
            _process_target(pod_uid="uid-p-r0", pattern="other"),
            subprocess.CalledProcessError(1, ["kubectl"]),
        ],
    )
    async def test_a_process_sample_of_another_pod_or_pattern_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, reply: object
    ) -> None:
        """A sample not answering for this pod UID and pattern is an error and is not attached."""
        _patch_http(monkeypatch, _FakeCellApi([_cell(_ROLLOUT_0, cell_type="rollout")]))
        monkeypatch.setattr(
            observers_module,
            "run_process",
            _FakeKubectl(
                pods=[_pod_json("p-r0", pool_id="rollout", cell_index=0)],
                process_targets={("p-r0", "engine"): reply},
            ),
        )

        observation = await CellObserver(
            base_url=_BASE_URL,
            cell_types={"rollout"},
            namespace="rl",
            release="miles-run-all",
            process_patterns_of_type={"rollout": {"engine": "sglang"}},
        ).observe()

        assert set(observation.errors) == {"processes:p-r0:engine"}
        assert observation.targets[0].pods[0].process_targets == {}


class TestCreateCellObserver:
    def test_a_ray_run_is_observed_through_the_api_server_only(self) -> None:
        """A Ray observer has no namespace or release, so it never reads pods."""
        observer = create_cell_observer(
            base_url=_BASE_URL,
            cell_types={"actor"},
            forms={"actor": [InjectFaultForm(base_url=_BASE_URL, failure_mode=FailureMode.SIGKILL)]},
            config=ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id="260926-120000-000"),
        )

        assert (observer.namespace, observer.release) == (None, None)
        assert observer.fault_target_cell_types == frozenset({"actor"})
        assert observer.process_patterns_of_type == {"actor": {}}


class TestObserverHttpBoundary:
    async def test_a_transport_failure_is_recorded_as_a_cell_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unreachable api server yields a failed observation instead of raising."""
        real_client = httpx.AsyncClient

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(refuse), **kwargs)
        )

        observation = await CellObserver(base_url=_BASE_URL, cell_types={"actor"}).observe()

        assert observation.targets is None
        assert "ConnectError" in observation.errors["cells"]
