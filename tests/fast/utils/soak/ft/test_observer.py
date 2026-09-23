import httpx
import pytest
from tests.fast.utils.soak.soak_fakes import _cell, _FakeCellApi, _patch_http
from tests.utils.soak.ft.observers import CellObserver
from tests.utils.soak.ft.types import CellTarget

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.workers.naming import compute_cell_id

_BASE_URL = "http://api:18080"
_ACTOR_0 = compute_cell_id(pool_id="actor", cell_index=0)
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
