import subprocess

import pytest

from tests.fast.utils.soak.soak_fakes import _cell
from tests.utils.soak.ft.observers import CellObserver
from miles.utils.ft_utils.api_server.models import TriState


class TestPoolObservation:
    @pytest.mark.parametrize(
        "cell_type,replicas,healthy,serving,ready",
        [
            ("actor", 1, TriState.TRUE, TriState.TRUE, True),
            ("actor", 2, TriState.TRUE, TriState.TRUE, False),
            ("actor", 1, TriState.TRUE, TriState.FALSE, True),
            ("actor", 1, TriState.FALSE, TriState.TRUE, False),
            ("rollout", 1, TriState.TRUE, TriState.FALSE, False),
            ("rollout", 1, TriState.TRUE, TriState.TRUE, True),
        ],
    )
    async def test_ready_requires_every_expected_cell(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cell_type: str,
        replicas: int,
        healthy: TriState,
        serving: TriState,
        ready: bool,
    ) -> None:
        """Readiness matches the desired count and applies serving requirements only to rollout cells."""

        def read_replicas(*, namespace: str, workload: str) -> int:
            assert (namespace, workload) == ("rl", "statefulset/train")
            return replicas

        monkeypatch.setattr("tests.utils.soak.ft.observers.read_replicas", read_replicas)
        observer = CellObserver(
            base_url="http://api",
            cell_types={cell_type},
            namespace="rl",
            pool_workloads_of_cell_type={cell_type: "statefulset/train"},
        )
        errors: dict[str, str] = {}
        pools = await observer._observe_pools(
            cells=[
                _cell("selected-0", cell_type=cell_type, healthy=healthy, serving=serving),
                _cell("other-0", cell_type="unrelated", healthy=TriState.FALSE),
            ],
            errors=errors,
        )
        assert errors == {}
        assert len(pools) == 1
        assert (pools[0].identity, pools[0].replicas, pools[0].ready) == ("statefulset/train", replicas, ready)

    async def test_failed_pool_read_is_not_reported_as_an_empty_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed Kubernetes read records an error instead of inventing zero replicas."""

        def read_replicas(*, namespace: str, workload: str) -> int:
            raise subprocess.CalledProcessError(returncode=1, cmd=["kubectl", "get", workload])

        monkeypatch.setattr("tests.utils.soak.ft.observers.read_replicas", read_replicas)
        observer = CellObserver(
            base_url="http://api",
            cell_types={"actor"},
            namespace="rl",
            pool_workloads_of_cell_type={"actor": "statefulset/train"},
        )
        errors: dict[str, str] = {}
        assert await observer._observe_pools(cells=[], errors=errors) == []
        assert set(errors) == {"pool:statefulset/train"}

    async def test_unavailable_cells_do_not_imply_a_ready_pool(self) -> None:
        """An unavailable cell listing cannot establish pool readiness."""
        observer = CellObserver(
            base_url="http://api",
            cell_types={"actor"},
            namespace="rl",
            pool_workloads_of_cell_type={"actor": "statefulset/train"},
        )
        assert await observer._observe_pools(cells=None, errors={}) == []
