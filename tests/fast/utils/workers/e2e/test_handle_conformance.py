from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.fast.utils.workers import conformance
from tests.fast.utils.workers.conformance import CHECK_IDS, CHECKS, ConformanceWorker, HandleCheck

from tests.fast.utils.workers.e2e.harness import ServerProcess, spawn_server, wait_until_serving

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle


# spawning the server costs a fresh interpreter that imports miles, and every check below
# only reads from it, so one server serves the whole class
@pytest.fixture(scope="class")
def conformance_server(tmp_path_factory) -> Iterator[ServerProcess]:
    root = tmp_path_factory.mktemp("conformance")
    state_dir = root / "state"
    state_dir.mkdir()
    server = spawn_server(
        state_dir=state_dir,
        log_path=root / "server.log",
        specs_path=f"{conformance.__name__}.compute_specs",
    )
    wait_until_serving(server)
    yield server
    server.stop()
    server.kill()


@pytest.fixture
def conformance_handle(conformance_server: ServerProcess, make_handle) -> Iterator[BaseWorkerHandle]:
    yield make_handle(conformance_server, worker_cls=ConformanceWorker)


class TestTheHandleContractOverAServeSubprocess:
    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_contract_holds(self, conformance_handle: RpcWorkerHandle, check: HandleCheck):
        """Every backend a handle can sit on answers the same contract; this is the serve-subprocess column."""
        await check(conformance_handle)
