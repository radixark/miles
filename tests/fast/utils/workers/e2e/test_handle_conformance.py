from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.fast.utils.workers import conformance
from tests.fast.utils.workers.conformance import CHECK_IDS, CHECKS, ConformanceWorker, HandleCheck

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle


@pytest.fixture
def conformance_handle(spawn, make_handle) -> Iterator[BaseWorkerHandle]:
    server = spawn(specs_path=f"{conformance.__name__}.compute_specs")
    yield make_handle(server, worker_cls=ConformanceWorker)


class TestTheHandleContractOverAServeSubprocess:
    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_contract_holds(self, conformance_handle: RpcWorkerHandle, check: HandleCheck):
        """Every backend a handle can sit on answers the same contract; this is the serve-subprocess column."""
        await check(conformance_handle)
