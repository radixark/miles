from __future__ import annotations

from collections.abc import Callable

import pytest
from tests.fast.utils.workers.rpc.server.fake_workers import ExecutorUnderTest

from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


@pytest.fixture
def make_executor() -> Callable[[type], ExecutorUnderTest]:
    def build(worker_cls: type) -> ExecutorUnderTest:
        specs = collect_rpc_method_specs(worker_cls)
        return ExecutorUnderTest(executor=RpcCallExecutor(worker=worker_cls(), specs=specs), specs=specs)

    return build
