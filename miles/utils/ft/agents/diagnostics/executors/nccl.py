from __future__ import annotations

import os

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.diagnostics.utils.nccl_utils import build_nccl_test_cmd, run_nccl_test
from miles.utils.ft.agents.types import DiagnosticResult

DEFAULT_NCCL_MASTER_PORT: int = 29500
_DEFAULT_NUM_GPUS: int = 8


class NcclNodeExecutor(BaseNodeExecutor):
    """Unified NCCL diagnostic executor.

    Handles both single-node (alltoall_perf) and pairwise two-node
    (all_gather_perf) diagnostics.  The ``diagnostic_type`` selects which
    variant is run; pairwise mode is activated by passing ``master_addr``
    and/or ``master_port`` to :meth:`run`.
    """

    diagnostic_type: str

    def __init__(
        self,
        diagnostic_type: str,
        expected_bandwidth_gbps: float,
        num_gpus: int = _DEFAULT_NUM_GPUS,
        nccl_test_binary: str = "alltoall_perf",
    ) -> None:
        self.diagnostic_type = diagnostic_type
        self._expected_bandwidth_gbps = expected_bandwidth_gbps
        self._num_gpus = num_gpus
        self._nccl_test_binary = nccl_test_binary

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        *,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> DiagnosticResult:
        env: dict[str, str] | None = None
        if master_addr is not None or master_port is not None:
            env = {**os.environ}
            if master_addr is not None:
                env["MASTER_ADDR"] = master_addr
            env["MASTER_PORT"] = str(master_port if master_port is not None else DEFAULT_NCCL_MASTER_PORT)

        return await run_nccl_test(
            cmd=build_nccl_test_cmd(binary=self._nccl_test_binary, num_gpus=self._num_gpus),
            node_id=node_id,
            diagnostic_type=self.diagnostic_type,
            expected_bandwidth_gbps=self._expected_bandwidth_gbps,
            timeout_seconds=timeout_seconds,
            log_prefix=self.diagnostic_type,
            env=env,
        )
