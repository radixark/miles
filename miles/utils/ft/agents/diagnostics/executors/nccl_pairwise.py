from __future__ import annotations

import os

from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.diagnostics.utils.nccl_utils import build_nccl_test_cmd, run_nccl_test
from miles.utils.ft.models.diagnostics import DiagnosticResult

DEFAULT_NCCL_MASTER_PORT: int = 29500
_DEFAULT_NUM_GPUS: int = 8


class NcclPairwiseNodeExecutor(BaseNodeExecutor):
    """Two-node pairwise NCCL diagnostic (all_gather_perf).

    Runs ``all_gather_perf`` on one side of a 2-node pair and compares
    the measured bus bandwidth against an expected baseline.  Peer
    coordination is handled via MASTER_ADDR / MASTER_PORT environment
    variables (NCCL TCPStore rendezvous).
    """

    diagnostic_type = "nccl_pairwise"

    def __init__(
        self,
        expected_bandwidth_gbps: float = 40.0,
        num_gpus: int = _DEFAULT_NUM_GPUS,
        master_addr: str = "",
        master_port: int = DEFAULT_NCCL_MASTER_PORT,
        nccl_test_binary: str = "all_gather_perf",
    ) -> None:
        self._expected_bandwidth_gbps = expected_bandwidth_gbps
        self._num_gpus = num_gpus
        self._master_addr = master_addr
        self._master_port = master_port
        self._nccl_test_binary = nccl_test_binary

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = 180,
        *,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> DiagnosticResult:
        addr = master_addr if master_addr is not None else self._master_addr
        port = master_port if master_port is not None else self._master_port

        env = {**os.environ}
        if addr:
            env["MASTER_ADDR"] = addr
        env["MASTER_PORT"] = str(port)

        return await run_nccl_test(
            cmd=build_nccl_test_cmd(binary=self._nccl_test_binary, num_gpus=self._num_gpus),
            node_id=node_id,
            diagnostic_type=self.diagnostic_type,
            expected_bandwidth_gbps=self._expected_bandwidth_gbps,
            timeout_seconds=timeout_seconds,
            log_prefix=self.diagnostic_type,
            env=env,
        )
