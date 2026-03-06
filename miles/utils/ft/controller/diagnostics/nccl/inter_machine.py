from __future__ import annotations

import os

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.diagnostics.nccl.utils import build_nccl_test_cmd, run_nccl_test
from miles.utils.ft.models.diagnostics import DiagnosticResult


class InterMachineCommDiagnostic(BaseDiagnostic):
    """Two-node inter-machine communication diagnostic.

    Runs ``all_gather_perf`` on one side of a 2-node pair and compares
    the measured bus bandwidth against an expected baseline.  Peer
    coordination is handled via MASTER_ADDR / MASTER_PORT environment
    variables (NCCL TCPStore rendezvous).
    """

    diagnostic_type = "inter_machine"

    def __init__(
        self,
        expected_bandwidth_gbps: float = 40.0,
        num_gpus: int = 8,
        master_addr: str = "",
        master_port: int = 29500,
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
