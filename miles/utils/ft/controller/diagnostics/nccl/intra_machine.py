from __future__ import annotations

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.diagnostics.nccl.utils import build_nccl_test_cmd, run_nccl_test
from miles.utils.ft.models.diagnostics import DiagnosticResult


class IntraMachineCommDiagnostic(BaseDiagnostic):
    """Single-node intra-machine communication diagnostic.

    Runs ``all_reduce_perf`` on one node and compares the measured
    bus bandwidth against an expected baseline.
    """

    diagnostic_type = "intra_machine"

    def __init__(
        self,
        expected_bandwidth_gbps: float = 350.0,
        num_gpus: int = 8,
        nccl_test_binary: str = "all_reduce_perf",
    ) -> None:
        self._expected_bandwidth_gbps = expected_bandwidth_gbps
        self._num_gpus = num_gpus
        self._nccl_test_binary = nccl_test_binary

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return await run_nccl_test(
            cmd=build_nccl_test_cmd(binary=self._nccl_test_binary, num_gpus=self._num_gpus),
            node_id=node_id,
            diagnostic_type=self.diagnostic_type,
            expected_bandwidth_gbps=self._expected_bandwidth_gbps,
            timeout_seconds=timeout_seconds,
            log_prefix=self.diagnostic_type,
        )
