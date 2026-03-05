from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.controller.diagnostics._nccl_utils import parse_avg_bus_bandwidth
from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models import DiagnosticResult

logger = logging.getLogger(__name__)


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
        cmd = [
            self._nccl_test_binary,
            "-b", "1M", "-e", "1G", "-f", "2",
            "-g", str(self._num_gpus),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            logger.warning(
                "intra_machine_exec_failed node=%s binary=%s",
                node_id, self._nccl_test_binary,
                exc_info=True,
            )
            return DiagnosticResult(
                diagnostic_type=self.diagnostic_type,
                node_id=node_id,
                passed=False,
                details=f"failed to execute {self._nccl_test_binary}",
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(
                "intra_machine_timeout node=%s timeout=%s",
                node_id, timeout_seconds,
                exc_info=True,
            )
            return DiagnosticResult(
                diagnostic_type=self.diagnostic_type,
                node_id=node_id,
                passed=False,
                details=f"timed out after {timeout_seconds}s",
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        if process.returncode != 0:
            logger.warning(
                "intra_machine_nonzero_exit node=%s rc=%s stderr=%s",
                node_id, process.returncode, stderr[:500],
            )
            return DiagnosticResult(
                diagnostic_type=self.diagnostic_type,
                node_id=node_id,
                passed=False,
                details=f"exit code {process.returncode}: {stderr[:500]}",
            )

        bandwidth = parse_avg_bus_bandwidth(stdout)
        if bandwidth is None:
            logger.warning(
                "intra_machine_parse_failure node=%s output_len=%d",
                node_id, len(stdout),
            )
            return DiagnosticResult(
                diagnostic_type=self.diagnostic_type,
                node_id=node_id,
                passed=False,
                details="failed to parse bandwidth from output",
            )

        passed = bandwidth >= self._expected_bandwidth_gbps
        if passed:
            details = f"bandwidth {bandwidth:.2f} GB/s >= threshold {self._expected_bandwidth_gbps:.2f} GB/s"
        else:
            details = (
                f"bandwidth {bandwidth:.2f} GB/s < threshold {self._expected_bandwidth_gbps:.2f} GB/s"
            )

        logger.info(
            "intra_machine_result node=%s bandwidth=%.2f threshold=%.2f passed=%s",
            node_id, bandwidth, self._expected_bandwidth_gbps, passed,
        )
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=passed,
            details=details,
        )
