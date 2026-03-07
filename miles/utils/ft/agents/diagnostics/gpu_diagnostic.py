"""GpuDiagnostic — pynvml extended checks + deterministic compute fingerprinting.

Launches ``gpu_check_script`` as a subprocess so that pynvml init/shutdown
and torch computation never block the NodeAgent event loop.

Per-GPU results include nvml health status and a SHA256 hash of a
deterministic computation.  The hashes are forwarded via
``DiagnosticResult.metadata["compute_hashes"]`` so the orchestrator can
compare across nodes and identify outliers (bitwise alignment test).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

from miles.utils.ft.agents.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.utils.subprocess import run_subprocess_with_timeout

logger = logging.getLogger(__name__)


class GpuDiagnostic(BaseDiagnostic):
    diagnostic_type = "gpu"

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        proc_result = await self._run_check_subprocess(
            node_id=node_id, timeout_seconds=timeout_seconds,
        )
        if isinstance(proc_result, DiagnosticResult):
            return proc_result

        stdout_bytes, stderr_bytes, returncode = proc_result

        if returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            logger.warning(
                "gpu_check_process_failed node_id=%s returncode=%d stderr=%s",
                node_id, returncode, stderr_text[:500],
            )
            return self._fail(node_id, f"gpu check process failed: {stderr_text[:500]}")

        gpu_results = self._parse_gpu_results(
            stdout_bytes=stdout_bytes, node_id=node_id,
        )
        if isinstance(gpu_results, DiagnosticResult):
            return gpu_results

        return self._collect_results(gpu_results=gpu_results, node_id=node_id)

    async def _run_check_subprocess(
        self, node_id: str, timeout_seconds: int,
    ) -> tuple[bytes, bytes, int] | DiagnosticResult:
        cmd = [
            sys.executable, "-m",
            "miles.utils.ft.agents.diagnostics.gpu_check_script",
        ]
        try:
            return await run_subprocess_with_timeout(
                cmd=cmd, timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "gpu_check_timeout node_id=%s timeout=%d",
                node_id, timeout_seconds,
            )
            return self._fail(node_id, "gpu check timed out")
        except OSError:
            logger.warning("gpu_check_launch_failed node_id=%s", node_id, exc_info=True)
            return self._fail(node_id, "gpu check process failed to launch")

    def _parse_gpu_results(
        self, stdout_bytes: bytes, node_id: str,
    ) -> list[dict[str, object]] | DiagnosticResult:
        stdout_text = stdout_bytes.decode(errors="replace")
        try:
            return json.loads(stdout_text)
        except json.JSONDecodeError:
            logger.warning(
                "gpu_check_invalid_json node_id=%s output=%s",
                node_id, stdout_text[:200],
            )
            return self._fail(node_id, "invalid output from gpu check")

    def _collect_results(
        self, gpu_results: list[dict[str, object]], node_id: str,
    ) -> DiagnosticResult:
        if not gpu_results:
            logger.warning("gpu_check_empty_results node_id=%s", node_id)
            return self._fail(node_id, "gpu check returned no results")

        failed_gpus: list[str] = []
        compute_hashes: dict[str, str] = {}

        for gpu_result in gpu_results:
            gpu_index = str(gpu_result.get("gpu_index", "?"))

            if not gpu_result.get("nvml_passed", False):
                details = gpu_result.get("details", "unknown failure")
                failed_gpus.append(f"GPU {gpu_index}: {details}")

            compute_error = gpu_result.get("compute_error", "")
            if compute_error:
                failed_gpus.append(f"GPU {gpu_index}: compute error: {compute_error}")

            compute_hash = gpu_result.get("compute_hash", "")
            if compute_hash:
                compute_hashes[gpu_index] = compute_hash

        metadata = {"compute_hashes": compute_hashes} if compute_hashes else None

        if failed_gpus:
            all_details = "; ".join(failed_gpus)
            logger.info(
                "gpu_check_failures node_id=%s failures=%s",
                node_id, all_details,
            )
            return self._fail(node_id, all_details, metadata=metadata)

        return self._pass(node_id, "all GPU checks passed", metadata=metadata)
