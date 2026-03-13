"""GpuNodeExecutor — pynvml extended checks + deterministic compute fingerprinting.

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

from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS
from miles.utils.ft.agents.diagnostics.base import BaseNodeExecutor
from miles.utils.ft.agents.diagnostics.utils.gpu_check_script import GpuCheckResult
from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.utils.subprocess import run_subprocess_with_timeout

logger = logging.getLogger(__name__)


class GpuNodeExecutor(BaseNodeExecutor):
    diagnostic_type = "gpu"

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult:
        logger.info("diagnostics: running GPU check: node_id=%s, timeout=%d", node_id, timeout_seconds)
        proc_result = await self._run_check_subprocess(
            node_id=node_id,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(proc_result, DiagnosticResult):
            logger.debug("diagnostics: GPU check subprocess returned early failure: node_id=%s", node_id)
            return proc_result

        stdout_bytes, stderr_bytes, returncode = proc_result

        if returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            logger.warning(
                "diagnostics: GPU check process failed: node_id=%s, returncode=%d, stderr=%s",
                node_id,
                returncode,
                stderr_text[:500],
            )
            return self._fail(node_id, f"gpu check process failed: {stderr_text[:500]}")

        gpu_results = self._parse_gpu_results(
            stdout_bytes=stdout_bytes,
            node_id=node_id,
        )
        if isinstance(gpu_results, DiagnosticResult):
            logger.debug("diagnostics: GPU check parse returned failure: node_id=%s", node_id)
            return gpu_results

        logger.debug("diagnostics: GPU check parsed results: node_id=%s, num_gpus=%d", node_id, len(gpu_results))
        return self._collect_results(gpu_results=gpu_results, node_id=node_id)

    async def _run_check_subprocess(
        self,
        node_id: str,
        timeout_seconds: int,
    ) -> tuple[bytes, bytes, int] | DiagnosticResult:
        cmd = [
            sys.executable,
            "-m",
            "miles.utils.ft.agents.diagnostics.utils.gpu_check_script",
        ]
        logger.info("diagnostics: launching GPU check subprocess: node_id=%s, cmd=%s", node_id, cmd)
        try:
            return await run_subprocess_with_timeout(
                cmd=cmd,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostics: GPU check subprocess timed out: node_id=%s, timeout=%d",
                node_id,
                timeout_seconds,
            )
            return self._fail(node_id, "gpu check timed out")
        except OSError:
            logger.warning("diagnostics: GPU check subprocess failed to launch: node_id=%s", node_id, exc_info=True)
            return self._fail(node_id, "gpu check process failed to launch")

    def _parse_gpu_results(
        self,
        stdout_bytes: bytes,
        node_id: str,
    ) -> list[GpuCheckResult] | DiagnosticResult:
        stdout_text = stdout_bytes.decode(errors="replace")
        try:
            raw = json.loads(stdout_text)
            return [GpuCheckResult(**item) for item in raw]
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "diagnostics: GPU check invalid JSON output: node_id=%s, output=%s",
                node_id,
                stdout_text[:200],
            )
            return self._fail(node_id, "invalid output from gpu check")

    def _collect_results(
        self,
        gpu_results: list[GpuCheckResult],
        node_id: str,
    ) -> DiagnosticResult:
        if not gpu_results:
            logger.warning("diagnostics: GPU check returned no results: node_id=%s", node_id)
            return self._fail(node_id, "gpu check returned no results")

        failed_gpus: list[str] = []
        compute_hashes: dict[str, str] = {}

        for r in gpu_results:
            idx = str(r.gpu_index)

            if not r.nvml_passed:
                failed_gpus.append(f"GPU {idx}: {r.details}")

            if r.compute_error:
                failed_gpus.append(f"GPU {idx}: compute error: {r.compute_error}")

            if r.compute_hash:
                compute_hashes[idx] = r.compute_hash

        metadata = {"compute_hashes": compute_hashes} if compute_hashes else None

        if failed_gpus:
            all_details = "; ".join(failed_gpus)
            logger.info(
                "diagnostics: GPU check failures: node_id=%s, num_failed=%d, details=%s",
                node_id,
                len(failed_gpus),
                all_details,
            )
            return self._fail(node_id, all_details, metadata=metadata)

        return self._pass(node_id, "all GPU checks passed", metadata=metadata)
