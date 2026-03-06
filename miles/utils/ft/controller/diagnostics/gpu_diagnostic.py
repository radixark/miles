"""GpuDiagnostic — pynvml extended checks + GPU matmul correctness verification.

Launches ``gpu_check_script`` as a subprocess so that pynvml init/shutdown
and torch computation never block the NodeAgent event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models.diagnostics import DiagnosticResult

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

        return self._collect_failures(gpu_results=gpu_results, node_id=node_id)

    async def _run_check_subprocess(
        self, node_id: str, timeout_seconds: int,
    ) -> tuple[bytes, bytes, int] | DiagnosticResult:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m",
                "miles.utils.ft.controller.diagnostics.gpu_check_script",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "gpu_check_timeout node_id=%s timeout=%d",
                node_id, timeout_seconds,
            )
            try:
                process.kill()
                await process.wait()
            except Exception:
                logger.warning("gpu_check_kill_failed node_id=%s", node_id, exc_info=True)
            return self._fail(node_id, "gpu check timed out")
        except Exception:
            logger.warning("gpu_check_launch_failed node_id=%s", node_id, exc_info=True)
            return self._fail(node_id, "gpu check process failed to launch")

        assert process.returncode is not None
        return stdout_bytes, stderr_bytes, process.returncode

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

    def _collect_failures(
        self, gpu_results: list[dict[str, object]], node_id: str,
    ) -> DiagnosticResult:
        if not gpu_results:
            logger.warning("gpu_check_empty_results node_id=%s", node_id)
            return self._fail(node_id, "gpu check returned no results")

        failed_gpus: list[str] = []
        for gpu_result in gpu_results:
            if not gpu_result.get("passed", False):
                gpu_index = gpu_result.get("gpu_index", "?")
                details = gpu_result.get("details", "unknown failure")
                failed_gpus.append(f"GPU {gpu_index}: {details}")

        if failed_gpus:
            all_details = "; ".join(failed_gpus)
            logger.info(
                "gpu_check_failures node_id=%s failures=%s",
                node_id, all_details,
            )
            return self._fail(node_id, all_details)

        return self._pass(node_id, "all GPU checks passed")
