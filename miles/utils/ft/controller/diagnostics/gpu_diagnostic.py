"""GpuDiagnostic — pynvml extended checks + GPU matmul correctness verification.

Launches ``_gpu_check_script`` as a subprocess so that pynvml init/shutdown
and torch computation never block the NodeAgent event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.models import DiagnosticResult

logger = logging.getLogger(__name__)


class GpuDiagnostic(BaseDiagnostic):
    diagnostic_type = "gpu"

    def _result(
        self, *, node_id: str, passed: bool, details: str,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=passed,
            details=details,
        )

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m",
                "miles.utils.ft.controller.diagnostics._gpu_check_script",
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
            return self._result(
                node_id=node_id, passed=False, details="gpu check timed out",
            )
        except Exception:
            logger.warning("gpu_check_launch_failed node_id=%s", node_id, exc_info=True)
            return self._result(
                node_id=node_id, passed=False,
                details="gpu check process failed to launch",
            )

        if process.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            logger.warning(
                "gpu_check_process_failed node_id=%s returncode=%d stderr=%s",
                node_id, process.returncode, stderr_text[:500],
            )
            return self._result(
                node_id=node_id, passed=False,
                details=f"gpu check process failed: {stderr_text[:500]}",
            )

        stdout_text = stdout_bytes.decode(errors="replace")
        try:
            gpu_results = json.loads(stdout_text)
        except json.JSONDecodeError:
            logger.warning(
                "gpu_check_invalid_json node_id=%s output=%s",
                node_id, stdout_text[:200],
            )
            return self._result(
                node_id=node_id, passed=False,
                details="invalid output from gpu check",
            )

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
            return self._result(
                node_id=node_id, passed=False, details=all_details,
            )

        return self._result(
            node_id=node_id, passed=True, details="all GPU checks passed",
        )
