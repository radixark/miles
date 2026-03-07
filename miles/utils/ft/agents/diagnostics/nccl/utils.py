from __future__ import annotations

import asyncio
import logging
import re
from functools import partial
from typing import Callable

from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.utils.subprocess import run_subprocess_with_timeout

logger = logging.getLogger(__name__)

_AVG_BUS_BW_PATTERN = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)")
_BUSBW_COLUMN_INDEX = 7

# nccl-tests CLI defaults: sweep from 1 MB to 1 GB, doubling each step
_NCCL_TEST_MIN_BYTES = "1M"
_NCCL_TEST_MAX_BYTES = "1G"
_NCCL_TEST_SIZE_FACTOR = "2"


def build_nccl_test_cmd(binary: str, num_gpus: int) -> list[str]:
    return [
        binary,
        "-b", _NCCL_TEST_MIN_BYTES,
        "-e", _NCCL_TEST_MAX_BYTES,
        "-f", _NCCL_TEST_SIZE_FACTOR,
        "-g", str(num_gpus),
    ]


def parse_avg_bus_bandwidth(output: str) -> float | None:
    """Parse average bus bandwidth (GB/s) from nccl-tests text output.

    Primary path: look for the ``# Avg bus bandwidth`` summary line.
    Fallback: parse the last data row and extract the busbw column
    (column index 7, out-of-place, 0-indexed).
    """
    match = _AVG_BUS_BW_PATTERN.search(output)
    if match:
        return float(match.group(1))

    last_data_row: list[str] | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) > _BUSBW_COLUMN_INDEX:
            try:
                float(parts[0])
                last_data_row = parts
            except ValueError:
                continue

    if last_data_row is not None:
        try:
            return float(last_data_row[_BUSBW_COLUMN_INDEX])
        except (IndexError, ValueError):
            return None

    return None


async def run_nccl_test(
    cmd: list[str],
    node_id: str,
    diagnostic_type: str,
    expected_bandwidth_gbps: float,
    timeout_seconds: int,
    log_prefix: str,
    env: dict[str, str] | None = None,
) -> DiagnosticResult:
    """Run an nccl-tests binary and return a DiagnosticResult.

    Shared subprocess lifecycle used by both IntraMachineCommDiagnostic
    and InterMachineCommDiagnostic.
    """
    fail: Callable[[str], DiagnosticResult] = partial(
        DiagnosticResult.fail_result, diagnostic_type=diagnostic_type, node_id=node_id,
    )

    try:
        stdout_bytes, stderr_bytes, returncode = await run_subprocess_with_timeout(
            cmd=cmd, timeout_seconds=timeout_seconds, env=env,
        )
    except OSError:
        logger.warning(
            "%s_exec_failed node=%s binary=%s",
            log_prefix, node_id, cmd[0],
            exc_info=True,
        )
        return fail(details=f"failed to execute {cmd[0]}")
    except asyncio.TimeoutError:
        logger.warning(
            "%s_timeout node=%s timeout=%s",
            log_prefix, node_id, timeout_seconds,
            exc_info=True,
        )
        return fail(details=f"timed out after {timeout_seconds}s")

    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")

    return _interpret_nccl_output(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        node_id=node_id,
        diagnostic_type=diagnostic_type,
        expected_bandwidth_gbps=expected_bandwidth_gbps,
        log_prefix=log_prefix,
    )


def _interpret_nccl_output(
    *,
    stdout: str,
    stderr: str,
    returncode: int | None,
    node_id: str,
    diagnostic_type: str,
    expected_bandwidth_gbps: float,
    log_prefix: str,
) -> DiagnosticResult:
    fail: Callable[[str], DiagnosticResult] = partial(
        DiagnosticResult.fail_result, diagnostic_type=diagnostic_type, node_id=node_id,
    )

    if returncode != 0:
        logger.warning(
            "%s_nonzero_exit node=%s rc=%s stderr=%s",
            log_prefix,
            node_id,
            returncode,
            stderr[:500],
        )
        return fail(details=f"exit code {returncode}: {stderr[:500]}")

    bandwidth = parse_avg_bus_bandwidth(stdout)
    if bandwidth is None:
        logger.warning(
            "%s_parse_failure node=%s output_len=%d",
            log_prefix,
            node_id,
            len(stdout),
        )
        return fail(details="failed to parse bandwidth from output")

    passed = bandwidth >= expected_bandwidth_gbps
    if passed:
        details = f"bandwidth {bandwidth:.2f} GB/s >= threshold {expected_bandwidth_gbps:.2f} GB/s"
    else:
        details = f"bandwidth {bandwidth:.2f} GB/s < threshold {expected_bandwidth_gbps:.2f} GB/s"

    logger.info(
        "%s_result node=%s bandwidth=%.2f threshold=%.2f passed=%s",
        log_prefix,
        node_id,
        bandwidth,
        expected_bandwidth_gbps,
        passed,
    )
    return DiagnosticResult(
        diagnostic_type=diagnostic_type,
        node_id=node_id,
        passed=passed,
        details=details,
    )
