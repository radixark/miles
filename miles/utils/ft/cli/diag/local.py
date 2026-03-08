from __future__ import annotations

import asyncio
import logging
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.ft.agents.collectors.disk import DiskCollector
from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.intra_machine import IntraMachineNodeExecutor
from miles.utils.ft.cli.diag.output import exit_with_results, print_results
from miles.utils.ft.controller.detectors.checks.gpu.checks import check_gpu_faults
from miles.utils.ft.controller.detectors.checks.hardware import _check_disk_fault, _check_majority_nic_down
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import NodeExecutorProtocol

logger = logging.getLogger(__name__)

LOCAL_CHECK_NAMES = ["gpu", "intra_machine", "disk", "network", "xid"]


def _build_local_registry(
    disk_mounts: list[Path],
    num_gpus: int,
    xid_since_minutes: int,
) -> dict[str, NodeExecutorProtocol]:
    return {
        "gpu": GpuNodeExecutor(),
        "intra_machine": IntraMachineNodeExecutor(num_gpus=num_gpus),
        "disk": CollectorBasedNodeExecutor(
            diagnostic_type="disk",
            collector=DiskCollector(disk_mounts=disk_mounts),
            check_fn=_check_disk_fault,
        ),
        "network": CollectorBasedNodeExecutor(
            diagnostic_type="network",
            collector=NetworkCollector(),
            check_fn=_check_majority_nic_down,
        ),
        "xid": CollectorBasedNodeExecutor(
            diagnostic_type="xid",
            collector=KmsgCollector(
                since=datetime.now(timezone.utc) - timedelta(minutes=xid_since_minutes),
            ),
            check_fn=check_gpu_faults,
        ),
    }


async def _run_checks(
    registry: dict[str, NodeExecutorProtocol],
    checks: list[str],
    node_id: str,
    timeout: int,
) -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    for name in checks:
        executor = registry[name]
        try:
            result = await asyncio.wait_for(
                executor.run(node_id=node_id, timeout_seconds=timeout),
                timeout=timeout + 30,
            )
        except Exception:
            logger.error("check %s failed with exception", name, exc_info=True)
            result = DiagnosticResult.fail_result(
                diagnostic_type=name,
                node_id=node_id,
                details="exception during check (see logs)",
            )
        results.append(result)
    return results


def local(
    checks: Annotated[list[str] | None, typer.Argument(help="Checks to run (default: all)")] = None,
    timeout: Annotated[int, typer.Option(help="Per-check timeout in seconds")] = 120,
    disk_mounts: Annotated[str, typer.Option(help="Comma-separated mount paths")] = "/",
    num_gpus: Annotated[int, typer.Option(help="Number of GPUs on this node")] = 8,
    xid_since_minutes: Annotated[int, typer.Option(help="Look back N minutes for XID errors")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
) -> None:
    """Run diagnostic checks on the local node."""
    selected = checks or LOCAL_CHECK_NAMES
    unknown = set(selected) - set(LOCAL_CHECK_NAMES)
    if unknown:
        typer.echo(f"Unknown checks: {', '.join(sorted(unknown))}", err=True)
        raise typer.Exit(code=1)

    mount_paths = [Path(m.strip()) for m in disk_mounts.split(",")]
    registry = _build_local_registry(
        disk_mounts=mount_paths,
        num_gpus=num_gpus,
        xid_since_minutes=xid_since_minutes,
    )

    node_id = socket.gethostname()
    results = asyncio.run(_run_checks(registry, selected, node_id, timeout))

    print_results(results, json_output=json_output, node_id=node_id)
    exit_with_results(results)
