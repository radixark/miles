from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer

from miles.utils.ft.cli.diag.output import exit_with_results, print_results
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

CLUSTER_CHECK_NAMES = ["gpu", "intra_machine", "inter_machine"]


async def _run_cluster_checks(
    checks: list[str],
    timeout: int,
) -> list[DiagnosticResult]:
    from miles.utils.ft.platform.ray_wrappers.standalone_diagnostic import (
        run_gpu_diagnostics,
        run_inter_machine_diagnostics,
        run_intra_machine_diagnostics,
    )

    all_results: list[DiagnosticResult] = []

    if "gpu" in checks:
        gpu_results, outlier_ids = await run_gpu_diagnostics(timeout_seconds=timeout)
        all_results.extend(gpu_results)
        if outlier_ids:
            all_results.append(
                DiagnosticResult.fail_result(
                    diagnostic_type="gpu_hash_comparison",
                    node_id="cluster",
                    details=f"outlier nodes: {', '.join(outlier_ids)}",
                )
            )

    if "intra_machine" in checks:
        intra_results = await run_intra_machine_diagnostics(timeout_seconds=timeout)
        all_results.extend(intra_results)

    if "inter_machine" in checks:
        bad_nodes = await run_inter_machine_diagnostics(timeout_seconds=timeout)
        if bad_nodes:
            all_results.append(
                DiagnosticResult.fail_result(
                    diagnostic_type="inter_machine",
                    node_id="cluster",
                    details=f"bad nodes: {', '.join(bad_nodes)}",
                )
            )
        else:
            all_results.append(
                DiagnosticResult.pass_result(
                    diagnostic_type="inter_machine",
                    node_id="cluster",
                    details="all inter-machine NCCL checks passed",
                )
            )

    return all_results


def cluster(
    checks: Annotated[list[str] | None, typer.Argument(help="Checks to run (default: all)")] = None,
    ray_address: Annotated[str, typer.Option(help="Ray cluster address")] = "auto",
    timeout: Annotated[int, typer.Option(help="Per-check timeout in seconds")] = 180,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
) -> None:
    """Run diagnostic checks across a Ray cluster."""
    import ray

    selected = checks or CLUSTER_CHECK_NAMES
    unknown = set(selected) - set(CLUSTER_CHECK_NAMES)
    if unknown:
        typer.echo(f"Unknown checks: {', '.join(sorted(unknown))}", err=True)
        raise typer.Exit(code=1)

    ray.init(address=ray_address)

    try:
        results = asyncio.run(_run_cluster_checks(selected, timeout))
    finally:
        ray.shutdown()

    print_results(results, json_output=json_output, node_id="cluster")
    exit_with_results(results)
