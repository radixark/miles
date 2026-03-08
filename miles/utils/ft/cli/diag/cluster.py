from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

import typer

from miles.utils.ft.cli.diag.output import exit_with_results, print_results, validate_check_names
from miles.utils.ft.models.diagnostic import DiagnosticResult
from miles.utils.ft.protocols.agents import ClusterExecutorProtocol

logger = logging.getLogger(__name__)


def cluster(
    checks: Annotated[list[str] | None, typer.Argument(help="Checks to run (default: all)")] = None,
    ray_address: Annotated[str, typer.Option(help="Ray cluster address")] = "auto",
    timeout: Annotated[int, typer.Option(help="Per-check timeout in seconds")] = 180,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
) -> None:
    """Run diagnostic checks across a Ray cluster."""
    import ray

    from miles.utils.ft.controller.diagnostics.executors import build_all_cluster_executors
    from miles.utils.ft.platform.ray_wrappers.node_discovery import get_alive_gpu_nodes

    ray.init(address=ray_address)

    try:
        nodes = get_alive_gpu_nodes()
        registry = build_all_cluster_executors()

        selected = checks or list(registry.keys())
        validate_check_names(selected, available=registry.keys())

        async def _run() -> list[DiagnosticResult]:
            async with _managed_agents(nodes) as agents:
                return await _run_cluster_checks(registry, agents, selected, timeout)

        results = asyncio.run(_run())
    finally:
        ray.shutdown()

    print_results(results, json_output=json_output, node_id="cluster")
    exit_with_results(results)


async def _run_cluster_checks(
    registry: dict[str, ClusterExecutorProtocol],
    agents: dict[str, Any],
    checks: list[str],
    timeout: int,
) -> list[DiagnosticResult]:
    all_results: list[DiagnosticResult] = []
    for name in checks:
        try:
            bad_nodes = await registry[name].execute(
                agents=agents,
                timeout_seconds=timeout,
            )
            if bad_nodes:
                all_results.append(
                    DiagnosticResult.fail_result(
                        diagnostic_type=name,
                        node_id="cluster",
                        details=f"bad nodes: {', '.join(bad_nodes)}",
                    )
                )
            else:
                all_results.append(
                    DiagnosticResult.pass_result(
                        diagnostic_type=name,
                        node_id="cluster",
                        details="all nodes healthy",
                    )
                )
        except Exception:
            logger.error("cluster check %s failed with exception", name, exc_info=True)
            all_results.append(
                DiagnosticResult.fail_result(
                    diagnostic_type=name,
                    node_id="cluster",
                    details="exception during check (see logs)",
                )
            )
    return all_results


@contextlib.asynccontextmanager
async def _managed_agents(
    nodes: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    import ray

    agents = _deploy_agents(nodes)
    try:
        yield agents
    finally:
        for actor in agents.values():
            ray.kill(actor)


def _deploy_agents(
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles.utils.ft.agents.diagnostics.dispatcher import NodeDiagnosticDispatcher
    from miles.utils.ft.models.diagnostic import DiagnosticResult
    from miles.utils.ft.platform.node_agent_factory import build_all_diagnostics
    from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS

    @ray.remote(num_gpus=0)
    class _DiagnosticAgent:
        def __init__(self, node_id: str, num_gpus: int) -> None:
            self._dispatcher = NodeDiagnosticDispatcher(
                node_id=node_id,
                diagnostics=build_all_diagnostics(num_gpus=num_gpus),
            )

        async def run_diagnostic(
            self,
            diagnostic_type: str,
            timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
            **kwargs: object,
        ) -> DiagnosticResult:
            return await self._dispatcher.run_diagnostic(
                diagnostic_type=diagnostic_type,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

    agents: dict[str, Any] = {}
    for node in nodes:
        node_id = node["NodeID"]
        num_gpus = int(node["Resources"]["GPU"])
        actor = _DiagnosticAgent.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(node_id=node_id, num_gpus=num_gpus)
        agents[node_id] = actor

    return agents
