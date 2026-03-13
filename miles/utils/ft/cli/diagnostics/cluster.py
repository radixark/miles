from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

import typer

from miles.utils.ft.adapters.types import ClusterExecutorProtocol
from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.cli.diagnostics.output import exit_with_results, print_results, validate_check_names

logger = logging.getLogger(__name__)


def cluster(
    checks: Annotated[list[str] | None, typer.Argument(help="Checks to run (default: all)")] = None,
    ray_address: Annotated[str, typer.Option(help="Ray cluster address")] = "auto",
    timeout: Annotated[int, typer.Option(help="Per-check timeout in seconds")] = 180,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
) -> None:
    """Run diagnostic checks across a Ray cluster."""
    import ray

    from miles.utils.ft.adapters.impl.ray.node_discovery import get_alive_gpu_nodes
    from miles.utils.ft.controller.diagnostics.executors import build_all_cluster_executors

    logger.info("cli: cluster diagnostics ray_address=%s", ray_address)
    ray.init(address=ray_address)

    try:
        nodes = get_alive_gpu_nodes()
        logger.info("cli: cluster diagnostics discovered gpu_nodes=%d", len(nodes))
        registry = build_all_cluster_executors()

        selected = checks or list(registry.keys())
        validate_check_names(selected, available=registry.keys())

        async def _run() -> list[DiagnosticResult]:
            async with _managed_agents(nodes) as node_agents:
                return await _run_cluster_checks(registry, node_agents, selected, timeout)

        results = asyncio.run(_run())
    finally:
        ray.shutdown()

    print_results(results, json_output=json_output, node_id="cluster")
    exit_with_results(results)


async def _run_cluster_checks(
    registry: dict[str, ClusterExecutorProtocol],
    node_agents: dict[str, Any],
    checks: list[str],
    timeout: int,
) -> list[DiagnosticResult]:
    logger.info("cli: running cluster checks=%s, node_count=%d, timeout=%d", checks, len(node_agents), timeout)
    all_results: list[DiagnosticResult] = []
    for name in checks:
        try:
            bad_nodes = await registry[name].execute(
                node_agents=node_agents,
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

    node_agents = _deploy_agents(nodes)
    logger.info("cli: deployed diagnostic agents count=%d", len(node_agents))
    try:
        yield node_agents
    finally:
        logger.info("cli: cleaning up diagnostic agents count=%d", len(node_agents))
        for actor in node_agents.values():
            ray.kill(actor)


def _deploy_agents(
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles.utils.ft.adapters.types import DIAGNOSTIC_TIMEOUT_SECONDS
    from miles.utils.ft.agents.diagnostics.dispatcher import NodeDiagnosticDispatcher
    from miles.utils.ft.factories.node_agent import build_all_diagnostics

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

    node_agents: dict[str, Any] = {}
    for node in nodes:
        node_id = node["NodeID"]
        num_gpus = int(node["Resources"]["GPU"])
        actor = _DiagnosticAgent.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(node_id=node_id, num_gpus=num_gpus)
        node_agents[node_id] = actor

    return node_agents
