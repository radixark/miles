from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import NodeAgentProtocol
from miles.utils.ft.agents.types import DiagnosticResult, UnknownDiagnosticError

logger = logging.getLogger(__name__)

RPC_TIMEOUT_BUFFER_SECONDS = 30


async def call_agent_diagnostic(
    agent: NodeAgentProtocol,
    node_id: str,
    diagnostic_type: str,
    timeout_seconds: int,
    **kwargs: object,
) -> DiagnosticResult:
    try:
        return await asyncio.wait_for(
            agent.run_diagnostic(
                diagnostic_type,
                timeout_seconds=timeout_seconds,
                **kwargs,
            ),
            timeout=timeout_seconds + RPC_TIMEOUT_BUFFER_SECONDS,
        )

    except asyncio.TimeoutError:
        logger.warning(
            "diagnostic_agent_rpc_timeout node=%s type=%s timeout=%d",
            node_id,
            diagnostic_type,
            timeout_seconds,
        )
        return DiagnosticResult.fail_result(
            diagnostic_type=diagnostic_type,
            node_id=node_id,
            details=f"agent RPC timed out after {timeout_seconds + RPC_TIMEOUT_BUFFER_SECONDS}s",
        )

    except UnknownDiagnosticError:
        logger.error(
            "diagnostic_type_not_registered node=%s type=%s — "
            "this is a pipeline configuration error, treating as fail",
            node_id,
            diagnostic_type,
            exc_info=True,
        )
        return DiagnosticResult.fail_result(
            diagnostic_type=diagnostic_type,
            node_id=node_id,
            details=f"config error: diagnostic type '{diagnostic_type}' not registered on node",
        )

    except Exception:
        logger.warning(
            "diagnostic_agent_call_failed node=%s type=%s",
            node_id,
            diagnostic_type,
            exc_info=True,
        )
        return DiagnosticResult.fail_result(
            diagnostic_type=diagnostic_type,
            node_id=node_id,
            details="agent call raised exception",
        )


async def gather_diagnostic_results(
    diagnostic_type: str,
    node_agents: dict[str, NodeAgentProtocol],
    timeout_seconds: int,
) -> dict[str, DiagnosticResult]:
    """Run one diagnostic on all agents in parallel and return the raw results."""
    node_ids = list(node_agents.keys())
    logger.info(
        "diagnostic_step_start type=%s nodes=%s",
        diagnostic_type,
        node_ids,
    )

    raw_results = await asyncio.gather(
        *(
            call_agent_diagnostic(
                agent=node_agents[node_id],
                node_id=node_id,
                diagnostic_type=diagnostic_type,
                timeout_seconds=timeout_seconds,
            )
            for node_id in node_ids
        )
    )
    return dict(zip(node_ids, raw_results, strict=True))


def partition_results(
    results: dict[str, DiagnosticResult],
    diagnostic_type: str,
) -> list[str]:
    """Return node IDs that failed the diagnostic."""
    bad_node_ids: list[str] = []

    for node_id, result in results.items():
        if not result.passed:
            bad_node_ids.append(node_id)
            logger.info(
                "diagnostic_node_failed type=%s node=%s details=%s",
                diagnostic_type,
                node_id,
                result.details,
            )

    return bad_node_ids
