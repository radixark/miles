"""Pairwise NCCL diagnostic executor.

Pairs nodes in a round-robin ring, runs a pairwise diagnostic on each
pair simultaneously, then cross-compares failure counts to isolate
the bad node(s).
"""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from miles.utils.ft.agents.diagnostics.executors.nccl_pairwise import DEFAULT_NCCL_MASTER_PORT
from miles.utils.ft.controller.diagnostics.utils import RPC_TIMEOUT_BUFFER_SECONDS
from miles.utils.ft.protocols.agents import NodeAgentProtocol

logger = logging.getLogger(__name__)


class _PairResult(NamedTuple):
    master_id: str
    worker_id: str
    passed: bool


class PairwiseClusterExecutor:
    """Pair-based pairwise diagnostic with cross-comparison.

    Implements the ClusterExecutorProtocol protocol.
    """

    def __init__(
        self,
        diagnostic_type: str,
        base_port: int = DEFAULT_NCCL_MASTER_PORT,
    ) -> None:
        self._diagnostic_type = diagnostic_type
        self._base_port = base_port

    async def execute(
        self,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]:
        sorted_ids = sorted(agents.keys())

        if len(sorted_ids) < 2:
            logger.info("pairwise_skip — fewer than 2 nodes")
            return []

        pairs = [
            (sorted_ids[i], sorted_ids[(i + 1) % len(sorted_ids)])
            for i in range(len(sorted_ids))
        ]
        logger.info("pairwise_step_start pairs=%s", pairs)

        tasks = []
        for pair_index, (master_id, worker_id) in enumerate(pairs):
            port = self._base_port + pair_index
            tasks.append(
                self._run_single_pair(
                    agents=agents,
                    master_id=master_id,
                    worker_id=worker_id,
                    master_addr=master_id,
                    port=port,
                    timeout_seconds=timeout_seconds,
                )
            )

        results = await asyncio.gather(*tasks)
        return _cross_compare(node_ids=sorted_ids, pair_results=list(results))

    async def _run_single_pair(
        self,
        agents: dict[str, NodeAgentProtocol],
        master_id: str,
        worker_id: str,
        master_addr: str,
        port: int,
        timeout_seconds: int,
    ) -> _PairResult:
        master_agent = agents.get(master_id)
        worker_agent = agents.get(worker_id)

        if master_agent is None or worker_agent is None:
            logger.warning(
                "pairwise_pair_skip_no_agent master=%s(%s) worker=%s(%s)",
                master_id,
                "ok" if master_agent else "missing",
                worker_id,
                "ok" if worker_agent else "missing",
            )
            return _PairResult(master_id=master_id, worker_id=worker_id, passed=False)

        try:
            master_result, worker_result = await asyncio.wait_for(
                asyncio.gather(
                    master_agent.run_diagnostic(
                        diagnostic_type=self._diagnostic_type,
                        timeout_seconds=timeout_seconds,
                        master_addr=master_addr,
                        master_port=port,
                    ),
                    worker_agent.run_diagnostic(
                        diagnostic_type=self._diagnostic_type,
                        timeout_seconds=timeout_seconds,
                        master_addr=master_addr,
                        master_port=port,
                    ),
                ),
                timeout=timeout_seconds + RPC_TIMEOUT_BUFFER_SECONDS,
            )
            passed = master_result.passed and worker_result.passed
        except asyncio.TimeoutError:
            logger.warning(
                "pairwise_pair_rpc_timeout master=%s worker=%s timeout=%d",
                master_id,
                worker_id,
                timeout_seconds,
            )
            passed = False
        except Exception:
            logger.warning(
                "pairwise_pair_failed master=%s worker=%s",
                master_id,
                worker_id,
                exc_info=True,
            )
            passed = False

        logger.info(
            "pairwise_pair_result master=%s worker=%s passed=%s",
            master_id,
            worker_id,
            passed,
        )
        return _PairResult(master_id=master_id, worker_id=worker_id, passed=passed)


def _cross_compare(
    node_ids: list[str],
    pair_results: list[_PairResult],
) -> list[str]:
    """Cross-compare pair results to isolate bad nodes.

    Algorithm:
    1. Count failures per node across all pairs.
    2. If no failures -> return empty (all healthy).
    3. Find nodes with the highest failure count.
    4. If ALL nodes share the same (non-zero) failure count -> cannot
       localize -> return empty (NOTIFY_HUMAN).
    5. Otherwise -> return nodes with the highest failure count.
    """
    failure_count: dict[str, int] = {nid: 0 for nid in node_ids}
    for result in pair_results:
        if not result.passed:
            failure_count[result.master_id] += 1
            failure_count[result.worker_id] += 1

    counts = list(failure_count.values())
    max_count = max(counts)
    if max_count == 0:
        return []

    if min(counts) == max_count:
        logger.warning("pairwise_all_failed — cannot localize bad node")
        return []

    return sorted(nid for nid, count in failure_count.items() if count == max_count)
