"""Inter-machine NCCL diagnostic orchestration.

Pairs nodes in a round-robin ring, runs all_gather_perf on each
pair simultaneously, then cross-compares failure counts to isolate
the bad node(s).
"""
from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from miles.utils.ft.models._diagnostics import DiagnosticResult
from miles.utils.ft.protocols.agents import NodeAgentProtocol

logger = logging.getLogger(__name__)

_BASE_PORT = 29500
_RPC_TIMEOUT_BUFFER_SECONDS = 30


class PairResult(NamedTuple):
    master_id: str
    worker_id: str
    passed: bool


class InterMachineOrchestrator:
    def __init__(
        self,
        agents: dict[str, NodeAgentProtocol],
        node_addresses: dict[str, str] | None = None,
        base_port: int = _BASE_PORT,
    ) -> None:
        self._agents = agents
        self._node_addresses = node_addresses
        self._base_port = base_port

    async def run(
        self,
        node_ids: list[str],
        timeout_seconds: int,
    ) -> list[str]:
        """Run inter-machine pair tests and return bad node IDs."""
        sorted_ids = sorted(node_ids)

        if len(sorted_ids) < 2:
            logger.info("inter_machine_skip — fewer than 2 nodes")
            return []

        pairs = [
            (sorted_ids[i], sorted_ids[(i + 1) % len(sorted_ids)])
            for i in range(len(sorted_ids))
        ]
        logger.info("inter_machine_step_start pairs=%s", pairs)

        tasks = []
        for pair_index, (master_id, worker_id) in enumerate(pairs):
            port = self._base_port + pair_index
            master_addr = self._resolve_address(master_id)
            tasks.append(self._run_single_pair(
                master_id=master_id,
                worker_id=worker_id,
                master_addr=master_addr,
                port=port,
                timeout_seconds=timeout_seconds,
            ))

        results = await asyncio.gather(*tasks)

        return cross_compare(node_ids=sorted_ids, pair_results=list(results))

    async def _run_single_pair(
        self,
        master_id: str,
        worker_id: str,
        master_addr: str,
        port: int,
        timeout_seconds: int,
    ) -> PairResult:
        master_agent = self._agents.get(master_id)
        worker_agent = self._agents.get(worker_id)

        if master_agent is None or worker_agent is None:
            logger.warning(
                "inter_machine_pair_skip_no_agent master=%s(%s) worker=%s(%s)",
                master_id, "ok" if master_agent else "missing",
                worker_id, "ok" if worker_agent else "missing",
            )
            return PairResult(master_id=master_id, worker_id=worker_id, passed=False)

        try:
            master_result, worker_result = await asyncio.wait_for(
                asyncio.gather(
                    master_agent.run_diagnostic(
                        diagnostic_type="inter_machine",
                        timeout_seconds=timeout_seconds,
                        master_addr=master_addr,
                        master_port=port,
                    ),
                    worker_agent.run_diagnostic(
                        diagnostic_type="inter_machine",
                        timeout_seconds=timeout_seconds,
                        master_addr=master_addr,
                        master_port=port,
                    ),
                ),
                timeout=timeout_seconds + _RPC_TIMEOUT_BUFFER_SECONDS,
            )
            passed = master_result.passed and worker_result.passed
        except asyncio.TimeoutError:
            logger.warning(
                "inter_machine_pair_rpc_timeout master=%s worker=%s timeout=%d",
                master_id, worker_id, timeout_seconds,
            )
            passed = False
        except Exception:
            logger.warning(
                "inter_machine_pair_failed master=%s worker=%s",
                master_id, worker_id,
                exc_info=True,
            )
            passed = False

        logger.info(
            "inter_machine_pair_result master=%s worker=%s passed=%s",
            master_id, worker_id, passed,
        )
        return PairResult(master_id=master_id, worker_id=worker_id, passed=passed)

    def _resolve_address(self, node_id: str) -> str:
        if self._node_addresses and node_id in self._node_addresses:
            return self._node_addresses[node_id]
        return node_id


def cross_compare(
    node_ids: list[str],
    pair_results: list[PairResult],
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
        logger.warning("inter_machine_all_failed — cannot localize bad node")
        return []

    return sorted(
        nid for nid, count in failure_count.items() if count == max_count
    )
