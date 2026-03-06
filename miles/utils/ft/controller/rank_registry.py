from __future__ import annotations

import logging

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.protocols.agents import NodeAgentProtocol
from miles.utils.ft.protocols.metrics import ScrapeTargetManagerProtocol

logger = logging.getLogger(__name__)


class RankRegistry:
    """Tracks registered agents, ranks, and the active training run."""

    def __init__(
        self,
        mini_wandb: MiniWandb,
        scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
    ) -> None:
        self._mini_wandb = mini_wandb
        self._scrape_target_manager = scrape_target_manager

        self.agents: dict[str, NodeAgentProtocol] = {}
        self.active_run_id: str | None = None
        self.expected_world_size: int | None = None
        self.rank_placement: dict[int, str] = {}
        self.rank_pids: dict[int, int] = {}

    @property
    def mini_wandb(self) -> MiniWandb:
        return self._mini_wandb

    def register_node_agent(self, node_id: str, agent: NodeAgentProtocol) -> None:
        self.agents[node_id] = agent
        logger.info("agent_registered node_id=%s", node_id)

    def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int | None = None,
    ) -> None:
        self._validate_rank(run_id=run_id, rank=rank, world_size=world_size, node_id=node_id)
        self._switch_run_if_needed(run_id=run_id)
        self._record_rank(
            run_id=run_id, rank=rank, world_size=world_size,
            node_id=node_id, exporter_address=exporter_address, pid=pid,
        )

    @staticmethod
    def _validate_rank(
        *, run_id: str, rank: int, world_size: int, node_id: str,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    def _switch_run_if_needed(self, *, run_id: str) -> None:
        if run_id == self.active_run_id:
            return

        logger.info(
            "new_run_registered run_id=%s previous_run_id=%s",
            run_id, self.active_run_id,
        )
        self.active_run_id = run_id
        self.expected_world_size = None
        self._mini_wandb.set_active_run_id(run_id)
        self._mini_wandb.clear()
        self._remove_old_scrape_targets()
        self.rank_placement = {}
        self.rank_pids = {}

    def _record_rank(
        self,
        *,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int | None,
    ) -> None:
        self.expected_world_size = world_size
        self.rank_placement[rank] = node_id
        if pid is not None:
            self.rank_pids[rank] = pid

        logger.info(
            "rank_registered run_id=%s rank=%d world_size=%d node_id=%s",
            run_id, rank, world_size, node_id,
        )

        if self._scrape_target_manager is not None:
            self._scrape_target_manager.add_scrape_target(
                target_id=f"rank-{rank}",
                address=exporter_address,
            )

    def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        self._mini_wandb.log_step(
            run_id=run_id,
            step=step,
            metrics=metrics,
        )

    def get_rank_pids_for_node(self, node_id: str) -> dict[int, int]:
        return {
            rank: self.rank_pids[rank]
            for rank, nid in self.rank_placement.items()
            if nid == node_id and rank in self.rank_pids
        }

    def warn_if_incomplete(self) -> None:
        if (
            self.expected_world_size is not None
            and len(self.rank_placement) < self.expected_world_size
        ):
            logger.warning(
                "incomplete_rank_registration registered=%d expected=%d run_id=%s",
                len(self.rank_placement),
                self.expected_world_size,
                self.active_run_id,
            )

    def _remove_old_scrape_targets(self) -> None:
        if self._scrape_target_manager is not None:
            for old_rank in self.rank_placement:
                self._scrape_target_manager.remove_scrape_target(f"rank-{old_rank}")
