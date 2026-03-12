from __future__ import annotations

import logging

from miles.utils.ft.controller.types import ScrapeTargetManagerProtocol

logger = logging.getLogger(__name__)


class TrainingRankRoster:
    """Tracks rank placement for a single training run."""

    def __init__(
        self,
        run_id: str,
        scrape_target_manager: ScrapeTargetManagerProtocol,
    ) -> None:
        self.run_id = run_id
        self.expected_world_size: int | None = None
        self.rank_placement: dict[int, str] = {}
        self.rank_pids: dict[int, int] = {}
        self._scrape_target_manager: ScrapeTargetManagerProtocol = scrape_target_manager

    def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int,
    ) -> None:
        if run_id != self.run_id:
            logger.warning(
                "rejected_register_training_rank run_id=%s expected=%s",
                run_id,
                self.run_id,
            )
            return
        _validate_rank(rank=rank, world_size=world_size, node_id=node_id)

        if self.expected_world_size is None:
            self.expected_world_size = world_size
        elif world_size != self.expected_world_size:
            logger.error(
                "rejected_inconsistent_world_size run_id=%s rank=%d reported_world_size=%d expected_world_size=%d node_id=%s",
                run_id,
                rank,
                world_size,
                self.expected_world_size,
                node_id,
            )
            return

        self.rank_placement[rank] = node_id
        self.rank_pids[rank] = pid
        logger.info(
            "rank_registered run_id=%s rank=%d world_size=%d node_id=%s",
            run_id,
            rank,
            world_size,
            node_id,
        )
        self._scrape_target_manager.add_scrape_target(
            target_id=f"rank-{rank}",
            address=exporter_address,
        )

    def get_rank_pids_for_node(self, node_id: str) -> dict[int, int]:
        return {rank: self.rank_pids[rank] for rank, nid in self.rank_placement.items() if nid == node_id}

    def warn_if_incomplete(self) -> None:
        if self.expected_world_size is not None and len(self.rank_placement) < self.expected_world_size:
            logger.warning(
                "incomplete_rank_registration registered=%d expected=%d run_id=%s",
                len(self.rank_placement),
                self.expected_world_size,
                self.run_id,
            )

    def cleanup(self) -> None:
        """Remove scrape targets. Call before discarding this registry."""
        for old_rank in self.rank_placement:
            self._scrape_target_manager.remove_scrape_target(f"rank-{old_rank}")


def _validate_rank(*, rank: int, world_size: int, node_id: str) -> None:
    if not node_id:
        raise ValueError("node_id must be non-empty")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
