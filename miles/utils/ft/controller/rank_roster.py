from __future__ import annotations

import logging

from miles.utils.ft.protocols.metrics import ScrapeTargetManagerProtocol

logger = logging.getLogger(__name__)


class RankRoster:
    """Tracks rank placement for a single training run.

    Each run gets a fresh instance; the FtController creates a new
    RankRoster via ``_activate_run`` whenever a training job is
    (re)submitted.
    """

    def __init__(
        self,
        run_id: str | None = None,
        scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
    ) -> None:
        self.run_id = run_id
        self.expected_world_size: int | None = None
        self.rank_placement: dict[int, str] = {}
        self.rank_pids: dict[int, int] = {}
        self._scrape_target_manager = scrape_target_manager

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
                run_id, self.run_id,
            )
            return
        _validate_rank(rank=rank, world_size=world_size, node_id=node_id)
        self.expected_world_size = world_size
        self.rank_placement[rank] = node_id
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

    def get_rank_pids_for_node(self, node_id: str) -> dict[int, int]:
        return {
            rank: self.rank_pids[rank]
            for rank, nid in self.rank_placement.items()
            if nid == node_id
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
                self.run_id,
            )

    def cleanup(self) -> None:
        """Remove scrape targets. Call before discarding this registry."""
        if self._scrape_target_manager is not None:
            for old_rank in self.rank_placement:
                self._scrape_target_manager.remove_scrape_target(f"rank-{old_rank}")


def _validate_rank(*, rank: int, world_size: int, node_id: str) -> None:
    if not node_id:
        raise ValueError("node_id must be non-empty")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
