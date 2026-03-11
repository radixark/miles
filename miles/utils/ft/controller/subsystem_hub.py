from __future__ import annotations

import logging

from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.types import ScrapeTargetManagerProtocol
from miles.utils.ft.utils.box import Box

logger = logging.getLogger(__name__)


class SubsystemHub:
    """Central hub for subsystem-specific runtime data.

    Holds training rank roster, rollout manager handle, scrape targets,
    and future subsystem handles. FtController and actor access subsystem
    details through this hub instead of holding them directly.

    Adding a new subsystem (e.g. reward_model) only requires changes here
    and in the factory -- FtController remains untouched.
    """

    def __init__(
        self,
        *,
        training_rank_roster_box: Box[TrainingRankRoster],
        scrape_target_manager: ScrapeTargetManagerProtocol | None,
    ) -> None:
        self._training_rank_roster_box = training_rank_roster_box
        self._scrape_target_manager = scrape_target_manager
        self._rollout_manager_handle: object | None = None

    @property
    def training_rank_roster(self) -> TrainingRankRoster:
        return self._training_rank_roster_box.value

    @property
    def training_rank_roster_box(self) -> Box[TrainingRankRoster]:
        return self._training_rank_roster_box

    @property
    def rollout_manager_handle(self) -> object:
        assert self._rollout_manager_handle is not None, "Rollout handle not yet set"
        return self._rollout_manager_handle

    def set_rollout_handle(self, handle: object) -> None:
        self._rollout_manager_handle = handle
        logger.info("rollout_handle_set")

    def add_scrape_target(self, target_id: str, address: str) -> None:
        if self._scrape_target_manager is not None:
            self._scrape_target_manager.add_scrape_target(
                target_id=target_id,
                address=address,
            )

    def activate_run(self, run_id: str) -> None:
        """Reset training rank roster for a new run."""
        self._training_rank_roster_box.value.cleanup()
        self._training_rank_roster_box.value = TrainingRankRoster(
            run_id=run_id,
            scrape_target_manager=self._scrape_target_manager,
        )
        logger.info("run_activated run_id=%s", run_id)
