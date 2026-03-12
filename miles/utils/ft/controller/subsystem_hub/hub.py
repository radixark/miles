from __future__ import annotations

import logging
from collections.abc import Iterable

from miles.utils.ft.controller.subsystem_hub.training_rank_roster import TrainingRankRoster
from miles.utils.ft.utils.box import Box

logger = logging.getLogger(__name__)


class SubsystemHub:
    """Central hub for subsystem-specific runtime data.

    Holds training rank roster, rollout manager handle, and future
    subsystem handles. FtController and actor access subsystem details
    through this hub instead of holding them directly.

    Adding a new subsystem (e.g. reward_model) only requires changes here
    and in the factory -- FtController remains untouched.
    """

    def __init__(
        self,
        *,
        training_rank_roster_box: Box[TrainingRankRoster | None],
    ) -> None:
        self._training_rank_roster_box = training_rank_roster_box
        self._rollout_manager_handle: object | None = None
        self._rollout_node_ids: dict[str, frozenset[str]] = {}

    @property
    def training_rank_roster(self) -> TrainingRankRoster:
        value = self._training_rank_roster_box.value
        assert value is not None, "TrainingRankRoster not yet initialized (call _activate_run first)"
        return value

    @property
    def training_rank_roster_box(self) -> Box[TrainingRankRoster | None]:
        return self._training_rank_roster_box

    @property
    def rollout_manager_handle(self) -> object:
        assert self._rollout_manager_handle is not None, "Rollout handle not yet set"
        return self._rollout_manager_handle

    def set_rollout_handle(self, handle: object) -> None:
        self._rollout_manager_handle = handle
        logger.info("rollout_handle_set")

    def set_rollout_node_ids(self, cell_id: str, node_ids: Iterable[str]) -> None:
        self._rollout_node_ids[cell_id] = frozenset(node_ids)
        logger.info("rollout_node_ids_set cell_id=%s nodes=%s", cell_id, sorted(node_ids))

    def get_rollout_node_ids(self, cell_id: str) -> frozenset[str]:
        return self._rollout_node_ids.get(cell_id, frozenset())
