import logging
from collections.abc import Callable

from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def run_write(self, write: Callable[[], None]) -> None:
        if self.is_errored:
            logger.warning(f"[P2P-Shared] skipping a queued write to rollout cell {self.cell_id}")
            return
        try:
            write()
        except Exception as error:
            self.mark_errored(error)
            raise
