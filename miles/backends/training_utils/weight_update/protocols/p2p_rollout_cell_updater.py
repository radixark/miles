from collections.abc import Callable


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater:
    def __init__(self, rollout_engine_ind: int) -> None:
        self.rollout_engine_ind = rollout_engine_ind

    def run_write(self, write: Callable[[], None]) -> None:
        write()
