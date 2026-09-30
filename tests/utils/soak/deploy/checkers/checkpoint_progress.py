from collections.abc import Sequence

from tests.utils.deploy.hot_restart.evidence import HotRestartRecord
from tests.utils.soak.core.events import SoakEvent

SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 2
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1


def assert_checkpoints_advanced_between_takeovers(events: list[SoakEvent]) -> None:
    raise NotImplementedError


def assert_take_over_loss_within_save_interval(records: Sequence[HotRestartRecord]) -> None:
    raise NotImplementedError


def assert_take_overs_resumed_within_save_interval(dump_dir: str, *, records: Sequence[HotRestartRecord]) -> None:
    raise NotImplementedError
