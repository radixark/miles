from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector import FailureMode

ACTOR_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]
ROLLOUT_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL]

CELL_TYPE_OF_FT_COMPONENT: dict[str, str] = {"train": ACTOR_CELL_TYPE, "rollout": ROLLOUT_CELL_TYPE}


def create_cell_fault_forms(*, base_url: str, config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    raise NotImplementedError


def compute_mean_interval_seconds_of_kind(
    ft_components: tuple[str, ...], *, trainer_crash_interval_seconds: float, rollout_crash_interval_seconds: float
) -> dict[str, float]:
    raise NotImplementedError
