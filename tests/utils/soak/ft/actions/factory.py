from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.actions.pod import DeletePodFaultForm
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend

FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL]

CELL_TYPE_OF_FT_COMPONENT: dict[str, str] = {"train": ACTOR_CELL_TYPE, "rollout": ROLLOUT_CELL_TYPE}


def create_cell_fault_forms(*, base_url: str, config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    inject_fault_forms = [
        InjectFaultForm(base_url=base_url, failure_mode=failure_mode) for failure_mode in FAILURE_MODES
    ]

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {ACTOR_CELL_TYPE: inject_fault_forms, ROLLOUT_CELL_TYPE: inject_fault_forms}
        case ClusterBackend.KUBERNETES:
            pod_form_kwargs: dict[str, str] = {"namespace": config.namespace, "run_id": config.run_id}
            delete_pod_form = DeletePodFaultForm(**pod_form_kwargs)
            return {
                ACTOR_CELL_TYPE: [*inject_fault_forms, delete_pod_form],
                ROLLOUT_CELL_TYPE: [delete_pod_form],
            }


def compute_mean_interval_seconds_of_kind(
    ft_components: tuple[str, ...], *, trainer_crash_interval_seconds: float, rollout_crash_interval_seconds: float
) -> dict[str, float]:
    return {
        CELL_TYPE_OF_FT_COMPONENT[component]: (
            trainer_crash_interval_seconds if component == "train" else rollout_crash_interval_seconds
        )
        for component in ft_components
    }
