from tests.utils.soak.core.utils import compute_base_url
from tests.utils.soak.ft.actions.base import BaseCellFaultForm, CellFaultForms
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.actions.pod import DeletePodFaultForm, ExecSigkillFaultForm, ExecSigstopFaultForm
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE, FaultTrigger

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector.actions.process import (
    DeadlockThreadAction,
    ExitProcessAction,
    KillProcessAction,
    SegfaultProcessAction,
    StopProcessAction,
)
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import FaultHookName
from miles.utils.workers.types import ClusterBackend

ACTOR_FAULT_ACTIONS: list[FaultAction] = [KillProcessAction(), ExitProcessAction(), SegfaultProcessAction()]
ROLLOUT_FAULT_ACTIONS: list[FaultAction] = [KillProcessAction()]

HOOK_FAULT_LIFETIME_SECONDS: float = 300.0
HOOK_FAULT_MAX_DELAY_MS: float = 1000.0
ACTOR_HOOK_FAULT_ACTIONS: list[FaultAction] = [KillProcessAction(), StopProcessAction(), DeadlockThreadAction()]
ROLLOUT_HOOK_FAULT_ACTIONS: list[FaultAction] = [KillProcessAction()]
HOOK_FAULT_NAMES: list[FaultHookName] = [
    FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER,
    FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND,
]

CELL_TYPE_OF_FT_COMPONENT: dict[str, str] = {"train": ACTOR_CELL_TYPE, "rollout": ROLLOUT_CELL_TYPE}


def create_cell_fault_forms(
    config: command_utils.ExecuteTrainConfig, *, triggers: frozenset[FaultTrigger]
) -> CellFaultForms:
    forms: CellFaultForms = {ACTOR_CELL_TYPE: [], ROLLOUT_CELL_TYPE: []}
    for trigger in sorted(triggers):
        for kind, kind_forms in _CREATE_FORMS_OF_TRIGGER[trigger](config).items():
            forms[kind] += kind_forms
    return forms


def compute_mean_interval_seconds_of_kind(
    ft_components: tuple[str, ...], *, trainer_crash_interval_seconds: float, rollout_crash_interval_seconds: float
) -> dict[str, float]:
    return {
        CELL_TYPE_OF_FT_COMPONENT[component]: (
            trainer_crash_interval_seconds if component == "train" else rollout_crash_interval_seconds
        )
        for component in ft_components
    }


def _create_timer_forms(config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    base_url = compute_base_url(config)
    actor_inject_fault_forms = _inject_fault_forms(base_url=base_url, actions=ACTOR_FAULT_ACTIONS)

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {
                ACTOR_CELL_TYPE: actor_inject_fault_forms,
                ROLLOUT_CELL_TYPE: _inject_fault_forms(base_url=base_url, actions=ROLLOUT_FAULT_ACTIONS),
            }
        case ClusterBackend.KUBERNETES:
            pod_form_kwargs: dict[str, str] = {"namespace": config.namespace, "run_id": config.run_id}
            delete_pod_form = DeletePodFaultForm(**pod_form_kwargs)
            return {
                ACTOR_CELL_TYPE: [*actor_inject_fault_forms, delete_pod_form],
                ROLLOUT_CELL_TYPE: [
                    ExecSigkillFaultForm(**pod_form_kwargs),
                    ExecSigstopFaultForm(**pod_form_kwargs),
                    delete_pod_form,
                ],
            }


def _create_hook_forms(config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    base_url = compute_base_url(config)
    actor_hook_forms: list[BaseCellFaultForm] = [
        InjectFaultForm(
            base_url=base_url,
            action=action,
            hook_name=hook_name,
            lifetime_seconds=HOOK_FAULT_LIFETIME_SECONDS,
            max_delay_ms=0 if isinstance(action, DeadlockThreadAction) else HOOK_FAULT_MAX_DELAY_MS,
        )
        for hook_name in HOOK_FAULT_NAMES
        for action in ACTOR_HOOK_FAULT_ACTIONS
    ]

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {
                ACTOR_CELL_TYPE: actor_hook_forms,
                ROLLOUT_CELL_TYPE: [
                    InjectFaultForm(
                        base_url=base_url,
                        action=action,
                        hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND,
                        lifetime_seconds=HOOK_FAULT_LIFETIME_SECONDS,
                        max_delay_ms=HOOK_FAULT_MAX_DELAY_MS,
                        through_trainer_hook=True,
                    )
                    for action in ROLLOUT_HOOK_FAULT_ACTIONS
                ],
            }
        case ClusterBackend.KUBERNETES:
            return {ACTOR_CELL_TYPE: actor_hook_forms, ROLLOUT_CELL_TYPE: []}


def _inject_fault_forms(*, base_url: str, actions: list[FaultAction]) -> list[BaseCellFaultForm]:
    return [InjectFaultForm(base_url=base_url, action=action) for action in actions]


_CREATE_FORMS_OF_TRIGGER = {FaultTrigger.TIMER: _create_timer_forms, FaultTrigger.HOOK: _create_hook_forms}
