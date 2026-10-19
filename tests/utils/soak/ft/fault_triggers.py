from pathlib import Path

from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakForms
from tests.utils.soak.core.views import read_training_events
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.checkers.fault_hook_dispatch import assert_hook_dispatches, assert_p2p_receiver_failures
from tests.utils.soak.ft.checkers.trainer_peer_progress import assert_trainer_peers_progress
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE, FaultTrigger

DEFAULT_FAULT_TRIGGERS: frozenset[FaultTrigger] = frozenset({FaultTrigger.TIMER, FaultTrigger.HOOK})
HOOK_TRAIN_ARGS: str = "--update-weights-timeout 600 "


def resolve(
    requested: list[FaultTrigger] | None, *, has_real_rollout: bool, trainer_ft: bool
) -> frozenset[FaultTrigger]:
    triggers = frozenset(requested) if requested else DEFAULT_FAULT_TRIGGERS
    if not has_real_rollout:
        assert requested is None or FaultTrigger.HOOK not in triggers, (
            "Without rollout engines no weight update ever reaches a trainer fault hook, so hook-triggered faults "
            "could only expire"
        )
        triggers -= {FaultTrigger.HOOK}
    elif not trainer_ft:
        assert requested is None or FaultTrigger.HOOK not in triggers, (
            "Without trainer fault tolerance the api server exposes no trainer cell to carry a fault hook, so "
            "hook-triggered faults are not supported"
        )
        triggers -= {FaultTrigger.HOOK}
    assert triggers, "At least one fault trigger is needed"
    return triggers


def compute_test_name_suffix(triggers: frozenset[FaultTrigger]) -> str:
    return "" if triggers == DEFAULT_FAULT_TRIGGERS else "_" + "_".join(sorted(triggers))


def compute_hook_train_args(triggers: frozenset[FaultTrigger]) -> str:
    return HOOK_TRAIN_ARGS if FaultTrigger.HOOK in triggers else ""


def assert_hook_evidence(*, forms: SoakForms, events: list[SoakEvent], dump_dir: str | Path) -> None:
    hooked_kinds = {
        kind
        for kind, kind_forms in forms.items()
        if any(isinstance(form, InjectFaultForm) and form.hook_name is not None for form in kind_forms)
    }
    if not hooked_kinds:
        return
    training_events = read_training_events(events, dump_dir=dump_dir)
    assert_hook_dispatches(events, training_events=training_events)
    if ROLLOUT_CELL_TYPE in hooked_kinds:
        assert_p2p_receiver_failures(events, training_events=training_events)
    if ACTOR_CELL_TYPE in hooked_kinds:
        assert_trainer_peers_progress(events, training_events=training_events)
