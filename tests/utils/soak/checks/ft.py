from collections import Counter
from pathlib import Path

from tests.utils.soak.entrypoint import FaultInjectorHandle
from tests.utils.soak.fault_forms import ACTOR_CELL_TYPE, CELL_TYPE_OF_FT_COMPONENT, ROLLOUT_CELL_TYPE
from tests.utils.soak.views import (
    compute_cells_not_serving_after_injection,
    compute_forms_drawn_without_success,
    compute_injected_cell_names,
    compute_num_injections,
    compute_states_of_cell_name,
    compute_successful_form_names,
)

from miles.utils.test_utils.reconfigure_assertions import (
    assert_min_soak_injections,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)
from miles.utils.workers.naming import parse_cell_id


def assert_healing(
    ft_components: tuple[str, ...], *, injector: FaultInjectorHandle, event_dir: Path, context: str
) -> None:
    events = injector.event_log.events

    _assert_drawn_fault_forms_worked(injector)

    if "train" in ft_components:
        assert_soak_reconfigure_events(
            event_dir, num_successful_injections=compute_num_injections(events, cell_type=ACTOR_CELL_TYPE)
        )
        assert_trainer_injections_healed(injector, event_dir=event_dir)

    if "rollout" in ft_components:
        assert_min_soak_injections(
            compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE), context=f"{context} rollout cells"
        )
        assert_rollout_cells_served_after_injection(injector)

    _assert_enabled_fault_forms_worked(injector, ft_components=ft_components)


def _assert_drawn_fault_forms_worked(injector: FaultInjectorHandle) -> None:
    never_worked = compute_forms_drawn_without_success(injector.event_log.events)
    assert not never_worked, f"Fault forms drawn but never once successful: {never_worked}"


def _assert_enabled_fault_forms_worked(injector: FaultInjectorHandle, *, ft_components: tuple[str, ...]) -> None:
    events = injector.event_log.events
    never_worked: list[tuple[str, str]] = []
    for component in ft_components:
        cell_type = CELL_TYPE_OF_FT_COMPONENT[component]
        if (forms := injector.cell_fault_forms.get(cell_type)) is None:
            continue
        worked = compute_successful_form_names(events, cell_type=cell_type)
        never_worked += [(cell_type, form.name) for form in forms if form.name not in worked]

    assert not never_worked, f"fault forms this soak enabled but never injected successfully: {sorted(never_worked)}"


def assert_trainer_injections_healed(injector: FaultInjectorHandle, *, event_dir: Path) -> None:
    injected: Counter[int] = Counter(
        parse_cell_id(name).cell_index
        for name in compute_injected_cell_names(injector.event_log.events, cell_type=ACTOR_CELL_TYPE)
    )
    healed: Counter[int] = Counter(
        cell_index for event in load_reconfigure_events(event_dir) for cell_index in event.healed_cell_indices
    )
    debt: Counter[int] = injected - healed

    assert not debt, (
        f"Trainer recovery witness failed: cell index -> accepted injection(s) never healed {dict(debt)} when "
        f"training ended (injected {dict(injected)}, healed {dict(healed)} across the events in {event_dir})"
    )

    print(
        f"Trainer recovery witness assertion passed: every one of {sum(injected.values())} accepted injection(s) "
        f"is paired with a healing of the same cell ({dict(healed)})"
    )


def assert_rollout_cells_served_after_injection(injector: FaultInjectorHandle) -> None:
    events = injector.event_log.events
    num_injections: int = compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE)
    offenders: dict[str, list[str]] = compute_cells_not_serving_after_injection(events, cell_type=ROLLOUT_CELL_TYPE)
    observed: dict[str, list[str]] = {
        name: [state.value for state in states] for name, states in compute_states_of_cell_name(events).items()
    }

    assert not offenders, (
        f"Rollout recovery witness failed: {sorted(offenders)} were never observed healthy and Serving on a "
        f"reading fresh enough to outlast the stale-status window after their last accepted injection, so the "
        f"run may have ended with a permanently missing replica ({num_injections} accepted injection(s); "
        f"observed states: {observed})"
    )

    print(
        f"Rollout recovery witness assertion passed: every injected cell was observed healthy and Serving on a "
        f"fresh reading after its last of {num_injections} accepted injection(s)"
    )
