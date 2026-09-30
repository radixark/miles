from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakForms, find_form
from tests.utils.soak.core.views import compute_num_injections, compute_successful_form_names, project_actions
from tests.utils.soak.ft.actions.factory import CELL_TYPE_OF_FT_COMPONENT
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

MIN_SOAK_INJECTIONS: int = 2


def assert_healing(
    ft_components: tuple[str, ...],
    *,
    events: list[SoakEvent],
    forms: SoakForms,
    context: str,
) -> None:
    if "train" in ft_components:
        assert_min_injections(events, kind=ACTOR_CELL_TYPE, context=f"{context} trainer cells")
        assert_injections_recovered(events, cell_type=ACTOR_CELL_TYPE, forms=forms)

    if "rollout" in ft_components:
        assert_min_injections(events, kind=ROLLOUT_CELL_TYPE, context=f"{context} rollout cells")
        assert_injections_recovered(events, cell_type=ROLLOUT_CELL_TYPE, forms=forms)

    _assert_enabled_fault_forms_worked(events, forms=forms, ft_components=ft_components)


def assert_min_injections(events: list[SoakEvent], *, kind: str, context: str) -> None:
    num_successful_injections = compute_num_injections(events, kind=kind)
    assert num_successful_injections >= MIN_SOAK_INJECTIONS, (
        f"Soak proved too little in {context}: the fault injector reported only "
        f"{num_successful_injections} successful injection(s), need >= {MIN_SOAK_INJECTIONS} "
        f"to exercise fault recovery more than once"
    )


def assert_injections_recovered(events: list[SoakEvent], *, cell_type: str, forms: SoakForms) -> None:
    actions = [
        action
        for action in project_actions(events).values()
        if action.applied is not None
        and action.requested.request.target.kind == cell_type
        and find_form(forms, kind=cell_type, name=action.requested.request.form_name).harms_target
    ]
    unresolved = {
        action.requested.request.target.identity: action.requested.request.request_id
        for action in actions
        if not find_form(forms, kind=cell_type, name=action.requested.request.form_name).is_recovered(
            action=action, events=events
        )
    }
    assert not unresolved, f"{cell_type} recovery witness failed: unresolved requests {unresolved}"
    print(f"{cell_type} recovery witness passed: {len(actions)} injections recovered")


def _assert_enabled_fault_forms_worked(
    events: list[SoakEvent], *, forms: SoakForms, ft_components: tuple[str, ...]
) -> None:
    never_worked: list[tuple[str, str]] = []
    for component in ft_components:
        cell_type = CELL_TYPE_OF_FT_COMPONENT[component]
        if (cell_forms := forms.get(cell_type)) is None:
            continue
        worked = compute_successful_form_names(events, kind=cell_type)
        never_worked += [(cell_type, form.name) for form in cell_forms if form.name not in worked]

    assert not never_worked, f"fault forms this soak enabled but never injected successfully: {sorted(never_worked)}"
