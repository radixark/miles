from pathlib import Path

from tests.utils.soak.fault_forms import ACTOR_CELL_TYPE, CELL_TYPE_OF_FT_COMPONENT, ROLLOUT_CELL_TYPE, CellFaultForms
from tests.utils.soak.recovery import compute_recovery_episodes
from tests.utils.soak.state import SoakEvent, SoakObservation, event_source
from tests.utils.soak.views import (
    compute_forms_drawn_without_success,
    compute_num_injections,
    compute_successful_form_names,
)

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.test_utils.reconfigure_assertions import assert_min_soak_injections, load_reconfigure_events


def assert_healing(
    ft_components: tuple[str, ...],
    *,
    events: list[SoakEvent],
    forms: CellFaultForms,
    event_dir: Path,
    context: str,
) -> None:
    event_dir = event_source(events, name="training_events", fallback=event_dir)

    _assert_drawn_fault_forms_worked(events)

    if "train" in ft_components:
        assert_min_soak_injections(
            compute_num_injections(events, cell_type=ACTOR_CELL_TYPE), context=f"{context} trainer cells"
        )
        assert_trainer_injections_healed(events, event_dir=event_dir)

    if "rollout" in ft_components:
        assert_min_soak_injections(
            compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE), context=f"{context} rollout cells"
        )
        assert_rollout_cells_served_after_injection(events)

    _assert_enabled_fault_forms_worked(events, forms=forms, ft_components=ft_components)


def _assert_drawn_fault_forms_worked(events: list[SoakEvent]) -> None:
    never_worked = compute_forms_drawn_without_success(events)
    assert not never_worked, f"Fault forms drawn but never once successful: {never_worked}"


def _assert_enabled_fault_forms_worked(
    events: list[SoakEvent], *, forms: CellFaultForms, ft_components: tuple[str, ...]
) -> None:
    never_worked: list[tuple[str, str]] = []
    for component in ft_components:
        cell_type = CELL_TYPE_OF_FT_COMPONENT[component]
        if (cell_forms := forms.get(cell_type)) is None:
            continue
        worked = compute_successful_form_names(events, cell_type=cell_type)
        never_worked += [(cell_type, form.name) for form in cell_forms if form.name not in worked]

    assert not never_worked, f"fault forms this soak enabled but never injected successfully: {sorted(never_worked)}"


def assert_trainer_injections_healed(events: list[SoakEvent], *, event_dir: Path) -> None:
    event_dir = event_source(events, name="training_events", fallback=event_dir)
    assert event_dir.is_dir(), f"Event directory {event_dir} does not exist or is not a directory"
    reconfigurations = [
        step
        for observation in events
        if isinstance(observation, SoakObservation)
        for step in observation.training_events
        if isinstance(step, CellReconfigureEvent)
    ]
    _assert_recovery_episodes(
        events,
        cell_type=ACTOR_CELL_TYPE,
        reconfigurations=[*reconfigurations, *load_reconfigure_events(event_dir)],
    )


def assert_rollout_cells_served_after_injection(events: list[SoakEvent]) -> None:
    _assert_recovery_episodes(events, cell_type=ROLLOUT_CELL_TYPE)


def _assert_recovery_episodes(
    events: list[SoakEvent],
    *,
    cell_type: str,
    reconfigurations: list[CellReconfigureEvent] | None = None,
) -> None:
    episodes = [
        episode
        for episode in compute_recovery_episodes(events, reconfigurations=reconfigurations)
        if episode.cell_type == cell_type
    ]
    expected = compute_num_injections(events, cell_type=cell_type)
    assert (
        sum(len(episode.request_ids) for episode in episodes) == expected
    ), f"{cell_type} injections lack request-bound incarnation evidence"
    unresolved = {
        episode.cell_id: episode.request_ids for episode in episodes if episode.recovered_incarnation is None
    }
    assert not unresolved, f"{cell_type} recovery witness failed: unresolved requests {unresolved}"
    print(f"{cell_type} recovery witness passed: {expected} injections in {len(episodes)} recovery episodes")
