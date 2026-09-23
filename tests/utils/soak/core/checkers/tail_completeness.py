from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import admission_closed, is_normal_step, project_actions, trainer_step_ends


def assert_tail_complete(events: list[SoakEvent]) -> None:
    closed = admission_closed(events)
    assert closed is not None, "Soak injection admission never closed"

    actions = project_actions(events)
    recovered_at = max(
        [closed.timestamp, *(action.applied.timestamp for action in actions.values() if action.applied is not None)]
    )
    progress = trainer_step_ends(events)
    before_close = max((event.rollout_id for event in progress if event.timestamp <= closed.timestamp), default=-1)
    assert any(
        event.timestamp > recovered_at and event.rollout_id > before_close and is_normal_step(event)
        for event in progress
    ), "Soak tail has no successful training progress after admission closed and faults recovered"
