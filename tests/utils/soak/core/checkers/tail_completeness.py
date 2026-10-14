from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import admission_closed, is_normal_step, tail_started_at, trainer_step_ends


def assert_tail_complete(events: list[SoakEvent]) -> None:
    closed = admission_closed(events)
    assert closed is not None, "Soak injection admission never closed"

    recovered_at = tail_started_at(events)
    progress = trainer_step_ends(events)
    before_close = max((event.rollout_id for event in progress if event.timestamp <= closed.timestamp), default=-1)
    assert any(
        event.timestamp > recovered_at and event.rollout_id > before_close and is_normal_step(event)
        for event in progress
    ), "Soak tail has no successful training progress after admission closed and faults recovered"
