from datetime import datetime

MIN_CRASHED_ROLLOUTS: int = 2


def _compute_crashed_rollouts(
    *, injected_at: list[datetime], rollout_completions: list[tuple[int, datetime]]
) -> set[int]:
    return {
        max((rollout_id for rollout_id, finished_at in rollout_completions if finished_at <= at), default=-1) + 1
        for at in injected_at
    }
