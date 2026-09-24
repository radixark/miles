from collections.abc import Sequence
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze


def assert_generations_recorded_their_steps(
    templates: Sequence[str], *, schedule: Sequence[ScheduledFreeze], num_rollouts: int
) -> None:
    directories = [Path(template).parent for template in templates]
    stray = sorted(set(directories[0].parent.iterdir()) - set(directories))
    assert not stray, (
        f"{directories[0].parent} holds {[one.name for one in stray]} beside the {len(templates)} generation "
        f"directories the take-overs relaunched with, so some executor wrote where nothing relaunched it"
    )

    for generation, (directory, rollout_ids) in enumerate(
        zip(directories, _compute_rollout_ids_of_generation(schedule, num_rollouts=num_rollouts), strict=True)
    ):
        recorded = sorted(int(one.stem) for one in directory.glob("*.pt"))
        assert recorded == rollout_ids, (
            f"generation {generation} of the rollout executor recorded the rollouts {recorded} under {directory}, "
            f"and the freeze schedule has that generation generate exactly {rollout_ids}: the relaunched executor "
            f"did not run with the arguments it was relaunched with, or generated steps it should not have"
        )

    print("every generation of the rollout executor recorded the steps it generated")


def _compute_rollout_ids_of_generation(schedule: Sequence[ScheduledFreeze], *, num_rollouts: int) -> list[list[int]]:
    windows: list[list[int]] = []
    start = 0
    for scheduled in schedule:
        windows.append(list(range(start, scheduled.frozen_rollout_id + 1)))
        start = 0 if scheduled.saved_iteration is None else scheduled.saved_iteration + 1
    windows.append(list(range(start, num_rollouts)))
    return windows
