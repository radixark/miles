from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import latest_observation


def assert_end_state_complete(events: list[SoakEvent], *, expected_count_of_kind: dict[str, int]) -> None:
    observation = latest_observation(events)
    assert observation is not None, "Soak ended without any observation"
    assert observation.targets is not None, "Soak ended on a failed observation"
    assert not observation.errors, f"Soak ended on an observation with errors: {observation.errors}"

    for kind, expected_count in expected_count_of_kind.items():
        targets = [target for target in observation.targets if target.kind == kind]
        assert (
            len(targets) == expected_count
        ), f"Soak ended with {len(targets)} {kind} targets, expected {expected_count}"
        assert all(
            target.alive and target.ready for target in targets
        ), f"Soak ended with a {kind} target that is not alive and ready"
