from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import tail_started_at


def assert_end_state_complete(
    events: list[SoakEvent], *, expected_count_of_kind: dict[str, int], observation_ends_with_sut: bool
) -> None:
    observations = [event for event in events if isinstance(event, SoakObservationEvent)]
    if observation_ends_with_sut:
        while observations and observations[-1].targets is None:
            observations.pop()

    assert observations, "Soak ended without any observation"
    observation = observations[-1]
    assert observation.targets is not None, "Soak ended on a failed observation"
    assert not observation.errors, f"Soak ended on an observation with errors: {observation.errors}"
    if observation_ends_with_sut:
        assert observation.timestamp >= tail_started_at(
            events
        ), "Soak's last successful observation was taken before its last fault"

    for kind, expected_count in expected_count_of_kind.items():
        targets = [target for target in observation.targets if target.kind == kind]
        assert (
            len(targets) == expected_count
        ), f"Soak ended with {len(targets)} {kind} targets, expected {expected_count}"
        assert all(
            target.alive and target.ready for target in targets
        ), f"Soak ended with a {kind} target that is not alive and ready"
