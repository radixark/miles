from tests.utils.soak.core.events import SoakEvent


def assert_end_state_complete(events: list[SoakEvent], *, expected_count_of_kind: dict[str, int]) -> None:
    raise NotImplementedError
