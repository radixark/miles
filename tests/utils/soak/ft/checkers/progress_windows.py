from tests.utils.soak.core.events import SoakEvent

MIN_FAULT_PROGRESS_WINDOWS: int = 2


def assert_faults_span_progress_windows(events: list[SoakEvent], *, dump_dir: str) -> None:
    raise NotImplementedError
