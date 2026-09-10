import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.fast.utils.soak.utils import RUNNING_NOT_SERVING, SERVING, cell, staged
from tests.utils.soak import state
from tests.utils.soak.utils import evidence_directory


def test_closing_admission_races_with_requests_but_preserves_observation_and_results(tmp_path: Path) -> None:
    """Closure atomically rejects later requests while accepted actions and observations can finish."""
    log = state.EventLog()
    path = tmp_path / "events.jsonl"
    log.persist_to(path)
    request = state.SoakActionRequest(target=cell("c", healthy=True), form_name="sigkill", harms_cell=True)
    barrier = threading.Barrier(2, timeout=5)

    def request_action() -> bool:
        barrier.wait()
        return log.note_action_requested(request)

    def close() -> None:
        barrier.wait()
        log.close_admission()

    with ThreadPoolExecutor(max_workers=2) as pool:
        requested = pool.submit(request_action)
        closed = pool.submit(close)
        accepted = requested.result(timeout=10)
        closed.result(timeout=10)

    log.close_admission()
    assert not log.note_action_requested(request)
    if accepted:
        log.note_action_result(state.SoakActionResultEvent(request_id=request.request_id, returned=False))
    log.note_observation(state.SoakObservation(cells=[]))
    log.finish()

    events = state.read_events(path)
    closure = next(index for index, event in enumerate(events) if isinstance(event, state.SoakAdmissionClosedEvent))
    assert sum(isinstance(event, state.SoakAdmissionClosedEvent) for event in events) == 1
    assert sum(isinstance(event, state.SoakActionRequestedEvent) for event in events) == int(accepted)
    assert not any(isinstance(event, state.SoakActionRequestedEvent) for event in events[closure:])
    assert isinstance(events[-2], state.SoakObservation)


def test_cell_is_alive_true_only_when_healthy_condition_is_true() -> None:
    """cell_is_alive reflects the Healthy condition status."""
    assert state.cell_is_alive(cell("c", healthy=True))
    assert not state.cell_is_alive(cell("c", healthy=False))


def test_cell_is_alive_false_when_no_healthy_condition_present() -> None:
    """A cell with no Healthy condition is not considered alive."""
    assert not state.cell_is_alive({"metadata": {"name": "c"}, "status": {"conditions": []}})


def test_a_running_cell_that_is_not_in_the_router_is_not_serving() -> None:
    """The api server renders PendingWeights and Serving alike, so the Serving condition must split them."""
    assert state.compute_observed_cell_state(staged("c", RUNNING_NOT_SERVING)) is RUNNING_NOT_SERVING
    assert state.compute_observed_cell_state(staged("c", SERVING)) is SERVING


def test_archived_generations_survive_training_directory_deletion_and_detect_later_corruption(tmp_path: Path) -> None:
    """Final evidence preserves both active and discarded generations after training data is cleared."""
    dump = tmp_path / "run"
    source = dump / "events"
    discarded = dump / ".trash_20260911_000000_abcd"
    source.mkdir(parents=True)
    discarded.mkdir()
    (source / "step.jsonl").write_text("active generation\n")
    (discarded / "step.jsonl").write_text("discarded generation\n")
    path = evidence_directory(dump) / "events.jsonl"
    log = state.EventLog()
    log.persist_to(path)
    log.note_context(
        state.SoakRunContextEvent(details={"helm_values": ("infra.yaml",)}, sources={"training_events": source})
    )
    log.observe([staged("c", SERVING)])
    log.finish()
    shutil.rmtree(dump)

    events = state.read_events(path)
    assert events == log.events
    archived = state.event_source(events, name="training_events", fallback=source)
    assert (archived / "step.jsonl").read_text() == "active generation\n"
    assert (archived.parent / discarded.name / "step.jsonl").read_text() == "discarded generation\n"
    (archived / "step.jsonl").write_text("corrupted\n")
    with pytest.raises(AssertionError, match="Archived evidence changed"):
        state.read_events(path)


@pytest.mark.parametrize("missing", ["tail", "middle"])
def test_an_incomplete_event_log_cannot_be_used_as_a_completed_collection(tmp_path: Path, missing: str) -> None:
    """A missing terminal marker or interior event must fail validation."""
    path = tmp_path / "events.jsonl"
    log = state.EventLog()
    log.persist_to(path)
    log.observe([])
    log.observe([staged("c", SERVING)])
    log.finish()
    lines = path.read_text().splitlines(keepends=True)
    del lines[-1 if missing == "tail" else 1]
    path.write_text("".join(lines))
    with pytest.raises(AssertionError, match="incomplete|Missing or reordered"):
        state.read_events(path)


def test_failed_durable_write_does_not_commit_a_request_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed evidence flush must prevent request admission from completing."""
    log = state.EventLog()
    log.persist_to(tmp_path / "events.jsonl")

    def fail_flush(fd: int) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(state.os, "fsync", fail_flush)
    with pytest.raises(OSError, match="disk unavailable"):
        log.note_action_requested(
            state.SoakActionRequest(target=cell("c", healthy=True), form_name="sigkill", harms_cell=True)
        )
    assert not log.events
