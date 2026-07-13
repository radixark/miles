import pytest

from miles.dashboard import hooks
from miles.dashboard.hooks import BATCH_MAX_EVENTS, TrajectorySink
from miles.dashboard.store import Stream, TrajectoryEvent, TrajectoryEventKind
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample


class FakeRemoteMethod:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def remote(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("collector unreachable")
        self.calls.append((args, kwargs))


class FakeHandle:
    def __init__(self, fail_push=False):
        self.push_trajectories = FakeRemoteMethod(fail=fail_push)


@pytest.fixture(autouse=True)
def clean_sink():
    TrajectoryLifecycle().sink = None
    yield
    TrajectoryLifecycle().sink = None


def _sample(index=7, group=2, versions=("3", "4")):
    return Sample(index=index, group_index=group, weight_versions=list(versions))


def _pushed(handle):
    return [event for (args, _) in handle.push_trajectories.calls for event in args[0]]


# --------------------------------- emitters ----------------------------------


def test_attempt_and_gen_events_carry_identity_and_version():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    sink.attempt_start(sample)
    sink.gen_start(sample)
    sink.attempt_end(sample)
    sink.flush()

    kinds = [e.kind for e in _pushed(handle)]
    assert kinds == [
        TrajectoryEventKind.ATTEMPT_START,
        TrajectoryEventKind.GEN_START,
        TrajectoryEventKind.ATTEMPT_END,
    ]
    for event in _pushed(handle):
        assert (event.sample_index, event.group_index) == (7, 2)
        assert event.weight_version == "4"
    assert _pushed(handle)[-1].detail == Sample.Status.PENDING.value


def test_spans_use_explicit_timestamps_and_turns():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    sink.gen_span(sample, 10.0, 14.5, turn=3, detail="120")
    sink.tool_span(sample, 14.5, 16.0, turn=3, detail="2 calls")
    sink.flush()

    events = _pushed(handle)
    assert [(e.kind, e.ts, e.turn) for e in events] == [
        (TrajectoryEventKind.GEN_START, 10.0, 3),
        (TrajectoryEventKind.GEN_END, 14.5, 3),
        (TrajectoryEventKind.TOOL_START, 14.5, 3),
        (TrajectoryEventKind.TOOL_END, 16.0, 3),
    ]
    assert events[1].detail == "120"


def test_attempt_end_drains_lifecycle_metadata_from_turn_samples():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    turns = []
    for i in range(2):
        sample = _sample()
        sample.metadata["lifecycle"] = dict(t0=100.0 + i, t1=110.0 + i, turn=i + 1)
        turns.append(sample)
    turns[0].metadata["lifecycle"]["t0"] = None  # unknown start: only the end emits

    sink.attempt_end(turns)
    sink.flush()
    events = _pushed(handle)
    assert [(e.kind, e.turn) for e in events] == [
        (TrajectoryEventKind.GEN_END, 1),
        (TrajectoryEventKind.GEN_START, 2),
        (TrajectoryEventKind.GEN_END, 2),
        (TrajectoryEventKind.ATTEMPT_END, -1),
    ]
    assert events[1].ts == 101.0 and events[2].ts == 111.0


def test_batching_and_failure_swallowed():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    for _ in range(BATCH_MAX_EVENTS - 1):
        sink.gen_start(sample)
    assert handle.push_trajectories.calls == []
    sink.gen_start(sample)  # crosses the batch threshold
    assert len(_pushed(handle)) == BATCH_MAX_EVENTS

    failing = TrajectorySink(FakeHandle(fail_push=True))
    for _ in range(BATCH_MAX_EVENTS + 1):
        failing.gen_start(sample)  # must never raise into the rollout path
    failing.flush()


def test_attach_detach_via_lifecycle_seam():
    assert TrajectoryLifecycle().sink is None
    handle = FakeHandle()
    hooks.attach_trajectory_sink(handle)
    sink = TrajectoryLifecycle().sink
    assert isinstance(sink, TrajectorySink)
    hooks.attach_trajectory_sink(FakeHandle())
    assert TrajectoryLifecycle().sink is sink  # one sink per process

    sink.gen_start(_sample())
    hooks.detach_and_flush()
    assert TrajectoryLifecycle().sink is None
    assert len(_pushed(handle)) == 1  # detach flushed the partial batch


def test_trajectory_events_roundtrip_store(tmp_path):
    from miles.dashboard.store import MetricStore

    writer = MetricStore(tmp_path)
    for ts in (10.0, 3700.0):
        writer.append(
            TrajectoryEvent(
                ts=ts,
                kind=TrajectoryEventKind.GEN_START,
                sample_index=5,
                group_index=-1,
                turn=1,
                weight_version="2",
                detail="",
            )
        )
    writer.flush()
    store = MetricStore.load(tmp_path)
    assert store.has_stream(Stream.TRAJECTORIES)
    assert len((tmp_path / "trajectories").glob("*.jsonl") and list((tmp_path / "trajectories").glob("*.jsonl"))) == 2
    assert [e.ts for e in store.trajectory_events()] == [10.0, 3700.0]
    assert [e.ts for e in store.trajectory_events(t0=3600.0)] == [3700.0]
    assert store.iter_records(Stream.TRAJECTORIES)[0].weight_version == "2"
