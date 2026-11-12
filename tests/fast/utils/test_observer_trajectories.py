"""Rollout code's Samples reach the observers as plain lifecycle events with the sample's identity and newest version."""

from types import SimpleNamespace

import pytest

from miles.utils import observers as observers_module
from miles.utils.misc import SingletonMeta
from miles.utils.observers import Observer, Observers
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class _EventRecorder(Observer):
    def __init__(self):
        self.events: list[dict] = []

    def trajectory_event(self, event):
        self.events.append(event)


@pytest.fixture
def recorder(monkeypatch) -> _EventRecorder:
    observer = _EventRecorder()
    monkeypatch.setattr(
        observers_module,
        "entry_points",
        lambda group: [SimpleNamespace(name="recorder", load=lambda: lambda: observer)],
    )
    SingletonMeta._instances.pop(Observers, None)
    return observer


@pytest.fixture
def reporter(recorder):
    return observers_module._TrajectoryReporter(Observers())


def _sample(index=7, group=2, versions=("3", "4")) -> Sample:
    return Sample(
        index=index,
        group_index=group,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan(version, i, i + 1)]) for i, version in enumerate(versions)
        ],
    )


def test_attempt_and_generation_events_carry_the_sample_identity_and_newest_version(recorder, reporter):
    """Each event names its sample and group and the newest weight version; the attempt end carries the status."""
    sample = _sample()

    reporter.attempt_start(sample)
    reporter.gen_start(sample)
    reporter.attempt_end(sample)

    assert [event["kind"] for event in recorder.events] == ["attempt_start", "gen_start", "attempt_end"]
    for event in recorder.events:
        assert (event["sample_index"], event["group_index"], event["weight_version"]) == (7, 2, "4")
    assert recorder.events[-1]["detail"] == Sample.Status.PENDING.value


def test_an_event_without_weight_version_spans_has_an_empty_version(recorder, reporter):
    """A sample whose generation reported no weight version is still reported, with an empty version."""
    reporter.attempt_start(_sample(versions=()))

    [event] = recorder.events
    assert (event["kind"], event["sample_index"], event["weight_version"]) == ("attempt_start", 7, "")


def test_the_version_is_the_last_span_of_the_last_call(recorder, reporter):
    """The reported version is the last span of a multi-span call, not that call's first span."""
    sample = Sample(
        index=7,
        group_index=2,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("4", 1, 2), WeightVersionSpan("5", 2, 3)]),
        ],
    )

    reporter.gen_start(sample)

    [event] = recorder.events
    assert event["weight_version"] == "5"


def test_spans_use_explicit_timestamps_and_turns(recorder, reporter):
    """Spans recorded after the fact keep their own times, turn and detail rather than the time of the report."""
    sample = _sample()

    reporter.gen_span(sample, 10.0, 14.5, turn=3, detail="120")
    reporter.tool_span(sample, 14.5, 16.0, turn=3, detail="2 calls")

    assert [(event["kind"], event["ts"], event["turn"]) for event in recorder.events] == [
        ("gen_start", 10.0, 3),
        ("gen_end", 14.5, 3),
        ("tool_start", 14.5, 3),
        ("tool_end", 16.0, 3),
    ]
    assert recorder.events[1]["detail"] == "120"


def test_a_turn_with_an_unknown_start_reports_only_its_end(recorder, reporter):
    """A turn whose engine reported no latency has no start; the agent gap before the next turn ends at its arrival."""
    turns = []
    for i in range(2):
        sample = _sample()
        sample.metadata["lifecycle"] = dict(t0=100.0 + i, t1=110.0 + i, turn=i + 1)
        turns.append(sample)
    turns[0].metadata["lifecycle"]["t0"] = None
    turns[1].metadata["lifecycle"]["prev_t1"] = 110.0
    turns[1].metadata["lifecycle"]["req_ts"] = 110.6

    reporter.attempt_end(turns)

    events = recorder.events
    assert [(event["kind"], event["turn"]) for event in events] == [
        ("gen_end", 1),
        ("tool_start", 2),
        ("tool_end", 2),
        ("gen_start", 2),
        ("gen_end", 2),
        ("attempt_end", -1),
    ]
    assert (events[1]["ts"], events[2]["ts"]) == (110.0, 110.6)
    assert events[1]["detail"] == "agent gap"
