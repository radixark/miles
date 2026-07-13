"""Pin for the dashboard lifecycle call site in
``compute_samples_from_openai_records`` (design doc §18.3): sample assembly
must keep chat-record timing on the Sample. If a session refactor moves the
assembly and drops the one-line call, this fails loudly."""

from types import SimpleNamespace

from miles.utils.lifecycle import TrajectoryLifecycle, attach_lifecycle_metadata
from miles.utils.types import Sample


def _record(ts, latency=None):
    meta_info = {} if latency is None else {"e2e_latency": latency}
    return SimpleNamespace(timestamp=ts, response={"choices": [{"meta_info": meta_info}]})


def test_segment_shape_with_and_without_latency():
    first, second = Sample(index=1), Sample(index=1)
    r1, r2 = _record(100.0, latency=8.5), _record(130.0)

    attach_lifecycle_metadata(first, r1, None, turn=1)
    assert first.metadata["lifecycle"] == dict(t0=91.5, t1=100.0, turn=1)

    attach_lifecycle_metadata(second, r2, r1, turn=2)
    assert second.metadata["lifecycle"] == dict(t0=None, t1=130.0, turn=2, prev_t1=100.0)


def test_sink_defaults_to_none():
    # without the dashboard installed every core probe is a guarded no-op
    assert TrajectoryLifecycle().sink is None
