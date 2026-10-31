"""Each step's samples reach run observers as plain records, one per sample."""

import logging
from types import SimpleNamespace

import pytest

from miles.ray.rollout import observer_reports
from miles.utils.misc import SingletonMeta
from miles.utils.observers import Observer, Observers
from miles.utils.types import Sample


class _Recorder(Observer):
    def __init__(self):
        self.steps = []

    def rollout_samples(self, rollout_id, *, evaluation, samples):
        self.steps.append((rollout_id, evaluation, samples))


@pytest.fixture
def recorder(monkeypatch):
    recorder = _Recorder()
    SingletonMeta._instances.pop(Observers, None)
    installed = [SimpleNamespace(name="recorder", load=lambda: lambda: recorder)]
    monkeypatch.setattr("miles.utils.observers.entry_points", lambda group: installed)
    yield recorder
    SingletonMeta._instances.pop(Observers, None)


def _sample(index: int, *, reward, messages=None) -> Sample:
    sample = Sample(index=index, group_index=0, tokens=[1, 2, 3, 4], response_length=2, reward=reward)
    sample.loss_mask = [1, 0]
    sample.rollout_log_probs = [-0.5, -1.5]
    if messages is not None:
        sample.metadata = {"messages": messages}
    return sample


def test_leaves_sharing_an_index_are_numbered_and_carry_their_conversations(recorder):
    messages = [{"role": "user", "content": "hi"}]
    samples = [_sample(5, reward=1.0), _sample(5, reward=0.0, messages=messages), _sample(6, reward={"score": 1})]

    observer_reports.report_rollout_samples(samples, rollout_id=3, evaluation=False)

    [(rollout_id, evaluation, records)] = recorder.steps
    assert (rollout_id, evaluation) == (3, False)
    assert [(r["sample_index"], r["sample_occurrence"]) for r in records] == [(5, 0), (5, 1), (6, 0)]
    assert [r["messages"] for r in records] == [None, messages, None]
    assert records[2]["reward"] is None, "a non-numeric reward is not a number to plot"
    assert records[0]["tokens"] == [1, 2, 3, 4]
    assert records[0]["rollout_log_probs"] == [-0.5, -1.5]


def test_samples_are_not_built_for_observers_that_do_not_take_them(monkeypatch, caplog):
    """An observer without ``rollout_samples`` must not cost the rollout the extraction of every sample."""

    class _MetricsOnly(Observer):
        def metrics(self, metrics, *, step_key, step):
            pass

    installed = [SimpleNamespace(name="metrics", load=lambda: _MetricsOnly)]
    monkeypatch.setattr("miles.utils.observers.entry_points", lambda group: installed)
    SingletonMeta._instances.pop(Observers, None)

    def _unexpected(samples):
        raise AssertionError("samples were built")

    monkeypatch.setattr(observer_reports, "_rollout_sample_records", _unexpected)
    with caplog.at_level(logging.WARNING):
        observer_reports.report_rollout_samples([_sample(5, reward=1.0)], rollout_id=3, evaluation=False)

    assert not [r for r in caplog.records if "rollout_samples" in r.message]
    SingletonMeta._instances.pop(Observers, None)
