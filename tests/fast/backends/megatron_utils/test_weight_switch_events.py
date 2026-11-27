"""A model switch rebuilds derived weight caches (TOP's FP8 caches) right after the restore, before the switched-in
model's forward allocates activations, instead of lazily inside that forward."""

from types import SimpleNamespace

import pytest

actor_module = pytest.importorskip("miles.backends.megatron_utils.actor")


@pytest.fixture
def events(monkeypatch):
    from miles.backends.megatron_utils import weight_events

    calls = []
    monkeypatch.setattr(weight_events, "_ON_WEIGHTS_CHANGED", [lambda: calls.append("changed")])
    monkeypatch.setattr(weight_events, "_ON_TRAIN_START", [lambda: calls.append("materialize")])
    return calls


def _actor(calls, active="actor"):
    backuper = SimpleNamespace(backup_tags={"actor", "ref"}, restore=lambda tag: calls.append(f"restore {tag}"))
    return SimpleNamespace(_enable_weight_backup=True, weights_backuper=backuper, _active_model_tag=active)


def test_switching_models_rebuilds_caches_after_the_restore(events):
    actor = _actor(events)
    actor_module.MegatronTrainRayActor._switch_model(actor, "ref")
    assert events == ["restore ref", "changed", "materialize"]
    assert actor._active_model_tag == "ref"


def test_restoring_the_active_model_rebuilds_nothing(events):
    actor_module.MegatronTrainRayActor._switch_model(_actor(events), "actor")
    assert events == ["restore actor"]
