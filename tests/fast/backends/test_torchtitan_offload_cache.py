import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

pytest.importorskip("torchtitan")

from torchtitan.components.checkpointer import MODEL  # noqa: E402
from torchtitan.components.checkpointer.base import ModelWrapper  # noqa: E402

from miles.backends.torchtitan_utils.actor import checkpoint_cache_released  # noqa: E402
from miles.backends.training_utils.torch_native import offload  # noqa: E402


def _trainer(model: nn.Module) -> SimpleNamespace:
    # the checkpointer torchtitan's engine builds: a ModelWrapper whose cached state dict shares parameter storage
    return SimpleNamespace(engine=SimpleNamespace(checkpointer=SimpleNamespace(states={MODEL: ModelWrapper(model)})))


def _old_storages_alive(model: nn.Module, move) -> int:
    old = [weakref.ref(param.untyped_storage()) for param in model.parameters()]
    move()
    gc.collect()
    return sum(ref() is not None for ref in old)


@pytest.fixture
def host_copy(monkeypatch):
    # a CPU run has no accelerator to pin memory for or to synchronize; the copy replaces every tensor the same way
    monkeypatch.setattr(offload, "_to_pinned_host", lambda tensor: torch.empty_like(tensor).copy_(tensor))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)


def test_the_checkpoint_cache_keeps_every_old_parameter_alive_across_a_bare_move(host_copy):
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    trainer = _trainer(model)  # noqa: F841  the wrapper holds the cache
    assert _old_storages_alive(model, lambda: offload.move_train_state([model], [], to_host=True)) == 2


def test_releasing_the_checkpoint_cache_frees_the_old_parameters_and_rebuilds_the_cache(host_copy):
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    trainer = _trainer(model)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    def move():
        with checkpoint_cache_released(trainer):
            offload.move_train_state([model], [], to_host=True)

    assert _old_storages_alive(model, move) == 0
    cache = trainer.engine.checkpointer.states[MODEL].cached_state_dict
    assert cache.keys() == before.keys()
    for name, param in model.named_parameters():
        assert cache[name].data_ptr() == param.data_ptr()
        assert torch.equal(param, before[name])
