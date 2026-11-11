"""A resume loads the optimizer state the checkpoint holds and starts fresh
only for the parameters it lacks; a missing model weight still fails.

A save written under torchtitan 0.3.0 lacks optimizer state for a parameter
that never received a gradient: its ``init_optim_state`` returned early once
any state existed. torchtitan 0.4.0 fills that state in before each save, so
its own saves hold every trainable parameter."""

import logging

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful

pytest.importorskip("torchtitan")

from torchtitan.components.checkpointer import MODEL, OPTIMIZER
from torchtitan.components.optim.utils import get_flat_optim_state_dict, init_optim_state, load_flat_optim_state_dict

from miles.backends.torchtitan_utils import components


class _Model(nn.Module):
    """``gate`` is a parameter no forward uses, like the DSA indexer's gates
    behind a no_grad block: it never receives a gradient."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.gate = nn.Parameter(torch.zeros(4, 4))

    def forward(self, x):
        return self.a(x).sum()


class _OptimizerState(Stateful):
    """What torchtitan 0.4.0's ``OptimizersContainer`` does for one optimizer."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def state_dict(self):
        init_optim_state(self.optimizer)
        return get_flat_optim_state_dict(self.optimizer)

    def load_state_dict(self, state_dict):
        init_optim_state(self.optimizer)
        load_flat_optim_state_dict(self.optimizer, state_dict)


class _OptimizerStateSavedBy030(_OptimizerState):
    """A save after training under torchtitan 0.3.0: state exists, so its
    ``init_optim_state`` did nothing and parameters without state are left out."""

    def state_dict(self):
        return get_flat_optim_state_dict(self.optimizer)


class _ModelState(Stateful):
    """torchtitan 0.4.0's ``ModelWrapper``: the checkpoint holds the model's
    keys at the top level, and the load hands it the whole flat dict."""

    def __init__(self, model, drop=()):
        self.model, self.drop = model, drop

    def state_dict(self):
        return {k: v for k, v in get_model_state_dict(self.model).items() if k not in self.drop}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)


def _adam(model):
    return torch.optim.Adam(list(model.named_parameters()), lr=1e-3)


def _trained(model):
    optimizer = _adam(model)
    model(torch.randn(2, 4)).backward()
    optimizer.step()
    optimizer.zero_grad()
    assert model.gate not in optimizer.state and model.a.weight in optimizer.state
    return optimizer


def _save(path, model, optimizer_state, drop_model_keys=()):
    dcp.save({**_ModelState(model, drop_model_keys).state_dict(), OPTIMIZER: optimizer_state}, checkpoint_id=str(path))


def _fresh_states():
    model = _Model()
    optimizer = _adam(model)
    return model, optimizer, {MODEL: _ModelState(model), OPTIMIZER: _OptimizerState(optimizer)}


def _load(states, path):
    """``TiedCheckpointManager._load_checkpoint`` on the non-HF path; nothing
    else of the manager is used there."""
    manager = object.__new__(components.TiedCheckpointManager)
    manager._load_checkpoint(states, str(path), from_hf=False, from_quantized=False)


def test_a_parameter_without_saved_state_starts_fresh_and_the_rest_load(tmp_path, caplog):
    torch.manual_seed(0)
    saved_model = _Model()
    saved_optimizer = _trained(saved_model)
    _save(tmp_path, saved_model, _OptimizerStateSavedBy030(saved_optimizer))

    _, _, states = _fresh_states()
    flat = {**states[MODEL].state_dict(), OPTIMIZER: states[OPTIMIZER]}
    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: optimizer.state.gate.step"):
        dcp.load(flat, checkpoint_id=str(tmp_path))

    model, optimizer, states = _fresh_states()
    with caplog.at_level(logging.INFO):
        _load(states, tmp_path)

    assert "3 optimizer state entries of 1 parameters" in caplog.text and "start fresh: gate" in caplog.text
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(optimizer.state[model.a.weight][name], saved_optimizer.state[saved_model.a.weight][name])
    assert model.gate not in optimizer.state
    assert torch.equal(model.a.weight, saved_model.a.weight)


def test_a_torchtitan_040_save_holds_every_parameter_and_loads_untouched(tmp_path):
    saved_model = _Model()
    _save(tmp_path, saved_model, _OptimizerState(_trained(saved_model)))
    _, _, states = _fresh_states()

    assert components.unsaved_optimizer_state(states[OPTIMIZER], str(tmp_path)) == []
    assert components.without_unsaved_optimizer_state(states, str(tmp_path)) is states


def test_a_missing_model_weight_still_fails(tmp_path):
    saved_model = _Model()
    _save(tmp_path, saved_model, _OptimizerStateSavedBy030(_trained(saved_model)), drop_model_keys=("gate",))
    _, _, states = _fresh_states()

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: gate"):
        _load(states, tmp_path)
