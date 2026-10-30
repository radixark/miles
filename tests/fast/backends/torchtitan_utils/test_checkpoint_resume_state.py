"""A resume loads the optimizer state the checkpoint holds and starts fresh
only for the parameters it lacks; a missing model weight still fails."""

import logging

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful

pytest.importorskip("torchtitan")

from torchtitan.components.optimizer.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

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
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def state_dict(self):
        init_optim_state(self.optimizer)
        return get_flat_optim_state_dict(self.optimizer)

    def load_state_dict(self, state_dict):
        init_optim_state(self.optimizer)
        load_flat_optim_state_dict(self.optimizer, state_dict)


class _ModelState(Stateful):
    def __init__(self, model, drop=()):
        self.model, self.drop = model, drop

    def state_dict(self):
        return {k: v for k, v in get_model_state_dict(self.model).items() if k not in self.drop}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def _adam(model):
    return torch.optim.Adam(list(model.named_parameters()), lr=1e-3)


def _trained_and_saved(tmp_path, drop_model_keys=()):
    torch.manual_seed(0)
    model, optimizer = _Model(), None
    optimizer = _adam(model)
    model(torch.randn(2, 4)).backward()
    optimizer.step()
    optimizer.zero_grad()
    assert model.gate not in optimizer.state and model.a.weight in optimizer.state
    dcp.save(
        {"model": _ModelState(model, drop_model_keys), "optimizer": _OptimizerState(optimizer)},
        checkpoint_id=str(tmp_path),
    )
    return model, optimizer


def test_a_parameter_without_saved_state_starts_fresh_and_the_rest_load(tmp_path, caplog):
    saved_model, saved_optimizer = _trained_and_saved(tmp_path)
    model, optimizer = _Model(), None
    optimizer = _adam(model)
    state_dict = {"model": _ModelState(model), "optimizer": _OptimizerState(optimizer)}

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: optimizer.state.gate.step"):
        dcp.load(dict(state_dict), checkpoint_id=str(tmp_path))

    with caplog.at_level(logging.INFO):
        narrowed = components.without_unsaved_optimizer_state(state_dict, str(tmp_path))
    dcp.load(narrowed, checkpoint_id=str(tmp_path))

    assert narrowed["model"] is state_dict["model"]
    assert "3 optimizer state entries of 1 parameters" in caplog.text and "start fresh: gate" in caplog.text
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(optimizer.state[model.a.weight][name], saved_optimizer.state[saved_model.a.weight][name])
    assert model.gate not in optimizer.state
    assert torch.equal(model.a.weight, saved_model.a.weight)


def test_a_complete_checkpoint_is_loaded_untouched(tmp_path):
    model = _Model()
    optimizer = _adam(model)
    model(torch.randn(2, 4)).backward()
    optimizer.step()
    optimizer.zero_grad()
    optimizer.state[model.gate] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros(4, 4),
        "exp_avg_sq": torch.zeros(4, 4),
    }
    dcp.save({"model": _ModelState(model), "optimizer": _OptimizerState(optimizer)}, checkpoint_id=str(tmp_path))
    fresh = _Model()
    state_dict = {"model": _ModelState(fresh), "optimizer": _OptimizerState(_adam(fresh))}

    assert components.without_unsaved_optimizer_state(state_dict, str(tmp_path)) is state_dict


def test_a_missing_model_weight_still_fails(tmp_path):
    _trained_and_saved(tmp_path, drop_model_keys=("gate",))
    model = _Model()
    state_dict = {"model": _ModelState(model), "optimizer": _OptimizerState(_adam(model))}
    narrowed = components.without_unsaved_optimizer_state(state_dict, str(tmp_path))

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: model.gate"):
        dcp.load(narrowed, checkpoint_id=str(tmp_path))
