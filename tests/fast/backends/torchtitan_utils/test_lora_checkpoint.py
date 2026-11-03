import socket
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils import lora_checkpoint  # noqa: E402
from miles.backends.torchtitan_utils.lora_checkpoint import (  # noqa: E402
    TorchtitanCheckpointLoader,
    TorchtitanLoraCheckpoint,
)

_LORA_A, _LORA_B = "layers.0.attention.wq.lora_a.weight", "layers.0.attention.wq.lora_b.weight"


@pytest.fixture
def single_rank_gloo():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@dataclass(frozen=True)
class _Run:
    base_model: str
    lora_alpha: int | None


def _load_dcp(tmp_path, saved: dict) -> TorchtitanCheckpointLoader:
    path = tmp_path / "step-10"
    dcp.save(saved, checkpoint_id=path)
    params = {key: torch.nn.Parameter(torch.zeros(2, 2)) for key in (_LORA_A, _LORA_B)}
    slots = SimpleNamespace(parameters=params)
    members = SimpleNamespace(adam_states=lambda: [{"step": torch.tensor(0.0)} for _ in params])
    sd_adapter = SimpleNamespace(to_hf=lambda tensors: {})
    loader = TorchtitanCheckpointLoader(
        slots, members, [], sd_adapter, _Run(str(tmp_path), 16), load_optimizer=False
    )
    with (
        patch.object(lora_checkpoint, "resolve_hf_dir", return_value=tmp_path),
        patch.object(lora_checkpoint, "hf_fingerprint_tensors", return_value={}),
        patch.object(lora_checkpoint, "assert_same_base_tensors"),
    ):
        loader(path)
    for key, param in params.items():
        assert torch.equal(param, saved[key])
    return loader


def test_a_model_only_dcp_loads_the_adapters_without_training_state(tmp_path, single_rank_gloo):
    """torchtitan's default final checkpoint (last_save_model_only) holds no train_state or lr_scheduler."""
    loader = _load_dcp(tmp_path, {_LORA_A: torch.ones(2, 2), _LORA_B: torch.full((2, 2), 2.0)})
    assert (loader.train_state, loader.lr_scheduler_state) == (None, None)


def test_a_full_dcp_loads_its_training_state(tmp_path, single_rank_gloo):
    saved = {_LORA_A: torch.ones(2, 2), _LORA_B: torch.full((2, 2), 2.0)}
    saved |= {"train_state.step": 8, "train_state.ntokens_seen": 640, "lr_scheduler.last_epoch": 8}
    loader = _load_dcp(tmp_path, saved)
    assert loader.train_state == {"step": 8, "ntokens_seen": 640}
    assert loader.lr_scheduler_state == {"last_epoch": 8}


_STEPS_PER_ROLLOUT = 4


class _Engine:
    def __init__(self):
        self.num_completed_steps, self.ntokens_seen = 0, 0
        self.optim = SimpleNamespace(lr_schedulers=MagicMock())
        self.state_dict_adapter = None

    def load_state_dict(self, state: dict) -> None:
        self.num_completed_steps, self.ntokens_seen = state["step"], state["ntokens_seen"]


class _Trainer:
    model_parts = []

    def __init__(self):
        self.engine = _Engine()
        self.optimizers = SimpleNamespace(optimizers=[])

    @property
    def step(self) -> int:
        return self.engine.num_completed_steps


def _restore(tmp_path, *, dcp_step=None, training_state=None, **flags):
    """``dcp_step``: the source is a torchtitan DCP with this train_state step; ``training_state``: an adapter dir's."""

    def load_adapter_dir(path, members, slot, run, targets, *, load_optimizer, in_model_loaders_by_kind):
        if dcp_step is not None:
            (loader,) = in_model_loaders_by_kind.values()
            loader.train_state = {"step": dcp_step, "ntokens_seen": 7}
            loader.lr_scheduler_state = {"last_epoch": dcp_step}

    backend = SimpleNamespace(no_load_optim=False, no_save_optim=False, save=None)
    args = SimpleNamespace(
        lora_adapter_path=str(tmp_path / "source"),
        lora_adapter_convert_save=None,
        num_rollout=10,
        hf_checkpoint=str(tmp_path),
        lora_rank=8,
        lora_alpha=16,
        lora_type="lora",
        backend=backend,
    )
    for name, value in flags.items():
        setattr(backend if hasattr(backend, name) else args, name, value)
    trainer = _Trainer()
    with (
        patch.object(lora_checkpoint, "_torchtitan_adapter_slots", return_value=MagicMock()),
        patch.object(lora_checkpoint, "load_adapter_dir", side_effect=load_adapter_dir),
        patch.object(lora_checkpoint, "read_training_state", return_value=training_state),
        patch.object(lora_checkpoint, "load_lr_schedulers") as load_lr_schedulers,
        patch.object(lora_checkpoint, "save_and_commit_adapter_dir") as save,
    ):
        iteration = TorchtitanLoraCheckpoint(trainer, args, _STEPS_PER_ROLLOUT).restore()
    lr_states = [call.args[1] for call in load_lr_schedulers.call_args_list]
    return SimpleNamespace(iteration=iteration, step=trainer.step, lr_states=lr_states, save=save)


@pytest.mark.parametrize("step", [6, 11 * _STEPS_PER_ROLLOUT])
def test_a_dcp_off_this_runs_rollout_grid_is_refused(tmp_path, step):
    """A standalone torchtitan run's step count says nothing about this run's rollouts."""
    with pytest.raises(AssertionError, match=f"at step {step}, not a whole rollout"):
        _restore(tmp_path, dcp_step=step)
