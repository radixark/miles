import socket
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils import lora_checkpoint  # noqa: E402
from miles.backends.torchtitan_utils.lora_checkpoint import TorchtitanCheckpointLoader  # noqa: E402

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
