import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from transformers import Qwen2Config

from miles.utils.debug_utils.run_megatron.worker import main as worker
from miles.utils.debug_utils.run_megatron.worker.batch import prepare_batch
from miles.utils.replay_base import Replay, routing_replay_manager


@dataclass
class MainReplayBoundary:
    hf: Path
    tokens: Path
    replay_directory: Path
    replay: Replay
    model: torch.nn.Module


@pytest.fixture
def main_replay_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sequence_parallel: bool
) -> MainReplayBoundary:
    hf = tmp_path / "hf"
    Qwen2Config(
        hidden_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, intermediate_size=256
    ).save_pretrained(hf)
    flags = ["--sequence-parallel"] if sequence_parallel else []
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    def device_properties(device: torch.device) -> SimpleNamespace:
        assert device == torch.device("cuda:0")
        return SimpleNamespace(major=9)

    monkeypatch.setattr(torch.cuda, "get_device_properties", device_properties)
    config = parse_megatron_test_config(
        "--ffn-hidden-size",
        "256",
        "--untie-embeddings-and-output-weights",
        "--norm-epsilon",
        "1e-6",
        "--debug-train-only",
        "--no-offload-train",
        "--actor-num-gpus-per-node",
        "2",
        "--tensor-model-parallel-size",
        "2",
        "--megatron-to-hf-mode",
        "raw",
        "--load",
        str(tmp_path / "run"),
        "--ref-load",
        str(tmp_path / "reference"),
        "--hf-checkpoint",
        str(hf),
        *flags,
    )
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps([1, 2, 3, 4]))
    replay_directory = tmp_path / "replay"
    replay_directory.mkdir()
    torch.save([[torch.arange(8)]], replay_directory / f"rank0_{routing_replay_manager.filename}")
    replay = Replay()
    model = torch.nn.Linear(2, 2)
    model.eval()
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=2, rank=1))

    monkeypatch.setattr(routing_replay_manager, "replays", [replay])
    monkeypatch.setattr(routing_replay_manager, "enabled", False)
    monkeypatch.setattr(routing_replay_manager, "stage", "fallthrough")
    monkeypatch.setattr(worker, "parse_args", lambda: config)
    monkeypatch.setattr(worker, "init", lambda args: None)
    monkeypatch.setattr(worker.dist, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(worker.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(worker.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(worker.dist, "barrier", lambda: None)
    monkeypatch.setattr(worker.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(worker.torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(worker, "get_model", lambda *args, **kwargs: [model])
    monkeypatch.setattr(worker, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker.dumper, "register_non_intrusive_dumper", lambda model: None)
    monkeypatch.setattr(worker.mpu, "is_initialized", lambda: True)
    monkeypatch.setattr(worker.mpu, "is_pipeline_last_stage", lambda: False)
    monkeypatch.setattr(worker, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr("miles.backends.training_utils.parallel.get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(worker, "prepare_batch", partial(prepare_batch, device="cpu"))
    monkeypatch.setattr(worker, "get_forward_backward_func", lambda: lambda **kwargs: [])
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("DUMPER_ENABLE", "0")
    return MainReplayBoundary(hf=hf, tokens=tokens, replay_directory=replay_directory, replay=replay, model=model)
