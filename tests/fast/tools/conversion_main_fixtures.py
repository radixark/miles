import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from tools import convert_hf_to_torch_dist as converter

from miles.backends.megatron_utils import initialize
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


@dataclass
class _ConversionRuntime:
    root: Path
    backend: MegatronArgsNamespace | None = None
    fails: bool = False
    bridge_calls: list[tuple[str, bool]] = field(default_factory=list)
    barriers: list[str] = field(default_factory=list)

    def set_args(self, backend: MegatronArgsNamespace) -> None:
        self.backend = backend

    def build_model(self, provider: Any, model_type: Any, *, wrap_with_ddp: bool) -> list[torch.nn.Module]:
        assert callable(provider)
        assert wrap_with_ddp is False
        assert self.backend is not None
        assert self.backend.rank == 0
        self.backend.model_type = model_type
        if self.fails:
            raise RuntimeError("model SDK failed")
        return [torch.nn.Linear(1, 1)]

    def save(self, *args: Any) -> None:
        assert self.backend is not None
        output = Path(self.backend.save) / "iter_0000001"
        output.mkdir(parents=True)
        (output / "weights.bin").write_bytes(b"converted")
        (Path(self.backend.save) / "latest_checkpointed_iteration.txt").write_text("1")

    def barrier(self) -> None:
        self.barriers.append((self.root / "latest_checkpointed_iteration.txt").read_text())


@pytest.fixture
def conversion_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _ConversionRuntime:
    runtime = _ConversionRuntime(root=tmp_path / "output")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("MILES_BACKEND", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "--hf-checkpoint",
            "/input",
            "--save",
            str(runtime.root),
            "--num-layers",
            "2",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "2",
        ],
    )
    original_get_args = converter.get_args

    def parsed_config() -> converter._ConversionConfig:
        config = original_get_args()
        with config.backend.mutable():
            config.backend.rank = -7
        return config

    monkeypatch.setattr(converter, "get_args", parsed_config)
    monkeypatch.setattr(initialize, "set_args", runtime.set_args)
    for name in (
        "_initialize_distributed",
        "create_indep_dp_group",
        "create_megatron_parallel_state",
        "set_parallel_state",
        "set_random_seed_from_args",
        "register_hf_config_aliases",
        "init_num_microbatches_calculator",
    ):
        monkeypatch.setattr(initialize, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(initialize, "_build_tokenizer", lambda backend: setattr(backend, "padded_vocab_size", 128))
    monkeypatch.setattr(converter.dist, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(converter.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(converter.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(converter.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(converter.dist, "barrier", runtime.barrier)
    monkeypatch.setattr(converter.dist, "destroy_process_group", lambda: None)
    for name in ("set_device", "synchronize", "empty_cache"):
        monkeypatch.setattr(converter.torch.cuda, name, lambda *args: None)
    monkeypatch.setattr("miles.utils.memory_utils.available_memory", lambda: "CPU fixture")
    monkeypatch.setattr(converter, "get_model", runtime.build_model)
    monkeypatch.setattr(converter, "save_checkpoint", runtime.save)

    class _Bridge:
        @classmethod
        def from_pretrained(cls, path: str, *, trust_remote_code: bool) -> "_Bridge":
            runtime.bridge_calls.append((path, trust_remote_code))
            return cls()

        def load_weights(self, model: list[torch.nn.Module], path: str, *, memory_efficient: bool) -> None:
            assert len(model) == 1
            assert path == "/input"
            assert memory_efficient is True

    bridge = ModuleType("mbridge")
    bridge.AutoBridge = _Bridge
    monkeypatch.setitem(sys.modules, "mbridge", bridge)
    monkeypatch.setitem(sys.modules, "miles_plugins.mbridge", ModuleType("miles_plugins.mbridge"))
    return runtime
