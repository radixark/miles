import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from tests.fast.fixtures.args_fixtures import make_trainer_config

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.function_registry import function_registry
from miles_plugins.models.inkling import lora


@pytest.fixture
def inkling_provider_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[SimpleNamespace]:
    from megatron.core import parallel_state

    from miles.backends.megatron_utils import model
    from miles.backends.training_utils import parallel

    layers = ModuleType("miles_plugins.models.inkling.layers")
    layers.__dict__.update(
        {
            name: type(name, (_UnusedInklingLayer,), {})
            for name in ("InklingDenseMLP", "InklingSelfAttention", "InklingSharedExperts")
        }
    )
    monkeypatch.setitem(sys.modules, layers.__name__, layers)

    provider_path = "miles_plugins.models.inkling.model.inkling_model_provider"
    args = make_trainer_config(
        custom_model_provider_path=CustomFunctionConfig(path=provider_path),
        model_name="inkling",
        megatron_to_hf_mode="raw",
        hf_checkpoint=str(tmp_path),
        load=None,
        pretrained_checkpoint=str(tmp_path),
        moe_use_upcycling=False,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0,
        lora_A_init_method="xavier",
        lora_type="lora",
        lora_adapter_path=None,
        debug_disable_optimizer=True,
        stream_optimizer_state_to_disk=False,
        enable_witness=False,
        optimizer="muon",
        muon_split_qkv=True,
        lr=0.001,
        use_gloo_process_groups=False,
    )
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    rank = SimpleNamespace(rank=0)
    monkeypatch.setattr(
        parallel, "get_parallel_state", lambda: SimpleNamespace(pp=rank, tp=rank, cp=rank, intra_dp=rank)
    )
    monkeypatch.setattr(model, "get_model", _build_local_model)
    monkeypatch.setattr(model, "is_first_replica_megatron_main_rank", lambda: True)
    monkeypatch.setattr(
        model, "get_megatron_muon_optimizer", lambda *, config, **kwargs: SimpleNamespace(config=config)
    )
    monkeypatch.setattr(model, "get_optimizer_param_scheduler", lambda args, optimizer: None)
    monkeypatch.setattr(model, "check_peak_gpu_memory_after_load", lambda args: None)
    monkeypatch.setattr(model, "clear_memory", lambda: None)
    monkeypatch.setattr(model, "check_model_hashes", lambda args, model, iteration: None)
    monkeypatch.setattr(lora, "_UNPADDED_VOCAB_CACHE", [None])

    with (
        function_registry.temporary(provider_path, _tiny_provider),
        function_registry.temporary("models.other.provider", _tiny_provider),
    ):
        yield SimpleNamespace(args=args, module=model)


@pytest.fixture
def inkling_adapter_model(inkling_provider_env: SimpleNamespace) -> torch.nn.Module:
    model = _TinyInklingModel()
    lora._apply_lm_head_lora(model, inkling_provider_env.args, scale=2, dropout=0, a_init="xavier")
    return model


@pytest.fixture
def inkling_tower_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    from miles.backends.megatron_utils.update_weight import hf_weight_iterator

    tensors = {"visual.weight": torch.ones(2), "audio.weight": torch.full((2,), 3), "language.weight": torch.zeros(2)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )
    monkeypatch.setattr(hf_weight_iterator, "_MM_TOWER_CACHE", None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    return SimpleNamespace(module=hf_weight_iterator, checkpoint=tmp_path, tensors=tensors)


@pytest.fixture
def inkling_reload_env(
    inkling_provider_env: SimpleNamespace,
    inkling_adapter_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    env = SimpleNamespace(
        args=inkling_provider_env.args,
        module=inkling_provider_env.module,
        model=inkling_adapter_model,
        optimizer=_OptimizerMasters(inkling_adapter_model),
        native_optimizer_restored=False,
    )
    monkeypatch.setattr(
        env.module, "load_checkpoint", lambda *args, **kwargs: (0, False, env.native_optimizer_restored)
    )
    return env


class _UnusedInklingLayer(torch.nn.Module):
    def __init__(self) -> None:
        raise AssertionError("The tiny Inkling fixture has no decoder layers")


class _TinyInklingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=2, sequence_parallel=False, inkling=SimpleNamespace(logits_mup_width_multiplier=None)
        )
        self.decoder = SimpleNamespace(layers=[])
        self.output_layer = torch.nn.Linear(2, 3, bias=False)
        self.post_process = True


def _tiny_provider(*, pre_process: bool, post_process: bool, args: Any) -> torch.nn.Module:
    return _TinyInklingModel()


def _build_local_model(provider: Callable[[], torch.nn.Module], model_type: Any) -> list[torch.nn.Module]:
    return [provider()]


class _OptimizerMasters:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.masters: dict[str, torch.Tensor] = {}

    def reload_model_params(self) -> None:
        self.masters = {name: tensor.detach().clone() for name, tensor in self.model.named_parameters()}
