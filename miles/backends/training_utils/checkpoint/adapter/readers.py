"""Identify an adapter source by its files and read the file-only kinds into HF-keyed tensors."""

import enum
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    ADAPTER_STATE_DIRNAME,
    AdapterState,
    load_adapter_state,
    read_adapter_state_keys,
)
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    is_tinker_only_name,
    map_peft_keys_to_hf_keys,
    map_tinker_keys_to_hf_keys,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    PARAMETRIZATION_FILENAME,
    AdapterParametrization,
    read_parametrization,
)
from miles.utils.lora.hf_lora_targets import GDN_IN_PROJ_COMPONENTS_BY_MODULE, HfLoraTargets

_PEFT_FILES = ("adapter_config.json", "adapter_model.safetensors")
# PEFT features that change what a tensor means; an adapter using them cannot be read as plain LoRA
_PEFT_CONFIG_MUST_BE_EMPTY = (
    "use_dora",
    "use_rslora",
    "modules_to_save",
    "rank_pattern",
    "alpha_pattern",
    "target_parameters",
)
_SAMPLER_EXPERT_NAMES = {
    ".experts.w1.": ".experts.gate_proj.",
    ".experts.w3.": ".experts.up_proj.",
    ".experts.w2.": ".experts.down_proj.",
}


class AdapterSourceKind(str, enum.Enum):
    ADAPTER_DIR = "adapter_dir"
    PEFT = "peft"
    EXTERNAL_WEIGHTS = "external_weights"
    # read into the run's model by the backend (in_model_loaders_by_kind)
    LEGACY_TINKER_CHECKPOINT = "legacy_tinker_checkpoint"
    LEGACY_RANK_SHARD = "legacy_rank_shard"
    MEGATRON_BRIDGE_CHECKPOINT = "megatron_bridge_checkpoint"
    TORCHTITAN_CHECKPOINT = "torchtitan_checkpoint"

    def is_read_into_model(self) -> bool:
        return self not in (AdapterSourceKind.ADAPTER_DIR, AdapterSourceKind.PEFT, AdapterSourceKind.EXTERNAL_WEIGHTS)


@dataclass(frozen=True)
class AdapterSource:
    """Metadata only; tensors are read separately."""

    kind: AdapterSourceKind
    parametrization: AdapterParametrization
    has_optimizer_state: bool
    weight_dtype: torch.dtype


def _has_peft_files(path: Path) -> bool:
    return all((path / f).is_file() for f in _PEFT_FILES)


def _safetensors_names(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(path / "adapter_model.safetensors"), "pt") as f:
        return list(f.keys())


def classify_adapter_source(path: Path, targets: HfLoraTargets, *, load_optimizer: bool) -> AdapterSourceKind:
    path = Path(path)
    if (path / PARAMETRIZATION_FILENAME).is_file():
        return AdapterSourceKind.ADAPTER_DIR
    if (path / ".metadata").is_file():
        if (path / "META.json").is_file():
            return AdapterSourceKind.LEGACY_TINKER_CHECKPOINT
        if (path / "run_config.yaml").is_file():
            return AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT
        return AdapterSourceKind.TORCHTITAN_CHECKPOINT
    if any(path.glob("adapter_megatron_rank*.pt")) or any(path.glob("adapter_megatron_tp*_pp*.pt")):
        if not load_optimizer and _has_peft_files(path):
            return AdapterSourceKind.PEFT
        return AdapterSourceKind.LEGACY_RANK_SHARD
    if _has_peft_files(path):
        names = _safetensors_names(path)
        if targets.tinker_name_renames and any(is_tinker_only_name(n, targets) for n in names):
            return AdapterSourceKind.EXTERNAL_WEIGHTS
        return AdapterSourceKind.PEFT
    raise ValueError(f"{path} is not a recognized LoRA adapter source")


def _collapse_shared_lora_a(weights: dict, parametrization: AdapterParametrization) -> dict:
    """Undo a written expansion: keep the first member's A, after checking every copy matches it."""
    for group in parametrization.shared_lora_a_groups:
        first = weights[f"{group[0]}.lora_A"]
        for member in group[1:]:
            assert torch.equal(
                weights.pop(f"{member}.lora_A"), first
            ), f"{member}.lora_A differs inside a shared-A group"
    return weights


def _read_peft_tensors(
    path: Path, map_name, targets: HfLoraTargets
) -> tuple[dict[str, torch.Tensor], list[tuple[str, ...]]]:
    """HF-keyed whole tensors; per-expert 2D stacks to 3D, an HF-fused module splits into components sharing A."""
    from safetensors.torch import load_file

    per_expert: dict[str, dict[int, torch.Tensor]] = {}
    weights, fused_groups = {}, []
    for name, value in load_file(str(path / "adapter_model.safetensors")).items():
        hf_key = map_name(name)
        if hf_key.expert_index is not None:
            per_expert.setdefault(hf_key.key, {})[hf_key.expert_index] = value
            continue
        module, factor = hf_key.key.rsplit(".", 1)
        components = _fused_components(module)
        if components is None:
            weights[hf_key.key] = value
        elif factor == "lora_A":
            weights[f"{components[0]}.lora_A"] = value
            fused_groups.append(components)
        else:
            rows = dict(targets.fused_component_rows)[module.rsplit(".", 1)[1]]
            for component, part in zip(components, value.split(rows), strict=True):
                weights[f"{component}.lora_B"] = part.clone()
    for key, experts in per_expert.items():
        assert sorted(experts) == list(range(len(experts))), f"{key}: expert indices are not 0..E-1"
        weights[key] = torch.stack([experts[i] for i in range(len(experts))])
    return weights, fused_groups


class MilesAdapterDirReader:
    """A directory this library wrote: ``adapter_state/`` (training) or a sampler export (weights only)."""

    def __init__(self, path: Path, targets: HfLoraTargets):
        self.path, self.targets = Path(path), targets
        parametrization = read_parametrization(self.path)
        has_state = (self.path / ADAPTER_STATE_DIRNAME).is_dir()
        has_optimizer_state = has_state and any(k.endswith(".exp_avg") for k in read_adapter_state_keys(self.path))
        self.source = AdapterSource(
            AdapterSourceKind.ADAPTER_DIR,
            parametrization,
            has_optimizer_state,
            torch.float32 if has_state else torch.bfloat16,
        )

    def has_adapter_state(self) -> bool:
        return (self.path / ADAPTER_STATE_DIRNAME).is_dir()

    def load_state(self, template: AdapterState) -> int:
        return load_adapter_state(self.path, template)

    def read_weights(self) -> dict[str, torch.Tensor]:
        if (self.path / ADAPTER_STATE_DIRNAME).is_dir():
            from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

            keys = {k for k in read_adapter_state_keys(self.path) if k.endswith(".weight")}
            loaded = _load_state_dict_from_keys(keys, checkpoint_id=self.path / ADAPTER_STATE_DIRNAME)
            return {k.removesuffix(".weight"): v for k, v in loaded.items()}

        def map_name(name):
            for sglang_fragment, hf_fragment in _SAMPLER_EXPERT_NAMES.items():
                name = name.replace(sglang_fragment, hf_fragment)
            return map_peft_keys_to_hf_keys(name)

        weights, fused_groups = _read_peft_tensors(self.path, map_name, self.targets)
        unfused = replace(
            self.source.parametrization,
            shared_lora_a_groups=tuple(
                g for g in self.source.parametrization.shared_lora_a_groups if g not in fused_groups
            ),
        )
        return _collapse_shared_lora_a(weights, unfused)


def _check_peft_config(config: dict, path: Path) -> None:
    used = [k for k in _PEFT_CONFIG_MUST_BE_EMPTY if config.get(k)]  # absent and empty both mean unused
    assert not used, f"{path}: PEFT features {used} change what the tensors mean; only plain LoRA can be read"


def _fused_components(module: str) -> tuple[str, ...] | None:
    parent, leaf = module.rsplit(".", 1)
    components = GDN_IN_PROJ_COMPONENTS_BY_MODULE.get(leaf)
    if components is None or len(components) == 1:
        return None
    return tuple(f"{parent}.{c}" for c in components)


class PeftAdapterReader:
    """HF-named PEFT (main's exports, HF Hub); per-expert 2D tensors are stacked to 3D."""

    _map_name = staticmethod(lambda name, targets: map_peft_keys_to_hf_keys(name))

    def __init__(self, path: Path, run: AdapterParametrization, targets: HfLoraTargets):
        self.path, self.targets = Path(path), targets
        config = json.loads((self.path / "adapter_config.json").read_text())
        _check_peft_config(config, self.path)
        self._weights, self._fused_groups = _read_peft_tensors(
            self.path, lambda name: self._map_name(name, targets), targets
        )
        modules = sorted({k.rsplit(".", 1)[0] for k in self._weights if k.endswith(".lora_B")})
        shared_outer = any(
            k.endswith(".experts.gate_proj.lora_A") and v.shape[0] == 1 for k, v in self._weights.items()
        )
        groups = tuple(self._fused_groups)
        # a PEFT file does not record the base model or LoRA flavour; they are the run's
        self.source = AdapterSource(
            self.kind,
            replace(
                run,
                lora_rank=config["r"],
                lora_alpha=config["lora_alpha"],
                target_modules=tuple(modules),
                experts_shared_outer_loras=shared_outer,
                shared_lora_a_groups=groups,
            ),
            has_optimizer_state=False,
            weight_dtype=next(iter(self._weights.values())).dtype,
        )

    kind = AdapterSourceKind.PEFT

    def read_weights(self) -> dict[str, torch.Tensor]:
        return dict(self._weights)


class ExternalWeightsReader(PeftAdapterReader):
    """Official Tinker export (F32, Tinker names, stacked 3D experts)."""

    kind = AdapterSourceKind.EXTERNAL_WEIGHTS
    _map_name = staticmethod(map_tinker_keys_to_hf_keys)
