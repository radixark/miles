"""Identify an adapter source by its files and read the file-only kinds into HF-keyed tensors."""

import enum
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    ADAPTER_STATE_DIRNAME,
    AdapterState,
    _load_adapter_state,
    read_adapter_state_keys,
    read_adapter_state_weights,
)
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    HfKey,
    is_hf_only_name,
    is_tinker_only_name,
    map_named_tensors_to_hf_keys,
    map_sampler_keys_to_hf_keys,
    map_tinker_keys_to_hf_keys,
    rename_sampler_experts_to_hf,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    PARAMETRIZATION_FILENAME,
    AdapterParametrization,
    read_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError, require_fields
from miles.utils.lora.hf_lora_targets import HfLoraTargets

_PEFT_FILES = ("adapter_config.json", "adapter_model.safetensors")
# PEFT features that change what a tensor means; an adapter using them cannot be read as plain LoRA
_PEFT_CONFIG_MUST_BE_EMPTY = (
    "use_dora",
    "use_rslora",
    "modules_to_save",
    "rank_pattern",
    "alpha_pattern",
    "target_parameters",
    "fan_in_fan_out",
    "layers_to_transform",
    "layer_replication",
    "lora_bias",
    "loftq_config",
    "alora_invocation_tokens",
)
# initializations that rewrite the base model the adapter was trained against
_BASE_REWRITING_INITS = ("pissa", "olora", "corda", "loftq")


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

    parametrization: AdapterParametrization
    has_optimizer_state: bool
    weight_dtype: torch.dtype


def _has_peft_files(path: Path) -> bool:
    return all((path / f).is_file() for f in _PEFT_FILES)


def _safetensors_names(path: Path) -> list[str]:
    with safe_open(str(path / "adapter_model.safetensors"), "pt") as f:
        return list(f.keys())


def classify_adapter_source(path: Path, targets: HfLoraTargets, *, load_optimizer: bool) -> AdapterSourceKind:
    path = Path(path)
    if (path / PARAMETRIZATION_FILENAME).is_file():
        return AdapterSourceKind.ADAPTER_DIR
    if (path / ".metadata").is_file():
        return _dcp_source_kind(path)
    if any(path.glob("adapter_megatron_rank*.pt")) or any(path.glob("adapter_megatron_tp*_pp*.pt")):
        if not load_optimizer and _has_peft_files(path):
            return AdapterSourceKind.PEFT
        return AdapterSourceKind.LEGACY_RANK_SHARD
    if _has_peft_files(path):
        names = _safetensors_names(path)
        # official Tinker shares sglang's stacked expert names (experts.w1 / w3 / w2) with main's native export;
        # only its head and dropped VL level tell it apart
        if targets.tinker_name_renames and any(
            is_tinker_only_name(rename_sampler_experts_to_hf(n, targets), targets) for n in names
        ):
            hf_only = [n for n in names if is_hf_only_name(n, targets)]
            if hf_only:
                raise AdapterSourceError(f"{path} mixes official Tinker and HF names, e.g. {hf_only[:3]}")
            return AdapterSourceKind.EXTERNAL_WEIGHTS
        return AdapterSourceKind.PEFT
    if (path / "adapter_model.safetensors").is_file():
        raise AdapterSourceError(
            f"{path} has adapter_model.safetensors but no adapter_config.json, which the load needs to check the "
            "adapter's rank, alpha, target modules and PEFT features against the run; write it next to the weights "
            "(PEFT's save_pretrained writes both)"
        )
    raise AdapterSourceError(f"{path} is not a recognized LoRA adapter source")


def _dcp_source_kind(path: Path) -> AdapterSourceKind:
    if (path / "META.json").is_file():
        return AdapterSourceKind.LEGACY_TINKER_CHECKPOINT
    keys = list(dcp.FileSystemReader(path).read_metadata().state_dict_metadata)
    if (path / "run_config.yaml").is_file() and any(".adapter." in k for k in keys):
        return AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT
    if any(".lora_a." in k or ".lora_b." in k for k in keys):
        return AdapterSourceKind.TORCHTITAN_CHECKPOINT
    raise AdapterSourceError(f"{path} is a DCP checkpoint with neither Megatron-Bridge nor torchtitan LoRA keys")


def read_source_lora_alpha(path: Path) -> float | None:
    """The alpha a source records, read without the model; None for a source that records none (torchtitan)."""
    if (path / PARAMETRIZATION_FILENAME).is_file():
        return read_parametrization(path).lora_alpha
    # a DCP source's alpha is in the file of the kind it classifies as (a legacy Tinker dir may hold a stray
    # run_config.yaml)
    if (path / ".metadata").is_file():
        kind = _dcp_source_kind(path)
        if kind is AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT:
            peft = yaml.safe_load((path / "run_config.yaml").read_text())["peft"]
            assert peft is not None, f"{path} holds Megatron-Bridge adapter keys but its run_config.yaml has no peft"
            return float(peft["alpha"])
        if kind is AdapterSourceKind.TORCHTITAN_CHECKPOINT:
            return None
    for name in ("META.json", "adapter_config.json"):
        if (path / name).is_file():
            return json.loads((path / name).read_text())["lora_alpha"]
    return None


def collapse_shared_lora_a(
    weights: dict, parametrization: AdapterParametrization, fused_groups: list[tuple[str, ...]]
) -> dict:
    """Undo a written expansion: keep the first member's A, after checking every copy matches it.

    A re-fused HF module was read back as components holding A under the sorted first one only.
    """
    read_without_a = {member for fused in fused_groups for member in fused if member != min(fused)}
    for group in parametrization.shared_lora_a_groups:
        first = weights[f"{group[0]}.lora_A"]
        for member in group[1:]:
            if member in read_without_a:
                continue
            if not torch.equal(weights.pop(f"{member}.lora_A"), first):
                raise AdapterSourceError(f"{member}.lora_A differs inside a shared-A group")
    return weights


def _read_peft_tensors(
    path: Path, map_name, targets: HfLoraTargets
) -> tuple[dict[str, torch.Tensor], list[tuple[str, ...]]]:
    return map_named_tensors_to_hf_keys(load_file(str(path / "adapter_model.safetensors")).items(), map_name, targets)


class MilesAdapterDirReader:
    """A directory this library wrote: ``adapter_state/`` (training) or a sampler export (weights only)."""

    def __init__(self, path: Path, targets: HfLoraTargets):
        self.path, self.targets = Path(path), targets
        has_state = self.has_adapter_state()
        parametrization = read_parametrization(self.path)
        if not has_state:
            config = json.loads((self.path / "adapter_config.json").read_text())
            require_fields(config, ("r", "lora_alpha"), self.path / "adapter_config.json")
            # adapter_config.json is derived from the parametrization; a mismatch means the export was edited
            if (config["r"], config["lora_alpha"]) != (parametrization.lora_rank, parametrization.lora_alpha):
                raise AdapterSourceError(f"{self.path}: adapter_config.json disagrees with {PARAMETRIZATION_FILENAME}")
        self.source = AdapterSource(
            parametrization,
            has_optimizer_state=has_state and any(k.endswith(".exp_avg") for k in read_adapter_state_keys(self.path)),
            weight_dtype=torch.float32 if has_state else torch.bfloat16,
        )

    def has_adapter_state(self) -> bool:
        return (self.path / ADAPTER_STATE_DIRNAME).is_dir()

    def load_state(self, template: AdapterState) -> int:
        return _load_adapter_state(self.path, template)

    def read_weights(self) -> dict[str, torch.Tensor]:
        if self.has_adapter_state():
            return read_adapter_state_weights(self.path)
        weights, fused_groups = _read_peft_tensors(
            self.path, lambda name: map_sampler_keys_to_hf_keys(name, self.targets), self.targets
        )
        return collapse_shared_lora_a(weights, self.source.parametrization, fused_groups)


def _assert_plain_lora_peft_config(config: dict, path: Path) -> None:
    used = [k for k in _PEFT_CONFIG_MUST_BE_EMPTY if config.get(k)]  # absent and empty both mean unused
    init = config.get("init_lora_weights")
    if isinstance(init, str) and init.startswith(_BASE_REWRITING_INITS):
        used.append(f"init_lora_weights={init}")
    if used:
        raise AdapterSourceError(
            f"{path}: PEFT features {used} change what the tensors mean; only plain LoRA can be read"
        )


class PeftAdapterReader:
    """HF-named PEFT (HF Hub, Bridge's exports), or main's native export, which stacks experts under sglang's names
    (experts.w1 / w3 / w2); per-expert 2D tensors are stacked to 3D."""

    def __init__(self, path: Path, run: AdapterParametrization, targets: HfLoraTargets):
        self.path, self.targets = Path(path), targets
        config = json.loads((self.path / "adapter_config.json").read_text())
        if config.get("peft_type", "LORA") != "LORA":
            raise AdapterSourceError(f"{self.path}: {config['peft_type']} is not LoRA")
        require_fields(config, ("r", "lora_alpha"), self.path / "adapter_config.json")
        _assert_plain_lora_peft_config(config, self.path)
        self._weights, self._fused_groups = _read_peft_tensors(self.path, self._map_name, targets)
        modules = sorted({k.rsplit(".", 1)[0] for k in self._weights if k.endswith(".lora_B")})
        shared_outer = any(".experts." in k and v.dim() == 3 and v.shape[0] == 1 for k, v in self._weights.items())
        groups = tuple(self._fused_groups)
        # PEFT's base_model_name_or_path is a path or hub id, not the run's base name; base and LoRA flavour are the run's
        self.source = AdapterSource(
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

    def _map_name(self, name: str) -> HfKey:
        return map_sampler_keys_to_hf_keys(name, self.targets)

    def read_weights(self) -> dict[str, torch.Tensor]:
        return dict(self._weights)


class ExternalWeightsReader(PeftAdapterReader):
    """Official Tinker export (F32, Tinker names, stacked 3D experts)."""

    def _map_name(self, name: str) -> HfKey:
        return map_tinker_keys_to_hf_keys(name, self.targets)
