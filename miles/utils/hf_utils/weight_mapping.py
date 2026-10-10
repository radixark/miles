import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.models.auto.auto_factory import _get_model_class


@dataclass(frozen=True)
class HfWeightMapping:
    parameter_names: frozenset[str]
    conversions: tuple = ()

    @classmethod
    def from_config(cls, config):
        # VLMs may also register a CausalLM compatibility class that drops the vision/text namespace.
        for auto_model in (AutoModelForImageTextToText, AutoModelForCausalLM):
            if type(config) in auto_model._model_mapping:
                # HF's lazy mapping also matches remote-code configs by class name.
                config_class = _get_model_class(config, auto_model._model_mapping).config_class
                if type(config) is not config_class:
                    config = config_class.from_dict(config.to_dict())
                # Only structure is needed; never allocate or load base weights.
                with torch.random.fork_rng(devices=[]), torch.device("meta"):
                    model = auto_model.from_config(config, attn_implementation="eager")
                parameter_names = frozenset(
                    name for name, param in model.named_parameters(remove_duplicate=False) if param.ndim in (2, 3)
                )
                return cls(parameter_names, tuple(get_model_conversion_mapping(model, add_legacy=False)))
        # Custom HF implementations without native conversion rules retain their checkpoint namespace.
        return cls(frozenset())

    def model_parameter(self, checkpoint_name):
        if checkpoint_name.removesuffix(".weight") in self.parameter_names:
            checkpoint_name = checkpoint_name.removesuffix(".weight")
        if checkpoint_name in self.parameter_names:
            return checkpoint_name
        for conversion in self.conversions:
            if isinstance(conversion, WeightRenaming):
                checkpoint_name, _ = conversion.rename_source_key(checkpoint_name)
        for conversion in self.conversions:
            if isinstance(conversion, WeightConverter):
                target, source_pattern = conversion.rename_source_key(checkpoint_name)
                if source_pattern is not None:
                    assert (
                        len(conversion.target_patterns) == 1
                    ), f"HF target binding does not support one-to-many conversion of {checkpoint_name!r}"
                    return target
        return checkpoint_name


def get_checkpoint_weight_map(checkpoint_dir: str | Path) -> dict[str, str]:
    """Read tensor names from an index or safetensors metadata without loading weights."""
    source = Path(checkpoint_dir)
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            return json.load(index_file)["weight_map"]

    weight_map = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in weight_map:
                    raise ValueError(f"Duplicate tensor {name} in {weight_map[name]} and {shard.name}")
                weight_map[name] = shard.name
    if not weight_map:
        raise FileNotFoundError(f"No safetensors weights or index under {source}")
    return weight_map


def get_param_name_remap(config_path: str, weight_map: dict[str, str]) -> Callable[[str], str]:
    """Return the checkpoint-to-HF name mapping for a supported checkpoint."""
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if "DeepseekV4ForCausalLM" in config.get("architectures", []) and "embed.weight" in weight_map:
        # The model-specific mapper depends on SGLang's optional runtime dependencies.
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        # Native mtp.0.* tensors belong to SGLang's extra layer after the decoder;
        # is_nextn leaves ordinary decoder names on the same mapping path.
        return partial(
            DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format,
            is_nextn=True,
            num_hidden_layers=config["num_hidden_layers"],
        )
    return lambda name: name


def get_param_name_remap_for_checkpoint(checkpoint_dir: str | Path) -> Callable[[str], str]:
    """Resolve the checkpoint-to-HF name mapping straight from a checkpoint directory."""
    checkpoint = Path(checkpoint_dir)
    return get_param_name_remap(str(checkpoint / "config.json"), get_checkpoint_weight_map(checkpoint))
