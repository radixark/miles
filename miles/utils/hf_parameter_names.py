"""Resolve checkpoint tensor names into the Hugging Face model namespace."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

from safetensors import safe_open


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
