"""Resolve checkpoint tensor names into the Hugging Face model namespace."""

import json
from collections.abc import Callable


def _is_deepseek_v4_native(config_path: str, weight_map: dict[str, str]) -> bool:
    """Return whether a DeepSeek V4 checkpoint uses release-native names."""
    with open(config_path, encoding="utf-8") as config_file:
        architectures = json.load(config_file).get("architectures", [])
    return "DeepseekV4ForCausalLM" in architectures and "embed.weight" in weight_map


def get_param_name_remap(config_path: str, weight_map: dict[str, str]) -> Callable[[str], str]:
    """Return the checkpoint-to-HF name mapping for a supported checkpoint."""
    if _is_deepseek_v4_native(config_path, weight_map):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        return DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    return lambda name: name
