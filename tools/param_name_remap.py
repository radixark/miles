import json
from collections.abc import Callable


def _is_deepseek_v4_native(config_path: str, weight_map: dict) -> bool:
    # Only native-format V4 checkpoints need the HF remap; re-saved HF-format V4
    # and V3/V3.2 checkpoints are already HF-named (embed.weight is the native marker).
    with open(config_path) as f:
        is_v4_arch = "DeepseekV4ForCausalLM" in json.load(f).get("architectures", [])
    return is_v4_arch and "embed.weight" in weight_map


def get_param_name_remap(config_path: str, weight_map: dict) -> Callable[[str], str]:
    # DeepSeek-V4 native checkpoints remap to HF names; everything else is identity.
    if _is_deepseek_v4_native(config_path, weight_map):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        return DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    return lambda name: name
