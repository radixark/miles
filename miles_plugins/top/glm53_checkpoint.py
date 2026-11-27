"""HF loading for the GLM-5.3-Flash TOP block: mbridge's glm5_next mapping, with TOP's per-expert parameters.

TOP's experts are a SequentialMLP (``mlp.experts.local_experts.{i}.linear_fc{1,2}.weight``); mbridge's mapping knows the
grouped form (``mlp.experts.linear_fc{1,2}.weight{expert}``), which it maps to HF ``mlp.experts.{expert}.*_proj``.
"""

import re

import torch
from megatron.core.utils import unwrap_model

from miles.utils.hf_utils.config import load_hf_config

_LOCAL_EXPERT = re.compile(r"\.mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight$")


def _bridge_cls():
    from miles_plugins.mbridge.glm5_next import Glm5NextBridge

    class _Glm53CheckpointBridge(Glm5NextBridge):
        def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep=True):
            names = super()._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
            per_rank = self.config.num_moe_experts // self.mpu.ep_size
            out = {}
            for local, name in names.items():
                match = _LOCAL_EXPERT.search(name)
                if match is not None:
                    expert = int(match.group(1)) + per_rank * self.mpu.ep_rank
                    name = f"{name[: match.start()]}.mlp.experts.{match.group(2)}.weight{expert}"
                out[local] = name
            return out

    return _Glm53CheckpointBridge


def load_glm53_hf_weights(models, checkpoint):
    models = unwrap_model(models)
    for model in models:
        if model.config.tensor_model_parallel_size != 1:
            raise NotImplementedError("[top] GLM-5.3-Flash checkpoint loading requires TP1")
    bridge = _bridge_cls()(load_hf_config(checkpoint), dtype=torch.bfloat16)
    with torch.no_grad():
        bridge.load_weights(models, checkpoint, memory_efficient=True)
