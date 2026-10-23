"""Map Kimi K3 HF checkpoint weights to TorchTitan parameters."""

import re

from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter


class KimiK3StateDictAdapter(MoEStateDictAdapter):
    def _hf_name(self, name):
        if name == "lm_head.weight":
            return "language_model.lm_head.weight"
        name = name.replace("tok_embeddings.", "embed_tokens.")
        name = name.replace(".moe.", ".block_sparse_moe.")
        name = name.replace(".self_attn.core.", ".self_attn.")
        return "language_model.model." + name

    def _titan_name(self, name):
        if name == "language_model.lm_head.weight":
            return "lm_head.weight"
        if not name.startswith("language_model.model."):
            return None
        name = name.removeprefix("language_model.model.")
        if name.endswith((".self_attn.A_log", ".self_attn.dt_bias")):
            name = name.replace(".self_attn.", ".self_attn.core.")
        return name.replace("embed_tokens.", "tok_embeddings.").replace(".block_sparse_moe.", ".moe.")

    def to_hf(self, state_dict):
        result = {}
        for name, weight in state_dict.items():
            if ".routed_experts.inner_experts." not in name:
                result[self._hf_name(name)] = weight
                continue
            layer = re.search(r"layers\.(\d+)", name).group(1)
            abstract = name.replace(f"layers.{layer}.", "layers.{}.")
            projection = name.rsplit(".", 1)[1].split("_")[0]
            hf_abstract = f"language_model.model.layers.{{}}.block_sparse_moe.experts.{{}}.{projection}.weight"
            if isinstance(weight, DTensor):
                self.grouped_expert_weight_placements[abstract] = weight.placements
                self.grouped_expert_weight_shape[abstract] = weight.shape
                self.grouped_expert_weight_mesh[abstract] = weight.device_mesh
                result.update(self._get_local_experts_weights(hf_abstract, abstract, layer, weight))
            else:
                for expert, tensor in enumerate(weight.unbind(0)):
                    result[hf_abstract.format(layer, expert)] = tensor
        return result

    def from_hf(self, hf_state_dict):
        result, pending = {}, {}
        for name, weight in hf_state_dict.items():
            match = re.fullmatch(
                r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", name
            )
            if match is None:
                target = self._titan_name(name)
                if target is not None:
                    result[target] = weight
                continue
            layer, expert, projection = match.groups()
            suffix = "w2_EDF" if projection == "w2" else projection + "_EFD"
            abstract = "layers.{}.moe.routed_experts.inner_experts." + suffix
            pending.setdefault(layer, {}).setdefault(abstract, {})[int(expert)] = weight
            if abstract in self.local_experts_indices:
                tensor = self._concatenate_expert_weights_dtensor(pending, abstract, layer)
            else:
                tensor = self._concatenate_expert_weights(pending, abstract, layer, self.model_config.num_experts)
            if tensor is not None:
                result[abstract.format(layer)] = tensor
        if pending:
            raise ValueError(f"Incomplete K3 expert groups: {list(pending)}")
        return result
