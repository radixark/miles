"""HF <-> torchtitan names for Qwen3.8-Next.

Left out on purpose, so neither loading nor weight sync touches them: the PLE n-gram
table and its hash constants (frozen, read straight from the checkpoint by
``ple.py``), ``mtp.*`` (not trained; sglang keeps its own copy) and ``model.visual.*``
(not built; weight sync re-sends the checkpoint's own towers).

Unlike Qwen3.5's, this checkpoint stores routed experts in torchtitan's own layout:
``gate_up_proj`` is ``[E, 2F, D]`` and ``down_proj`` ``[E, D, F]``, so nothing is
transposed.
"""

import re
from typing import Any

import torch
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

_HF_LAYER = "model.language_model.layers.{}"

_GLOBAL = {
    "tok_embeddings.weight": "model.language_model.embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
    "norm.hc_norm": "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "norm.w_down": "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "norm.w_up": "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
}

# torchtitan suffix under layers.N. -> HF suffix under model.language_model.layers.N.
_PER_LAYER = {
    **{
        f"{tt_site}.{tt}": f"{hf_site}.{hf}"
        for tt_site, hf_site in (("attn_hc", "attn_hyper_connection"), ("mlp_hc", "mlp_hyper_connection"))
        for tt, hf in (
            ("hc_norm", "hc_norm.weight"),
            ("w_down", "input_mix_weight_down.weight"),
            ("w_up", "input_mix_weight_up.weight"),
            ("w_inject", "block_inject_weight.weight"),
        )
    },
    # full attention + QSA indexer
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "attn.index_qk_proj.weight": "self_attn.indexer.index_qk_proj.weight",
    "attn.index_q_norm": "self_attn.indexer.q_layernorm.weight",
    "attn.index_k_norm": "self_attn.indexer.k_layernorm.weight",
    # GatedDeltaNet; in_proj_qkv and conv1d are fused in HF, see _FUSED
    "attn.in_proj_z.weight": "linear_attn.in_proj_z.weight",
    "attn.in_proj_a.weight": "linear_attn.in_proj_a.weight",
    "attn.in_proj_b.weight": "linear_attn.in_proj_b.weight",
    "attn.A_log": "linear_attn.A_log",
    "attn.dt_bias": "linear_attn.dt_bias",
    "attn.norm.weight": "linear_attn.norm.weight",
    "attn.out_proj.weight": "linear_attn.out_proj.weight",
    # MoE; the fused routed gate_up is in _FUSED
    "moe.router.gate.weight": "mlp.gate.weight",
    "moe.routed_experts.inner_experts.w2_EDF": "mlp.experts.down_proj",
    "moe.shared_experts.w1.weight": "mlp.shared_expert.gate_proj.weight",
    "moe.shared_experts.w3.weight": "mlp.shared_expert.up_proj.weight",
    "moe.shared_experts.w2.weight": "mlp.shared_expert.down_proj.weight",
    "moe.shared_experts.gate.weight": "mlp.shared_expert_gate.weight",
    # PLE's trained part
    "ple.key_proj.weight": "ple.key_proj.weight",
    "ple.value_proj.weight": "ple.value_proj.weight",
    "ple.norm_key": "ple.norm_key.weight",
    "ple.norm_query": "ple.norm_query.weight",
    "ple.norm_conv": "ple.norm_conv.weight",
    "ple.conv1d": "ple.conv1d.weight",
}

# HF suffix -> the torchtitan suffixes it concatenates, along dim 0 unless noted
_FUSED = {
    "linear_attn.in_proj_qkv.weight": ("attn.in_proj_q.weight", "attn.in_proj_k.weight", "attn.in_proj_v.weight"),
    "linear_attn.conv1d.weight": ("attn.conv_q.weight", "attn.conv_k.weight", "attn.conv_v.weight"),
    "mlp.experts.gate_up_proj": ("moe.routed_experts.inner_experts.w1_EFD", "moe.routed_experts.inner_experts.w3_EFD"),
}
_FUSED_DIM = {"mlp.experts.gate_up_proj": 1}

_TT_LAYER_KEY = re.compile(r"layers\.(\d+)\.(.+)")
_HF_LAYER_KEY = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")


class Qwen38NextStateDictAdapter(StateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self._part_of = {part: (hf, i) for hf, parts in _FUSED.items() for i, part in enumerate(parts)}

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf: dict[str, Any] = {}
        fused: dict[tuple[str, str], dict[int, Any]] = {}
        for key, value in state_dict.items():
            if key in _GLOBAL:
                hf[_GLOBAL[key]] = value
                continue
            match = _TT_LAYER_KEY.fullmatch(key)
            if match is None:
                continue
            layer, suffix = match.groups()
            if suffix in _PER_LAYER:
                hf[f"{_HF_LAYER.format(layer)}.{_PER_LAYER[suffix]}"] = value
            elif suffix in self._part_of:
                hf_suffix, i = self._part_of[suffix]
                fused.setdefault((layer, hf_suffix), {})[i] = value

        for (layer, hf_suffix), parts in fused.items():
            pieces = [parts[i] for i in range(len(_FUSED[hf_suffix]))]
            hf[f"{_HF_LAYER.format(layer)}.{hf_suffix}"] = torch.cat(pieces, dim=_FUSED_DIM.get(hf_suffix, 0))
        return hf

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        from_global = {hf: tt for tt, hf in _GLOBAL.items()}
        from_layer = {hf: tt for tt, hf in _PER_LAYER.items()}
        tt: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key in from_global:
                tt[from_global[key]] = value
                continue
            match = _HF_LAYER_KEY.fullmatch(key)
            if match is None:
                continue
            layer, suffix = match.groups()
            if suffix in from_layer:
                tt[f"layers.{layer}.{from_layer[suffix]}"] = value
            elif suffix in _FUSED:
                for part, piece in zip(_FUSED[suffix], self._split(int(layer), suffix, value), strict=True):
                    tt[f"layers.{layer}.{part}"] = piece
        return tt

    def _split(self, layer: int, hf_suffix: str, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if hf_suffix == "mlp.experts.gate_up_proj":
            return value.chunk(2, dim=1)
        delta_net = self.model_config.layers[layer].delta_net
        key_dim, value_dim = delta_net.in_proj_q.out_features, delta_net.in_proj_v.out_features
        return value.split([key_dim, key_dim, value_dim], dim=0)
