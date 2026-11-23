"""Stream a GLM-5 (glm_moe_dsa) checkpoint into real Megatron GLM layers.

FP8 weights get BF16 masters behind serving's processed FP8 (fp8_masters), BF16 weights stay
trainable parameters, and the DSA indexer stays frozen: its selection is discrete and its scores
are discarded.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from miles_plugins.top.fp8_forward import AbsorbedFP8Weight, TrainableFP8Linear

_INDEXER = ("wq_b", "wk", "weights_proj", "k_norm")


class GLMCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.indexer_interleave = json.loads((self.path / "config.json").read_text()).get(
            "indexer_rope_interleave", False
        )
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, name):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name).cuda()
        if self.indexer_interleave:
            # CUDA roll does not support E4M3. Permute its one-byte payload
            # without dequantizing or changing any checkpoint values.
            is_fp8 = value.dtype == torch.float8_e4m3fn
            if is_fp8:
                value = value.view(torch.uint8)
            if name.endswith("indexer.wq_b.weight"):
                value = value.reshape(-1, 128, value.shape[-1]).roll(64, 1).reshape_as(value)
            elif name.endswith(("indexer.wk.weight", "indexer.k_norm.weight", "indexer.k_norm.bias")):
                value = value.roll(64, 0)
            if is_fp8:
                value = value.view(torch.float8_e4m3fn)
        return value

    def absorbed(self, prefix):
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        # Serving constructs absorbed MLA weights inside model.load_weights,
        # before the linear quantization hook requantizes payloads to UE8M0.
        # This branch must dequantize the original checkpoint representation.
        weight = self.tensor(prefix + ".weight")
        if weight.dtype != torch.float8_e4m3fn:
            return weight
        scale = self.tensor(prefix + ".weight_scale_inv")
        return block_quant_dequant(weight, scale, [128, 128], torch.bfloat16)

    def install_linear(self, parent, name, prefix, absorbed=False):
        old = getattr(parent, name)  # config-access-exempt: attribute selected at runtime from a table of names
        train = name not in _INDEXER
        raw = self.tensor(prefix + ".weight")
        if absorbed:
            if train and raw.dtype == torch.float8_e4m3fn:
                setattr(parent, name, AbsorbedFP8Weight(raw, self.tensor(prefix + ".weight_scale_inv")))
            else:
                old.weight = torch.nn.Parameter(self.absorbed(prefix), requires_grad=train)
            return
        if raw.dtype == torch.float8_e4m3fn:
            # Matched TP1: no row split; the master is the dequantized checkpoint. The frozen indexer keeps one
            # too (requires_grad off), so weight sync sees a BF16 weight it can quantize, not processed FP8.
            linear = TrainableFP8Linear(raw, self.tensor(prefix + ".weight_scale_inv"))
            linear.weight.requires_grad_(train)
            setattr(parent, name, linear)
            return
        old.weight = torch.nn.Parameter(raw, requires_grad=train)

    def install_mlp(self, mlp, prefix):
        # Requantize gate/up jointly exactly as the serving merged projection.
        gate = self.tensor(prefix + ".gate_proj.weight")
        up = self.tensor(prefix + ".up_proj.weight")
        joined = torch.cat([gate, up])
        if gate.dtype == torch.float8_e4m3fn:
            scale = torch.cat(
                [
                    self.tensor(prefix + ".gate_proj.weight_scale_inv"),
                    self.tensor(prefix + ".up_proj.weight_scale_inv"),
                ]
            )
            mlp.linear_fc1 = TrainableFP8Linear(joined, scale)
        else:
            mlp.linear_fc1.weight = torch.nn.Parameter(joined, requires_grad=True)
        self.install_linear(mlp, "linear_fc2", prefix + ".down_proj")

    def install_layer(self, layer, index, ep_rank, ep_size):
        prefix = f"model.layers.{index}"
        for local, remote in [
            ("input_layernorm", "input_layernorm"),
            ("pre_mlp_layernorm", "post_attention_layernorm"),
        ]:
            getattr(layer, local).weight.data.copy_(
                self.tensor(f"{prefix}.{remote}.weight")
            )  # config-access-exempt: attribute selected at runtime from a table of names
        attention = layer.self_attention
        mappings = {
            "linear_q_down_proj": "q_a_proj",
            "linear_q_up_proj": "q_b_proj",
            "linear_kv_down_proj": "kv_a_proj_with_mqa",
            "linear_kv_up_proj": "kv_b_proj",
            "linear_proj": "o_proj",
            "wq_b": "indexer.wq_b",
            "wk": "indexer.wk",
            "weights_proj": "indexer.weights_proj",
        }
        for local, remote in mappings.items():
            key = f"{prefix}.self_attn.{remote}.weight"
            if key not in self.index:
                if local in ("wq_b", "wk", "weights_proj") and attention.skip_topk:
                    continue
                raise KeyError(f"Missing required projection: {key}")
            self.install_linear(
                attention, local, f"{prefix}.self_attn.{remote}", absorbed=local == "linear_kv_up_proj"
            )
        for local, remote in [
            ("q_layernorm", "q_a_layernorm"),
            ("kv_layernorm", "kv_a_layernorm"),
            ("k_norm", "indexer.k_norm"),
        ]:
            for attr in ["weight", "bias"]:
                key = f"{prefix}.self_attn.{remote}.{attr}"
                if key in self.index:
                    value = self.tensor(key)
                    if local == "k_norm":
                        value = value.float()
                    getattr(getattr(attention, local), attr).data.copy_(
                        value
                    )  # config-access-exempt: attribute selected at runtime from a table of names
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):  # config-access-exempt: module variants differ in whether they define experts
            self.install_mlp(mlp, prefix + ".mlp")
        else:
            mlp.router.weight = torch.nn.Parameter(
                self.tensor(prefix + ".mlp.gate.weight").bfloat16(), requires_grad=True
            )
            mlp.router.expert_bias = self.tensor(prefix + ".mlp.gate.e_score_correction_bias").float()
            self.install_mlp(mlp.shared_experts, prefix + ".mlp.shared_experts")
            n = layer.config.num_moe_experts // ep_size
            ws13 = []
            ss13 = []
            ws2 = []
            ss2 = []
            for expert in range(ep_rank * n, (ep_rank + 1) * n):
                ep = f"{prefix}.mlp.experts.{expert}"
                ws13.append(torch.cat([self.tensor(ep + ".gate_proj.weight"), self.tensor(ep + ".up_proj.weight")]))
                ss13.append(
                    torch.cat(
                        [
                            self.tensor(ep + ".gate_proj.weight_scale_inv"),
                            self.tensor(ep + ".up_proj.weight_scale_inv"),
                        ]
                    )
                )
                ws2.append(self.tensor(ep + ".down_proj.weight"))
                ss2.append(self.tensor(ep + ".down_proj.weight_scale_inv"))
            from miles_plugins.top.fp8_masters import install_expert_masters

            install_expert_masters(mlp.experts, ws13, ss13, ws2, ss2)
        self._freeze_indexer(layer)

    @staticmethod
    def _freeze_indexer(layer):
        for name in _INDEXER:
            module = getattr(
                layer.self_attention, name, None
            )  # config-access-exempt: attribute selected at runtime from a table of names
            if module is not None:
                module.requires_grad_(False)
