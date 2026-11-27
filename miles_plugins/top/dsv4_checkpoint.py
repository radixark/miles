"""Stream a DeepSeek-V4 FP8 checkpoint (raw DeepSeek names) into TOP Megatron layers.

FP8 weights get BF16 master parameters (fp8_masters) behind serving's processed
representation; FP32 parameters (mHC, router bias, sinks) stay FP32 after the module's
BF16 cast. Every installed parameter keeps requires_grad.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from miles_plugins.top.fp8_forward import TrainableFP8Linear


class DSV4Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = json.loads((self.path / "config.json").read_text())

    def tensor(self, name):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as reader:
            return reader.get_tensor(name).cuda()

    def fp8_config(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        return Fp8Config.from_config(dict(self.config["quantization_config"]))

    def _set(self, param, value):
        if param.shape != value.shape:
            raise ValueError(
                f"[top] DeepSeek-V4 checkpoint tensor {tuple(value.shape)} does not fit parameter {tuple(param.shape)}"
            )
        param.data = value.to(param.device)
        param.requires_grad_(True)

    def install_hyper_connection(self, hc, prefix):
        scale = self.tensor(prefix + "_scale").float()
        self._set(hc.mapping_proj.weight, self.tensor(prefix + "_fn").float())
        self._set(hc.bias, self.tensor(prefix + "_base").float())
        for i, name in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
            self._set(
                getattr(hc, name), scale[i : i + 1].clone()
            )  # config-access-exempt: attribute selected at runtime from a table of names

    def install_layer(self, layer, index, ep_rank, ep_size, fp8_config):
        p = f"layers.{index}."
        self._set(layer.input_layernorm.weight, self.tensor(p + "attn_norm.weight"))
        self._set(layer.pre_mlp_layernorm.weight, self.tensor(p + "ffn_norm.weight"))
        self.install_hyper_connection(layer.self_attention_hyper_connection, p + "hc_attn")
        self.install_hyper_connection(layer.mlp_hyper_connection, p + "hc_ffn")
        layer.self_attention.install(self, p + "attn.", fp8_config)
        self.install_moe(layer.mlp, p + "ffn.", ep_rank, ep_size)

    def install_moe(self, mlp, p, ep_rank, ep_size):
        router = mlp.router
        self._set(router.weight, self.tensor(p + "gate.weight"))
        if router.is_hash_layer:
            router.tid2eid = self.tensor(p + "gate.tid2eid").to(torch.int32)
        else:
            router.expert_bias = self.tensor(p + "gate.bias").float()
        shared = mlp.shared_experts
        shared.linear_fc1 = TrainableFP8Linear(
            torch.cat([self.tensor(p + "shared_experts.w1.weight"), self.tensor(p + "shared_experts.w3.weight")]),
            torch.cat([self.tensor(p + "shared_experts.w1.scale"), self.tensor(p + "shared_experts.w3.scale")]),
        )
        shared.linear_fc2 = TrainableFP8Linear(
            self.tensor(p + "shared_experts.w2.weight"), self.tensor(p + "shared_experts.w2.scale")
        )
        n = mlp.config.num_moe_experts // ep_size
        ws13, ss13, ws2, ss2 = [], [], [], []
        for e in range(ep_rank * n, (ep_rank + 1) * n):
            ep = f"{p}experts.{e}."
            ws13.append(torch.cat([self.tensor(ep + "w1.weight"), self.tensor(ep + "w3.weight")]))
            ss13.append(torch.cat([self.tensor(ep + "w1.scale"), self.tensor(ep + "w3.scale")]))
            ws2.append(self.tensor(ep + "w2.weight"))
            ss2.append(self.tensor(ep + "w2.scale"))
        from miles_plugins.top.fp8_masters import install_expert_masters

        install_expert_masters(mlp.experts, ws13, ss13, ws2, ss2)

    def install_head(self, head):
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            self._set(
                getattr(head, name), self.tensor(name).float()
            )  # config-access-exempt: attribute selected at runtime from a table of names
