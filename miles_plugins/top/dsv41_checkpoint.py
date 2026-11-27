"""Stream the DeepSeek-V4.1 release checkpoint (raw DeepSeek names) into the TOP V4.1 Megatron block.

32x32 FP8 projections get BF16 masters behind serving's bytes (fp8_b32), MXFP4 experts get BF16 masters behind the
cast serving runs (dsv41_experts); FP32 parameters (mHC, router bias, sinks) stay FP32 after the module's BF16 cast.
"""

import torch

from miles_plugins.top.dsv4_checkpoint import DSV4Checkpoint
from miles_plugins.top.fp8_b32 import TrainableFP8B32Linear


class DSV41Checkpoint(DSV4Checkpoint):
    def fp8_config(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        return Fp8Config.from_config(dict(self.config["quantization_config"]))

    def _b32(self, prefix, *, name):
        return TrainableFP8B32Linear(
            self.tensor(prefix + ".weight"), self.tensor(prefix + ".scale").float(), name=name
        )

    def install_layer(self, layer, index, ep_rank, ep_size, fp8_config):
        p = f"layers.{index}."
        self._set(layer.input_layernorm.weight, self.tensor(p + "attn_norm.weight"))
        self._set(layer.pre_mlp_layernorm.weight, self.tensor(p + "ffn_norm.weight"))
        self.install_hyper_connection(layer.self_attention_hyper_connection, p + "hc_attn")
        self.install_hyper_connection(layer.mlp_hyper_connection, p + "hc_ffn")
        layer.self_attention.install(self, p + "attn.", fp8_config)
        if layer.engram is not None:
            self.install_engram(layer.engram, p + "engram.")
        self.install_moe(layer.mlp, p + "ffn.", ep_rank, ep_size)

    def install_engram(self, engram, p):
        engram.linear_wkv = self._b32(p + "wkv", name="engram_wkv")
        self._set(engram.q_weight, self.tensor(p + "q_weight").float())
        self._set(engram.k_weight, self.tensor(p + "k_weight").float())

    def install_moe(self, mlp, p, ep_rank, ep_size):
        from miles_plugins.top.dsv41_experts import install_expert_masters

        router = mlp.router
        self._set(router.weight, self.tensor(p + "gate.weight"))
        router.expert_bias = self.tensor(p + "gate.bias").float()
        if router.image_token_id is not None:
            router.expert_bias_vl = self.tensor(p + "gate.bias_vl").float()
        shared = mlp.shared_experts
        shared.linear_fc1 = TrainableFP8B32Linear(
            torch.cat([self.tensor(p + "shared_experts.w1.weight"), self.tensor(p + "shared_experts.w3.weight")]),
            torch.cat(
                [self.tensor(p + "shared_experts.w1.scale"), self.tensor(p + "shared_experts.w3.scale")]
            ).float(),
            name="shared_fc1",
        )
        shared.linear_fc2 = self._b32(p + "shared_experts.w2", name="shared_fc2")
        n = mlp.config.num_moe_experts // ep_size
        fp4_13, fp4_2 = [], []
        for e in range(ep_rank * n, (ep_rank + 1) * n):
            ep = f"{p}experts.{e}."
            pair = lambda w, ep=ep: (self.tensor(ep + w + ".weight"), self.tensor(ep + w + ".scale"))  # noqa: E731
            fp4_13.append((pair("w1"), pair("w3")))
            fp4_2.append(pair("w2"))
        install_expert_masters(mlp.experts, fp4_13, fp4_2)
