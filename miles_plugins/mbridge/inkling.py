import re

import torch

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


def _deinterleave_w13(w13: torch.Tensor) -> torch.Tensor:
    # HF interleaved fc1 [g0,u0,...] -> megatron [gates; ups]
    gate = w13[0::2]
    up = w13[1::2]
    return torch.cat([gate, up], dim=0)


@register_model("inkling_mm_model")
class InklingBridge(Qwen2MoEBridge):
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.llm.embed.weight",
        "embedding.embed_norm.weight": "model.llm.embed_norm.weight",
        "decoder.final_layernorm.weight": "model.llm.norm.weight",
        "output_layer.weight": "model.llm.unembed.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_qkv.layer_norm_weight": ["model.llm.layers.{layer_number}.attn_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.llm.layers.{layer_number}.attn.wq_du.weight",
            "model.llm.layers.{layer_number}.attn.wk_dv.weight",
            "model.llm.layers.{layer_number}.attn.wv_dv.weight",
            "model.llm.layers.{layer_number}.attn.wr_du.weight",
        ],
        "self_attention.linear_proj.weight": ["model.llm.layers.{layer_number}.attn.wo_ud.weight"],
        "self_attention.q_norm.weight": ["model.llm.layers.{layer_number}.attn.q_norm.weight"],
        "self_attention.k_norm.weight": ["model.llm.layers.{layer_number}.attn.k_norm.weight"],
        "self_attention.rel_proj": ["model.llm.layers.{layer_number}.attn.rel_logits_proj.proj"],
        "self_attention.k_sconv.weight": ["model.llm.layers.{layer_number}.attn.k_sconv.weight"],
        "self_attention.v_sconv.weight": ["model.llm.layers.{layer_number}.attn.v_sconv.weight"],
        # attn_sconv is layer-level (post linear_proj), NOT under .attn
        "self_attention.attn_sconv.weight": ["model.llm.layers.{layer_number}.attn_sconv.weight"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["model.llm.layers.{layer_number}.mlp_norm.weight"],
        "mlp.mlp_sconv.weight": ["model.llm.layers.{layer_number}.mlp_sconv.weight"],
        "mlp.router.weight": ["model.llm.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.shared_gate": ["model.llm.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.global_scale": ["model.llm.layers.{layer_number}.mlp.gate.global_scale"],
        "mlp.router.expert_bias": ["model.llm.layers.{layer_number}.mlp.gate.bias"],
        "mlp.experts.linear_fc1": ["model.llm.layers.{layer_number}.mlp.experts.w13_weight"],
        "mlp.experts.linear_fc2": ["model.llm.layers.{layer_number}.mlp.experts.w2_weight"],
        "mlp.linear_fc1.layer_norm_weight": ["model.llm.layers.{layer_number}.mlp_norm.weight"],
        "mlp.linear_fc1.weight": ["model.llm.layers.{layer_number}.mlp.w13_dn.weight"],
        "mlp.linear_fc2.weight": ["model.llm.layers.{layer_number}.mlp.w2_md.weight"],
        "mlp.global_scale": ["model.llm.layers.{layer_number}.mlp.global_scale"],
        "mlp.shared_experts.experts": [
            "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w13_weight",
            "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w2_weight",
        ],
    }

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": ("attention_dropout", 0.0),
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _get_text_config(self):
        if hasattr(self.hf_config, "text_config"):
            return self.hf_config.text_config
        return self.hf_config

    def _is_tied_word_embeddings(self) -> bool:
        return False

    def _adjust_mapping_for_shared_weights(self):
        pass

    def _get_hf_shared_weight_keys(self):
        return []

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        m = re.search(r"mlp\.shared_experts\.experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight", name)
        if m is not None:
            which = m.group(2)
            key = (
                "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w13_weight"
                if which == "linear_fc1"
                else "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w2_weight"
            )
            return [key.format(layer_number=layer_number)]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        # linear_qkv = cat([wq, wk, wv, wr], dim 0)
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_weights) == 4, f"qkvr expects 4 HF tensors, got {len(hf_weights)}"
            if getattr(self, "dtype", None) is not None:
                hf_weights = [w.to(self.dtype) if w.dtype != self.dtype else w for w in hf_weights]
            return torch.cat(hf_weights, dim=0).contiguous()

        # router.weight = gate.weight[:nr]; router.shared_gate = gate.weight[nr:nr+ns]
        if mcore_weights_name.endswith("mlp.router.weight"):
            assert len(hf_weights) == 1
            tc = self._get_text_config()
            nr = tc.n_routed_experts
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return w[:nr].contiguous()
        if mcore_weights_name.endswith("mlp.router.shared_gate"):
            assert len(hf_weights) == 1
            tc = self._get_text_config()
            nr = tc.n_routed_experts
            ns = tc.n_shared_experts
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return w[nr : nr + ns].contiguous()
        if mcore_weights_name.endswith("global_scale"):  # global_scale keeps source dtype (fp32, no cast)
            assert len(hf_weights) == 1
            return hf_weights[0].reshape(1).contiguous()
        if mcore_weights_name.endswith("mlp.router.expert_bias"):
            assert len(hf_weights) == 1
            return hf_weights[0].contiguous()

        # routed experts: index fused 3-D HF tensor by GLOBAL eid = ep_rank*num_local + local
        def _global_eid(name):
            nr = self._get_text_config().n_routed_experts
            return self.mpu.ep_rank * (nr // self.mpu.ep_size) + int(name.split("weight")[-1])

        if "mlp.experts.linear_fc1" in mcore_weights_name and len(hf_weights) == 1 and hf_weights[0].dim() == 3:
            return _deinterleave_w13(hf_weights[0][_global_eid(mcore_weights_name)]).contiguous()
        if "mlp.experts.linear_fc2" in mcore_weights_name and len(hf_weights) == 1 and hf_weights[0].dim() == 3:
            return hf_weights[0][_global_eid(mcore_weights_name)].contiguous()

        # shared experts: fused [ns, ...] HF tensor -> expert e (de-interleave fc1)
        sm = re.search(r"mlp\.shared_experts\.experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight", mcore_weights_name)
        if sm is not None and len(hf_weights) == 1 and hf_weights[0].dim() == 3:
            e = int(sm.group(1))
            which = sm.group(2)
            w = hf_weights[0][e]
            if which == "linear_fc1":
                w = _deinterleave_w13(w)
            return w.contiguous()

        # dense MLP fc1 <- w13_dn (de-interleave [g0,u0,..]->[gates;ups]); fc2 <- w2_md
        if mcore_weights_name.endswith("mlp.linear_fc1.weight") and len(hf_weights) == 1 and hf_weights[0].dim() == 2:
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return _deinterleave_w13(w).contiguous()
        if mcore_weights_name.endswith("mlp.linear_fc2.weight") and len(hf_weights) == 1 and hf_weights[0].dim() == 2:
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return w.contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_split_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: torch.Tensor,
        param: torch.Tensor,
        tp_split_size: int,
    ) -> list[torch.Tensor]:
        """Split fused linear_qkv into per-rank [q_i|k_i|v_i|r_i] blocks (base chunk(tp) scrambles them)."""
        if (
            tp_split_size > 1
            and "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            tc = self._get_text_config()
            # total = q(nh*hd) + 2*kv + r(nh*d_rel); derive kv from row count, not is_local
            nh = int(tc.num_attention_heads)
            hd = int(tc.head_dim)
            d_rel = int(getattr(tc, "d_rel", 16) or 16)
            q_rows = nh * hd
            r_rows = nh * d_rel
            total = mcore_weights.shape[0]
            kv_rows = (total - q_rows - r_rows) // 2
            nkv = kv_rows // hd if hd else 0
            assert (
                kv_rows > 0
                and (total - q_rows - r_rows) % 2 == 0
                and nkv * hd == kv_rows
                and q_rows + 2 * kv_rows + r_rows == total
            ), (
                f"qkv split {mcore_weights_name}: total={total} nh={nh} hd={hd} d_rel={d_rel} "
                f"q={q_rows} kv={kv_rows} r={r_rows} nkv={nkv}"
            )
            assert (
                nh % tp_split_size == 0 and nkv % tp_split_size == 0
            ), f"qkv tp={tp_split_size} must divide nh={nh} and nkv={nkv} for {mcore_weights_name}"
            q, k, v, r = mcore_weights.split([q_rows, kv_rows, kv_rows, r_rows], dim=0)
            qs, ks, vs, rs = (
                q.chunk(tp_split_size),
                k.chunk(tp_split_size),
                v.chunk(tp_split_size),
                r.chunk(tp_split_size),
            )
            return [torch.cat([qs[i], ks[i], vs[i], rs[i]], dim=0).contiguous() for i in range(tp_split_size)]
        return super()._weight_split_across_tp(mcore_weights_name, mcore_weights, param, tp_split_size)

    def _build_config(self):
        tc = self._get_text_config()
        return self._build_base_config(
            text_config_key="text_config" if hasattr(self.hf_config, "text_config") else None,
            use_cpu_initialization=False,
            num_moe_experts=tc.n_routed_experts,
            moe_router_topk=tc.num_experts_per_tok,
            moe_ffn_hidden_size=tc.intermediate_size,
            moe_shared_expert_intermediate_size=tc.intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            qk_layernorm=True,
            add_qkv_bias=False,
            add_bias_linear=False,
        )
