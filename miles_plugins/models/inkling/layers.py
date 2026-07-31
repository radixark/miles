"""Inkling transformer layers: attention (dual global/SWA + rel bias + sconv), router
(sigmoid + shared-expert joint renorm + global_scale), shared experts, MoE/dense MLP."""

from __future__ import annotations

import megatron.core.parallel_state as ps
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter

from miles_plugins.models.inkling.ops import (
    cp_all_gather,
    cp_world,
)
from miles_plugins.models.inkling.ops import create_block_mask as _create_block_mask
from miles_plugins.models.inkling.ops import flex_compiled as _flex
from miles_plugins.models.inkling.ops import (
    inkling_sconv_fp32_packed,
    inkling_sp_residual_conv,
    inkling_sum_fp32,
    inkling_swiglu_fp32,
    seqlens_from_packed,
)

_cp_world = cp_world
_cp_all_gather = cp_all_gather
_sp_residual_conv = inkling_sp_residual_conv
_seqlens_from_packed = seqlens_from_packed


class _Sconv(nn.Module):
    def __init__(self, channels, kernel, dtype, impl="triton", packed=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel, dtype=dtype))
        self.k = kernel
        self.impl = impl
        self.packed = packed

    def forward(self, x, seqlens=None):
        return inkling_sconv_fp32_packed(x, self.weight, seqlens, impl=self.impl, packed=self.packed)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        return make_sharded_tensors_for_checkpoint({"weight": self.weight}, prefix, sharded_offsets=sharded_offsets)


class InklingSelfAttention(SelfAttention):
    def __init__(self, config, submodules, layer_number, attn_mask_type=AttnMaskType.causal, **kw):
        super().__init__(config, submodules, layer_number, attn_mask_type=attn_mask_type, **kw)
        t = config.inkling
        self.is_local = (layer_number - 1) in t.local_layer_ids
        self.nh = t.swa_num_attention_heads if self.is_local else t.num_attention_heads
        self.nkv = t.swa_num_key_value_heads if self.is_local else t.num_key_value_heads
        self.hd = t.swa_head_dim if self.is_local else t.head_dim
        self.d_rel = t.d_rel
        self.rel_extent = t.sliding_window_size if self.is_local else t.rel_extent
        self.attn_backend = t.attn_backend
        self._bm_cache = {}
        tp = ps.get_tensor_model_parallel_world_size()
        self.nh_l = self.nh // tp
        self.nkv_l = max(1, self.nkv // tp)
        qkvr_out = self.nh * self.hd + 2 * self.nkv * self.hd + self.nh * self.d_rel
        self.linear_qkv = TELayerNormColumnParallelLinear(
            config.hidden_size,
            qkvr_out,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )
        self.linear_proj = TERowParallelLinear(
            self.nh * self.hd,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )
        self.q_norm = te.RMSNorm(self.hd, eps=t.rms_norm_eps, params_dtype=config.params_dtype)
        self.k_norm = te.RMSNorm(self.hd, eps=t.rms_norm_eps, params_dtype=config.params_dtype)
        self.rel_proj = nn.Parameter(
            torch.zeros(self.d_rel, self.rel_extent, dtype=config.params_dtype), requires_grad=False
        )
        self.window = (t.sliding_window_size - 1, 0) if self.is_local else (-1, -1)
        self.dpa = te.DotProductAttention(
            num_attention_heads=self.nh,
            kv_channels=self.hd,
            num_gqa_groups=self.nkv,
            attention_dropout=0.0,
            qkv_format="sbhd",
            softmax_scale=1.0 / self.hd,
            tp_size=tp,
            tp_group=ps.get_tensor_model_parallel_group(),
        )
        if t.use_sconv:
            dt = config.params_dtype
            sk = t.sconv_kernel_size
            sconv_kw = {"impl": t.sconv_impl, "packed": t.sconv_packed}
            self.k_sconv = _Sconv(self.nkv_l * self.hd, sk, dt, **sconv_kw)
            self.v_sconv = _Sconv(self.nkv_l * self.hd, sk, dt, **sconv_kw)
            for _w in (self.k_sconv.weight, self.v_sconv.weight):
                _w.tensor_model_parallel = True
                _w.partition_dim = 0
                _w.partition_stride = 1
            self.attn_sconv = _Sconv(config.hidden_size, sk, dt, **sconv_kw)
        else:
            self.k_sconv = self.v_sconv = self.attn_sconv = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=None,
        **kw,
    ):
        if hidden_states.dtype != self.config.params_dtype:
            hidden_states = hidden_states.to(self.config.params_dtype)
        sq = hidden_states.shape[0]
        x = hidden_states.reshape(sq, -1)
        qkvr, _ = self.linear_qkv(x)
        T = qkvr.shape[0]
        cp, cp_rank, cp_group = _cp_world()
        q, k, v, r = qkvr.split(
            [self.nh_l * self.hd, self.nkv_l * self.hd, self.nkv_l * self.hd, self.nh_l * self.d_rel], dim=-1
        )
        if cp > 1:
            k = _cp_all_gather(k, cp_group, cp)
            v = _cp_all_gather(v, cp_group, cp)
            T_full = k.shape[0]
            seqlens = _seqlens_from_packed(packed_seq_params, T_full)
            self.config.inkling._seqlens = seqlens
            if self.k_sconv is not None:
                k = self.k_sconv(k, seqlens)
                v = self.v_sconv(v, seqlens)
            q = self.q_norm(q.reshape(-1, self.hd)).reshape(T, self.nh_l, self.hd)
            k = self.k_norm(k.reshape(-1, self.hd)).reshape(T_full, self.nkv_l, self.hd)
            v = v.reshape(T_full, self.nkv_l, self.hd)
            r = r.reshape(T, self.nh_l, self.d_rel)
            if self.attn_backend == "flex":
                out = self._seg_cp_flex(q, k, v, r, seqlens, cp_rank * T, T_full)
            else:
                out = self._seg_cp(q, k, v, r, seqlens, cp_rank * T, T_full)
        else:
            seqlens = _seqlens_from_packed(packed_seq_params, T)
            self.config.inkling._seqlens = seqlens
            if self.k_sconv is not None:
                k = self.k_sconv(k, seqlens)
                v = self.v_sconv(v, seqlens)
            q = self.q_norm(q.reshape(-1, self.hd)).reshape(T, self.nh_l, self.hd)
            k = self.k_norm(k.reshape(-1, self.hd)).reshape(T, self.nkv_l, self.hd)
            v = v.reshape(T, self.nkv_l, self.hd)
            r = r.reshape(T, self.nh_l, self.d_rel)
            out = self._attend(q, k, v, r, seqlens)
        proj_out, proj_bias = self.linear_proj(out.unsqueeze(1))
        if self.attn_sconv is not None:
            proj_out = _sp_residual_conv(self.config, self.attn_sconv, proj_out, seqlens)
        return proj_out, proj_bias

    def _attend(self, q, k, v, r, seqlens):
        if self.attn_backend == "fa4":
            from miles_plugins.models.inkling.ops import inkling_fa4_attention as _fa4

            T, nh_l, hd = q.shape
            rel_logits = torch.einsum("thd,de->the", r.float(), self.rel_proj.float())
            out = _fa4(q, k, v, rel_logits, seqlens, self.window[0], self.is_local, 1.0 / self.hd, self.rel_extent)
            return out.reshape(T, nh_l * hd)
        if self.attn_backend == "flex":
            return self._seg_flex(q, k, v, r, seqlens)
        return self._seg(q, k, v, r, seqlens)

    def _seg(self, q, k, v, r, seqlens=None):
        T, nh_l, hd = q.shape
        rl = torch.einsum("thd,de->the", r.float(), self.rel_proj.float())
        RE = rl.shape[-1]
        qi = torch.arange(T, device=q.device).view(T, 1)
        ki = torch.arange(T, device=q.device).view(1, T)
        rd = qi - ki
        valid = (rd >= 0) & (rd < RE)
        idx = rd.clamp(0, RE - 1)
        rb = rl.permute(1, 0, 2)
        bias = (torch.gather(rb, 2, idx.unsqueeze(0).expand(nh_l, T, T)) * valid.unsqueeze(0)).unsqueeze(0)
        if seqlens is not None and len(seqlens) > 1:
            seg_id = torch.repeat_interleave(
                torch.arange(len(seqlens), device=q.device), torch.tensor(seqlens, device=q.device)
            )
            cross = (seg_id.view(T, 1) != seg_id.view(1, T)).view(1, 1, T, T)
            bias = bias.masked_fill(cross, -1e9)
        ctx = self.dpa(
            q.unsqueeze(1).contiguous(),
            k.unsqueeze(1).contiguous(),
            v.unsqueeze(1).contiguous(),
            attn_mask_type="causal",
            window_size=self.window,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=bias.to(q.dtype),
        )
        return ctx.reshape(T, nh_l * hd)

    def _seg_cp(self, q, k, v, r, seqlens, q_off, T_full):
        T_loc, nh_l, hd = q.shape
        rl = torch.einsum("thd,de->the", r.float(), self.rel_proj.float())
        RE = rl.shape[-1]
        qi = (q_off + torch.arange(T_loc, device=q.device)).view(T_loc, 1)
        ki = torch.arange(T_full, device=q.device).view(1, T_full)
        rd = qi - ki
        idx = rd.clamp(0, RE - 1)
        rb = rl.permute(1, 0, 2)
        bias = torch.gather(rb, 2, idx.unsqueeze(0).expand(nh_l, T_loc, T_full)) * ((rd >= 0) & (rd < RE)).unsqueeze(0)
        mask = rd < 0
        if self.is_local:
            mask = mask | (rd > self.window[0])
        if seqlens is not None and len(seqlens) > 1:
            seg = torch.repeat_interleave(
                torch.arange(len(seqlens), device=q.device), torch.tensor(seqlens, device=q.device)
            )
            mask = mask | (seg[q_off : q_off + T_loc].view(T_loc, 1) != seg.view(1, T_full))
        bias = bias.masked_fill(mask.unsqueeze(0), -1e9).unsqueeze(0).to(q.dtype)
        ctx = self.dpa(
            q.unsqueeze(1).contiguous(),
            k.unsqueeze(1).contiguous(),
            v.unsqueeze(1).contiguous(),
            attn_mask_type="no_mask",
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=bias,
        )
        return ctx.reshape(T_loc, nh_l * hd)

    def _seg_id(self, seqlens, T, device):
        if seqlens is None or len(seqlens) <= 1:
            return None
        return torch.repeat_interleave(torch.arange(len(seqlens), device=device), torch.tensor(seqlens, device=device))

    def _block_mask(self, mask_mod, q_len, kv_len, key, device):
        bm = self._bm_cache.get(key)
        if bm is None:
            if len(self._bm_cache) > 128:
                self._bm_cache.clear()
            bm = _create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
                _compile=True,
            )
            self._bm_cache[key] = bm
        return bm

    def _seg_flex(self, q, k, v, r, seqlens=None):
        T, nh_l, hd = q.shape
        RE, W, is_local = self.rel_extent, self.window[0], self.is_local
        rl = torch.einsum("thd,de->the", r.float(), self.rel_proj.float())
        rel_logits = rl.permute(1, 0, 2).contiguous()
        seg_id = self._seg_id(seqlens, T, q.device)

        def score_mod(score, b, h, q_idx, kv_idx):
            rd = q_idx - kv_idx
            idx = torch.clamp(rd, 0, RE - 1)
            bias = rel_logits[h, q_idx, idx]
            valid = (rd >= 0) & (rd < RE)
            return torch.where(valid, score + bias.to(score.dtype), score)

        def mask_mod(b, h, q_idx, kv_idx):
            m = q_idx >= kv_idx
            if seg_id is not None:
                m = m & (seg_id[q_idx] == seg_id[kv_idx])
            if is_local:
                m = m & (q_idx - kv_idx <= W)
            return m

        block_mask = self._block_mask(mask_mod, T, T, (T, tuple(seqlens) if seqlens else None, is_local), q.device)
        qf = q.permute(1, 0, 2).unsqueeze(0)
        kf = k.permute(1, 0, 2).unsqueeze(0)
        vf = v.permute(1, 0, 2).unsqueeze(0)
        out = _flex()(
            qf,
            kf,
            vf,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=1.0 / self.hd,
            enable_gqa=True,
        )
        return out.squeeze(0).permute(1, 0, 2).reshape(T, nh_l * hd)

    def _seg_cp_flex(self, q, k, v, r, seqlens, q_off, T_full):
        T_loc, nh_l, hd = q.shape
        RE, W, is_local = self.rel_extent, self.window[0], self.is_local
        rl = torch.einsum("thd,de->the", r.float(), self.rel_proj.float())
        rel_logits = rl.permute(1, 0, 2).contiguous()
        seg = self._seg_id(seqlens, T_full, q.device)

        def score_mod(score, b, h, q_idx, kv_idx):
            rd = (q_off + q_idx) - kv_idx
            idx = torch.clamp(rd, 0, RE - 1)
            bias = rel_logits[h, q_idx, idx]
            valid = (rd >= 0) & (rd < RE)
            return torch.where(valid, score + bias.to(score.dtype), score)

        def mask_mod(b, h, q_idx, kv_idx):
            gq = q_off + q_idx
            rd = gq - kv_idx
            m = rd >= 0
            if seg is not None:
                m = m & (seg[gq] == seg[kv_idx])
            if is_local:
                m = m & (rd <= W)
            return m

        block_mask = self._block_mask(
            mask_mod,
            T_loc,
            T_full,
            (T_loc, T_full, tuple(seqlens) if seqlens else None, is_local, q_off),
            q.device,
        )
        qf = q.permute(1, 0, 2).unsqueeze(0)
        kf = k.permute(1, 0, 2).unsqueeze(0)
        vf = v.permute(1, 0, 2).unsqueeze(0)
        out = _flex()(
            qf,
            kf,
            vf,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=1.0 / self.hd,
            enable_gqa=True,
        )
        return out.squeeze(0).permute(1, 0, 2).reshape(T_loc, nh_l * hd)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
        from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_world_size()
        qkey = f"{prefix}linear_qkv.weight"
        if qkey in sd:
            orig = sd[qkey]
            splits = [self.nh_l * self.hd, self.nkv_l * self.hd, self.nkv_l * self.hd, self.nh_l * self.d_rel]
            so = tuple(sharded_offsets)
            prepend = len(so)

            def _qkvr_build(
                key, t, replica_id, flattened_range, _s=splits, _so=so, _p=prepend, _tr=tp_rank, _ts=tp_size
            ):
                comps = torch.split(t, _s, dim=0)
                return [
                    ShardedTensor.from_rank_offsets(
                        f"{key}.{nm}", c.contiguous(), *_so, (_p, _tr, _ts), replica_id=replica_id, prepend_axis_num=_p
                    )
                    for nm, c in zip(("q", "k", "v", "r"), comps, strict=False)
                ]

            def _qkvr_merge(sub):
                return torch.cat(list(sub), dim=0)

            sd[qkey] = ShardedTensorFactory(
                orig.key, orig.data, _qkvr_build, _qkvr_merge, orig.replica_id, flattened_range=orig.flattened_range
            )
        if self.k_sconv is not None:
            for sub in ("k_sconv", "v_sconv"):
                key = f"{prefix}{sub}.weight"
                if key in sd:
                    sd[key] = make_tp_sharded_tensor_for_checkpoint(
                        getattr(self, sub).weight, key, tp_axis=0, prepend_offsets=sharded_offsets
                    )
        return sd


class InklingRouter(TopKRouter):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        H = self.config.hidden_size
        ns = self.config.inkling.n_shared_experts
        self.shared_gate = nn.Parameter(torch.zeros(ns, H, dtype=self.config.params_dtype))
        _gs_dtype = torch.float32
        self.global_scale = nn.Parameter(torch.ones(1, dtype=_gs_dtype))
        if _gs_dtype == torch.float32:
            self.global_scale._keep_fp32 = True
        self._cache_key = None
        self._cache = None

    def forward(self, input, padding_mask=None, input_ids=None):
        key = (input.data_ptr(), tuple(input.shape))
        if self._cache_key == key and self._cache is not None:
            probs, routing_map = self._cache
            self._cache_key = None
            self._cache = None
            return probs, routing_map

        self._maintain_float32_expert_bias()
        H = input.shape[-1]
        nr, topk = self.config.num_moe_experts, self.topk
        logits = self.gating(input).view(-1, nr).float()
        shared_logits = input.reshape(-1, H).float() @ self.shared_gate.float().t()
        score = logits.sigmoid() + self.expert_bias.float()
        from miles.utils.replay_base import routing_replay_manager

        _sel_topk = routing_replay_manager.get_topk_fn(lambda s, k: s.topk(k, dim=-1).indices, return_probs=False)
        topk_ids = _sel_topk(score, topk).long()
        sel_logits = logits.gather(-1, topk_ids)
        active = torch.cat([sel_logits, shared_logits], dim=-1)
        lp = F.logsigmoid(active)
        w = torch.softmax(lp, dim=-1) * float(self.config.inkling.route_scale) * self.global_scale.float()
        routed_w, shared_w = w[:, :topk], w[:, topk:]
        probs = torch.zeros_like(logits).scatter(-1, topk_ids, routed_w)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, topk_ids, True)
        self.config.inkling._shared_w = shared_w
        self._apply_expert_bias(routing_map, padding_mask)
        if self.config.moe_router_dtype in ("fp32", "fp64"):
            return probs, routing_map
        return probs.type_as(input), routing_map


class InklingSharedExperts(MegatronModule):
    def __init__(self, config, submodules=None, gate=False, pg_collection=None):
        super().__init__(config=config)
        ns = config.inkling.n_shared_experts
        inter = config.inkling.intermediate_size
        tp = pg_collection.tp if pg_collection is not None else None
        import copy as _copy

        shared_cfg = _copy.copy(config)
        shared_cfg.ffn_hidden_size = inter
        self.experts = nn.ModuleList(
            [MLP(config=shared_cfg, submodules=submodules, ffn_hidden_size=inter, tp_group=tp) for _ in range(ns)]
        )

    def forward(self, hidden_states):
        sw = self.config.inkling._shared_w
        s, b, h = hidden_states.shape
        sw_ = sw.view(s, b, -1)
        if self.config.sequence_parallel and ps.get_tensor_model_parallel_world_size() > 1:
            sw_ = gather_from_sequence_parallel_region(sw_)
        sw32 = sw_.float()
        ys = []
        for j, mlp in enumerate(self.experts):
            fc1_out, _ = mlp.linear_fc1(hidden_states)
            act = inkling_swiglu_fp32(fc1_out, per_token_scale=sw32[:, :, j : j + 1])
            yj, _ = mlp.linear_fc2(act)
            ys.append(yj)
        return inkling_sum_fp32(ys)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
        from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

        singleton = (metadata or {}).get("singleton_local_shards", False)
        for j in range(len(self.experts)):
            k1 = f"{prefix}experts.{j}.linear_fc1.weight"
            if k1 in sd:
                base = make_tp_sharded_tensor_for_checkpoint(
                    self.experts[j].linear_fc1.weight, k1, tp_axis=0, prepend_offsets=sharded_offsets
                )
                sd[k1] = apply_swiglu_sharded_factory(base, sharded_offsets, singleton)
            k2 = f"{prefix}experts.{j}.linear_fc2.weight"
            if k2 in sd:
                sd[k2] = make_tp_sharded_tensor_for_checkpoint(
                    self.experts[j].linear_fc2.weight, k2, tp_axis=1, prepend_offsets=sharded_offsets
                )
        return sd


class InklingMoELayer(MoELayer):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        t = self.config.inkling
        self.mlp_sconv = (
            _Sconv(
                self.config.hidden_size,
                t.sconv_kernel_size,
                self.config.params_dtype,
                impl=t.sconv_impl,
                packed=t.sconv_packed,
            )
            if t.use_sconv
            else None
        )

    def shared_experts_compute(self, hidden_states):
        if self.use_shared_expert and not self.shared_expert_overlap:
            probs, routing_map = self.router(hidden_states)
            self.router._cache_key = (hidden_states.data_ptr(), tuple(hidden_states.shape))
            self.router._cache = (probs, routing_map)
        return super().shared_experts_compute(hidden_states)

    def forward(self, hidden_states, *args, **kw):
        if hidden_states.dtype != self.config.params_dtype:
            hidden_states = hidden_states.to(self.config.params_dtype)
        out, bias = super().forward(hidden_states, *args, **kw)
        if self.mlp_sconv is not None:
            seqlens = getattr(self.config.inkling, "_seqlens", None)
            out = _sp_residual_conv(self.config, self.mlp_sconv, out, seqlens)
        return out, bias


class InklingDenseMLP(MLP):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        t = self.config.inkling
        self.mlp_sconv = (
            _Sconv(
                self.config.hidden_size,
                t.sconv_kernel_size,
                self.config.params_dtype,
                impl=t.sconv_impl,
                packed=t.sconv_packed,
            )
            if t.use_sconv
            else None
        )
        self.global_scale = (
            nn.Parameter(torch.ones(1, dtype=self.config.params_dtype))
            if getattr(t, "use_global_scale", False)
            else None
        )

    def forward(self, hidden_states, *args, **kw):
        if hidden_states.dtype != self.config.params_dtype:
            hidden_states = hidden_states.to(self.config.params_dtype)
        fc1_out, _ = self.linear_fc1(hidden_states)
        act = inkling_swiglu_fp32(fc1_out)
        out, bias = self.linear_fc2(act)
        if self.global_scale is not None:
            out = out * self.global_scale.to(out.dtype)
        if self.mlp_sconv is not None:
            seqlens = getattr(self.config.inkling, "_seqlens", None)
            out = _sp_residual_conv(self.config, self.mlp_sconv, out, seqlens)
        return out, bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.global_scale is not None:
            from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

            sd.update(
                make_sharded_tensors_for_checkpoint(
                    {"global_scale": self.global_scale}, prefix, sharded_offsets=sharded_offsets
                )
            )
        return sd
