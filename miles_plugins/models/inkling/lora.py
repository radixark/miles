from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_APPLIED_MODELS: list = []


def _make_adapter_cls():
    import torch.nn as nn

    class InklingLoRAAdapter(nn.Module):
        def __init__(self, kind: str, hf_prefix: str):
            super().__init__()
            self.kind = kind
            self.hf_prefix = hf_prefix
            self.load_meta: dict = {}

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            return {}

    return InklingLoRAAdapter


_ADAPTER_CLS = None


def _adapter_cls():
    global _ADAPTER_CLS
    if _ADAPTER_CLS is None:
        _ADAPTER_CLS = _make_adapter_cls()
    return _ADAPTER_CLS


def _rmsnorm(x, gamma, eps):
    """Recompute the RMSNorm fused into TELayerNormColumnParallelLinear (eager, fp32 internals)."""
    import torch

    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xn * gamma.float()).to(x.dtype)


def _new_param(ref_weight, shape, *, init: str, grad_sum_group: str | None, expert: bool):
    import torch
    import torch.nn as nn

    t = torch.empty(*shape, dtype=ref_weight.dtype, device=ref_weight.device)
    if init == "zero":
        t.zero_()
    else:
        if t.ndim == 2:
            nn.init.xavier_uniform_(t)
        else:
            for i in range(t.shape[0]):
                nn.init.xavier_uniform_(t[i])
    p = nn.Parameter(t)
    p.tensor_model_parallel = False
    p.partition_dim = -1
    p.partition_stride = 1
    if expert:
        p.allreduce = False
    if grad_sum_group is not None:
        p._lora_grad_sum_group = grad_sum_group
    return p


def _dropout(x, p, training):
    import torch.nn.functional as F

    if p and training:
        return F.dropout(x, p=p, training=True)
    return x


def apply_inkling_lora(model, args):
    """Attach Inkling LoRA to ONE built model chunk (before Float16Module / DDP wrapping)."""
    import torch
    import torch.nn.functional as F
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    from miles_plugins.models.inkling.inkling import InklingDenseMLP, InklingSelfAttention, InklingSharedExperts

    Adapter = _adapter_cls()

    rank = int(args.lora_rank)
    assert rank > 0, "apply_inkling_lora requires --lora-rank > 0"
    scale = float(args.lora_alpha) / float(rank)
    drop_p = float(getattr(args, "lora_dropout", 0.0) or 0.0)
    a_init = getattr(args, "lora_A_init_method", "xavier") or "xavier"

    cfg = model.config
    H = cfg.hidden_size
    eps = cfg.layernorm_epsilon
    sp = bool(cfg.sequence_parallel)
    tp = ps.get_tensor_model_parallel_world_size()
    ep_size = ps.get_expert_model_parallel_world_size()

    n_frozen = 0
    for p in model.parameters():
        p.requires_grad = False
        n_frozen += 1

    def _mk(ad, name, ref, shape, grad_sum=None, expert=False, init="zero"):
        ad.register_parameter(name, _new_param(ref, shape, init=init, grad_sum_group=grad_sum, expert=expert))

    def _gather_sp(x):
        return gather_from_sequence_parallel_region(x) if sp else x

    def _reduce_row(s):
        if tp <= 1:
            return s
        if sp:
            return reduce_scatter_to_sequence_parallel_region(s)
        return reduce_from_tensor_model_parallel_region(s)

    for layer in model.decoder.layers:
        lidx = layer.layer_number - 1
        hp = f"language_model.layers.{lidx}."

        attn = layer.self_attention
        assert isinstance(attn, InklingSelfAttention), f"layer {lidx}: unexpected attention {type(attn)}"
        ad = Adapter("attn", hp + "attn.")
        ref = attn.linear_qkv.weight
        _mk(ad, "wq_A", ref, (rank, H), grad_sum="tp", init=a_init)
        _mk(ad, "wq_B", ref, (attn.nh_l * attn.hd, rank))
        _mk(ad, "wk_A", ref, (rank, H), grad_sum="tp", init=a_init)
        _mk(ad, "wk_B", ref, (attn.nkv_l * attn.hd, rank))
        _mk(ad, "wv_A", ref, (rank, H), grad_sum="tp", init=a_init)
        _mk(ad, "wv_B", ref, (attn.nkv_l * attn.hd, rank))
        _mk(ad, "wr_A", ref, (rank, H), grad_sum="tp", init=a_init)
        _mk(ad, "wr_B", ref, (attn.nh_l * attn.d_rel, rank))
        _mk(ad, "wo_A", attn.linear_proj.weight, (rank, attn.nh_l * attn.hd), init=a_init)
        _mk(ad, "wo_B", attn.linear_proj.weight, (H, rank), grad_sum=("tp" if sp else None))
        ad.load_meta = dict(
            nh_l=attn.nh_l, nkv_l=attn.nkv_l, hd=attn.hd, d_rel=attn.d_rel, tp_rank=ps.get_tensor_model_parallel_rank()
        )
        attn.lora_adapter = ad

        qkv_mod = attn.linear_qkv
        orig_qkv = qkv_mod.forward

        def qkv_fwd(x, *a, _orig=orig_qkv, _m=qkv_mod, _ad=ad, **kw):
            out, bias = _orig(x, *a, **kw)
            xn = _rmsnorm(x, _m.layer_norm_weight, eps)
            xn = _dropout(_gather_sp(xn), drop_p, _m.training)
            s = F.linear(xn, torch.cat([_ad.wq_A, _ad.wk_A, _ad.wv_A, _ad.wr_A], 0))
            r = _ad.wq_A.shape[0]
            delta = torch.cat(
                [
                    F.linear(s[..., 0 * r : 1 * r], _ad.wq_B),
                    F.linear(s[..., 1 * r : 2 * r], _ad.wk_B),
                    F.linear(s[..., 2 * r : 3 * r], _ad.wv_B),
                    F.linear(s[..., 3 * r : 4 * r], _ad.wr_B),
                ],
                dim=-1,
            )
            return torch.add(out, delta, alpha=scale), bias

        qkv_mod.forward = qkv_fwd

        proj_mod = attn.linear_proj
        orig_proj = proj_mod.forward

        def proj_fwd(x, *a, _orig=orig_proj, _m=proj_mod, _ad=ad, **kw):
            out, bias = _orig(x, *a, **kw)
            s = F.linear(_dropout(x, drop_p, _m.training), _ad.wo_A)
            delta = F.linear(_reduce_row(s), _ad.wo_B)
            return torch.add(out, delta, alpha=scale), bias

        proj_mod.forward = proj_fwd

        mlp = layer.mlp
        if isinstance(mlp, InklingDenseMLP):
            dense_i = cfg.ffn_hidden_size
            i_loc = dense_i // tp
            ad = Adapter("dense_mlp", hp + "mlp.")
            ref1, ref2 = mlp.linear_fc1.weight, mlp.linear_fc2.weight
            _mk(ad, "fc1_A", ref1, (rank, H), grad_sum="tp", init=a_init)
            _mk(ad, "fc1_B", ref1, (2 * i_loc, rank))  # [gate_local; up_local] rows
            _mk(ad, "fc2_A", ref2, (rank, i_loc), init=a_init)
            _mk(ad, "fc2_B", ref2, (H, rank), grad_sum=("tp" if sp else None))
            ad.load_meta = dict(dense_i=dense_i, i_loc=i_loc, tp_rank=ps.get_tensor_model_parallel_rank())
            mlp.lora_adapter = ad

            fc1_mod = mlp.linear_fc1
            orig_fc1 = fc1_mod.forward

            def dense_fc1_fwd(x, *a, _orig=orig_fc1, _m=fc1_mod, _ad=ad, **kw):
                out, bias = _orig(x, *a, **kw)
                xn = _rmsnorm(x, _m.layer_norm_weight, eps)
                xn = _dropout(_gather_sp(xn), drop_p, _m.training)
                delta = F.linear(F.linear(xn, _ad.fc1_A), _ad.fc1_B)
                return torch.add(out, delta, alpha=scale), bias

            fc1_mod.forward = dense_fc1_fwd

            fc2_mod = mlp.linear_fc2
            orig_fc2 = fc2_mod.forward

            def dense_fc2_fwd(x, *a, _orig=orig_fc2, _m=fc2_mod, _ad=ad, **kw):
                out, bias = _orig(x, *a, **kw)
                s = F.linear(_dropout(x, drop_p, _m.training), _ad.fc2_A)
                delta = F.linear(_reduce_row(s), _ad.fc2_B)
                return torch.add(out, delta, alpha=scale), bias

            fc2_mod.forward = dense_fc2_fwd
        else:
            experts = mlp.experts  # TEGroupedMLP
            e_local = experts.num_local_experts
            moe_i = cfg.moe_ffn_hidden_size
            assert (getattr(cfg, "expert_tensor_parallel_size", 1) or 1) == 1, "Inkling LoRA assumes ETP=1"
            ad = Adapter("experts", hp + "mlp.experts.")
            ref1 = experts.linear_fc1.weight0
            ref2 = experts.linear_fc2.weight0
            is_ep = ep_size > 1
            _mk(ad, "w1_A", ref1, (rank, H), grad_sum=("ep" if is_ep else None), expert=is_ep, init=a_init)
            _mk(ad, "w3_A", ref1, (rank, H), grad_sum=("ep" if is_ep else None), expert=is_ep, init=a_init)
            _mk(ad, "w1_B", ref1, (e_local, moe_i, rank), expert=is_ep)
            _mk(ad, "w3_B", ref1, (e_local, moe_i, rank), expert=is_ep)
            _mk(ad, "w2_A", ref2, (e_local, rank, moe_i), expert=is_ep, init=a_init)
            _mk(ad, "w2_B", ref2, (H, rank), grad_sum=("ep" if is_ep else None), expert=is_ep)
            ad.load_meta = dict(e_local=e_local, moe_i=moe_i, ep_rank=ps.get_expert_model_parallel_rank())
            experts.lora_adapter = ad

            fc1_mod = experts.linear_fc1
            orig_efc1 = fc1_mod.forward

            def experts_fc1_fwd(x, m_splits, *a, _orig=orig_efc1, _m=fc1_mod, _ad=ad, **kw):
                out, bias = _orig(x, m_splits, *a, **kw)
                xd = _dropout(x, drop_p, _m.training)
                r = _ad.w1_A.shape[0]
                s13 = F.linear(xd, torch.cat([_ad.w1_A, _ad.w3_A], 0))  # [N, 2r]
                g = _grouped_B(s13[..., :r].contiguous(), _ad.w1_B, m_splits)  # [N, I]
                u = _grouped_B(s13[..., r:].contiguous(), _ad.w3_B, m_splits)
                delta = torch.cat([g, u], dim=-1)  # fc1 out layout = [gate_half | up_half]
                return torch.add(out, delta, alpha=scale), bias

            fc1_mod.forward = experts_fc1_fwd

            fc2_mod = experts.linear_fc2
            orig_efc2 = fc2_mod.forward

            def experts_fc2_fwd(x, m_splits, *a, _orig=orig_efc2, _m=fc2_mod, _ad=ad, **kw):
                out, bias = _orig(x, m_splits, *a, **kw)
                s = _grouped_B(_dropout(x, drop_p, _m.training), _ad.w2_A, m_splits)  # [N, r]
                delta = F.linear(s, _ad.w2_B)
                return torch.add(out, delta, alpha=scale), bias

            fc2_mod.forward = experts_fc2_fwd

            shared = mlp.shared_experts
            if shared is not None:
                assert isinstance(shared, InklingSharedExperts)
                ns = len(shared.experts)
                si_loc = moe_i // tp
                ad = Adapter("shared_experts", hp + "mlp.shared_experts.")
                ref1 = shared.experts[0].linear_fc1.weight
                ref2 = shared.experts[0].linear_fc2.weight
                _mk(ad, "w1_A", ref1, (rank, H), grad_sum="tp", init=a_init)
                _mk(ad, "w3_A", ref1, (rank, H), grad_sum="tp", init=a_init)
                _mk(ad, "w1_B", ref1, (ns, si_loc, rank))
                _mk(ad, "w3_B", ref1, (ns, si_loc, rank))
                _mk(ad, "w2_A", ref2, (ns, rank, si_loc), init=a_init)
                _mk(ad, "w2_B", ref2, (H, rank), grad_sum=("tp" if sp else None))
                ad.load_meta = dict(ns=ns, moe_i=moe_i, si_loc=si_loc, tp_rank=ps.get_tensor_model_parallel_rank())
                shared.lora_adapter = ad

                for j, sub in enumerate(shared.experts):
                    s_fc1 = sub.linear_fc1
                    orig_s1 = s_fc1.forward

                    def shared_fc1_fwd(x, *a, _orig=orig_s1, _m=s_fc1, _ad=ad, _j=j, **kw):
                        out, bias = _orig(x, *a, **kw)
                        xd = _dropout(_gather_sp(x), drop_p, _m.training)
                        d1 = F.linear(F.linear(xd, _ad.w1_A), _ad.w1_B[_j])
                        d3 = F.linear(F.linear(xd, _ad.w3_A), _ad.w3_B[_j])
                        delta = torch.cat([d1, d3], dim=-1)
                        return torch.add(out, delta, alpha=scale), bias

                    s_fc1.forward = shared_fc1_fwd

                    s_fc2 = sub.linear_fc2
                    orig_s2 = s_fc2.forward

                    def shared_fc2_fwd(x, *a, _orig=orig_s2, _m=s_fc2, _ad=ad, _j=j, **kw):
                        out, bias = _orig(x, *a, **kw)
                        s = F.linear(_dropout(x, drop_p, _m.training), _ad.w2_A[_j])
                        delta = F.linear(_reduce_row(s), _ad.w2_B)
                        return torch.add(out, delta, alpha=scale), bias

                    s_fc2.forward = shared_fc2_fwd

    if getattr(model, "post_process", False) and getattr(model, "output_layer", None) is not None:
        ol = model.output_layer
        vocab_local = ol.weight.shape[0]
        ad = Adapter("lm_head", "language_model.lm_head.")
        _mk(ad, "head_A", ol.weight, (rank, H), grad_sum="tp", init=a_init)
        _mk(ad, "head_B", ol.weight, (vocab_local, rank))
        ad.load_meta = dict(vocab_local=vocab_local, tp_rank=ps.get_tensor_model_parallel_rank())
        model.lora_lm_head_adapter = ad
        mup = getattr(cfg.inkling, "logits_mup_width_multiplier", None)
        mup = float(mup) if mup else None

        orig_ol = ol.forward

        def ol_fwd(x, *a, _orig=orig_ol, _m=ol, _ad=ad, _mup=mup, **kw):
            out, bias = _orig(x, *a, **kw)
            xin = x / _mup if _mup else x
            xin = _dropout(_gather_sp(xin), drop_p, _m.training)
            delta = F.linear(F.linear(xin, _ad.head_A), _ad.head_B)
            return torch.add(out, delta, alpha=scale), bias

        ol.forward = ol_fwd

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "[inkling-lora] applied: rank=%d alpha=%s scale=%.3f dropout=%s | trainable %d (%.4f%%) of %d params",
        rank,
        args.lora_alpha,
        scale,
        drop_p,
        n_train,
        100.0 * n_train / max(n_total, 1),
        n_total,
    )
    logger.info(
        "[inkling-lora] trainable params: %s / %s (%.4f%%)",
        f"{n_train:,}",
        f"{n_total:,}",
        100.0 * n_train / max(n_total, 1),
    )
    _APPLIED_MODELS.append(model)
    return model


def _grouped_B(s, B_stack, m_splits):
    """Per-local-expert matmul over the permuted token buffer, one grouped GEMM."""
    import torch
    import torch.nn.functional as F

    if s.is_cuda:
        offs = torch.as_tensor(list(m_splits), device=s.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        return F.grouped_mm(s, B_stack.transpose(1, 2), offs=offs)
    segs = torch.split(s, list(m_splits), dim=0)
    outs = [F.linear(seg, B_stack[e]) for e, seg in enumerate(segs)]
    return torch.cat(outs, dim=0)


def wrap_model_provider_with_inkling_lora(provider_func, args):
    """Wrap a miles model provider so every built chunk gets LoRA before DDP wrap."""

    def wrapped(*a, **kw):
        m = provider_func(*a, **kw)
        return apply_inkling_lora(m, args)

    return wrapped


def load_inkling_lora_adapter(model_chunks, adapter_path):
    """Load the Inkling HF-format LoRA release into the applied lora params (call AFTER load_checkpoint)."""
    import torch
    from safetensors import safe_open

    Adapter = _adapter_cls()

    path = f"{adapter_path}/adapter_model.safetensors"
    n_loaded = 0
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())

        def _get(name):
            assert name in keys, f"[inkling-lora] adapter tensor missing: {name}"
            return f.get_tensor(name)

        def _copy(param, tensor):
            nonlocal n_loaded
            assert (
                param.shape == tensor.shape
            ), f"[inkling-lora] shape mismatch: param {tuple(param.shape)} vs adapter slice {tuple(tensor.shape)}"
            with torch.no_grad():
                param.copy_(tensor.to(dtype=param.dtype, device=param.device))
            n_loaded += 1

        for chunk in model_chunks:
            mod = chunk
            while hasattr(mod, "module"):
                mod = mod.module
            for m in mod.modules():
                if not isinstance(m, Adapter):
                    continue
                hp, meta = m.hf_prefix, m.load_meta
                if m.kind == "attn":
                    t, nh_l, nkv_l, hd, dr = meta["tp_rank"], meta["nh_l"], meta["nkv_l"], meta["hd"], meta["d_rel"]
                    for proj, aname, bname, rows in (
                        ("wq_du", "wq_A", "wq_B", nh_l * hd),
                        ("wk_dv", "wk_A", "wk_B", nkv_l * hd),
                        ("wv_dv", "wv_A", "wv_B", nkv_l * hd),
                        ("wr_du", "wr_A", "wr_B", nh_l * dr),
                    ):
                        _copy(getattr(m, aname), _get(f"{hp}{proj}.lora_A.weight"))
                        B = _get(f"{hp}{proj}.lora_B.weight")
                        _copy(getattr(m, bname), B[t * rows : (t + 1) * rows])
                    A = _get(f"{hp}wo_ud.lora_A.weight")  # [r, nh*hd], column-shard
                    cols = nh_l * hd
                    _copy(m.wo_A, A[:, t * cols : (t + 1) * cols])
                    _copy(m.wo_B, _get(f"{hp}wo_ud.lora_B.weight"))
                elif m.kind == "dense_mlp":
                    t, dense_i, i_loc = meta["tp_rank"], meta["dense_i"], meta["i_loc"]
                    _copy(m.fc1_A, _get(f"{hp}gate_up_proj.lora_A.weight"))
                    B = _get(f"{hp}gate_up_proj.lora_B.weight")  # [2*dense_i, r] = [gate; up]
                    gate = B[:dense_i][t * i_loc : (t + 1) * i_loc]
                    up = B[dense_i:][t * i_loc : (t + 1) * i_loc]
                    _copy(m.fc1_B, torch.cat([gate, up], dim=0))
                    A2 = _get(f"{hp}down_proj.lora_A.weight")  # [r, dense_i]
                    _copy(m.fc2_A, A2[:, t * i_loc : (t + 1) * i_loc])
                    _copy(m.fc2_B, _get(f"{hp}down_proj.lora_B.weight"))
                elif m.kind == "experts":
                    e_local, ep_rank = meta["e_local"], meta["ep_rank"]
                    lo, hi = ep_rank * e_local, (ep_rank + 1) * e_local
                    _copy(m.w1_A, _get(f"{hp}w1.lora_A.weight").squeeze(0))
                    _copy(m.w3_A, _get(f"{hp}w3.lora_A.weight").squeeze(0))
                    _copy(m.w1_B, _get(f"{hp}w1.lora_B.weight")[lo:hi])
                    _copy(m.w3_B, _get(f"{hp}w3.lora_B.weight")[lo:hi])
                    _copy(m.w2_A, _get(f"{hp}w2.lora_A.weight")[lo:hi])
                    _copy(m.w2_B, _get(f"{hp}w2.lora_B.weight").squeeze(0))
                elif m.kind == "shared_experts":
                    t, ns, moe_i, si_loc = meta["tp_rank"], meta["ns"], meta["moe_i"], meta["si_loc"]
                    _copy(m.w1_A, _get(f"{hp}w1.lora_A.weight"))
                    _copy(m.w3_A, _get(f"{hp}w3.lora_A.weight"))
                    B1 = _get(f"{hp}w1.lora_B.weight")  # [ns*moe_i, r] expert-major
                    B3 = _get(f"{hp}w3.lora_B.weight")
                    _copy(
                        m.w1_B,
                        torch.stack([B1[j * moe_i + t * si_loc : j * moe_i + (t + 1) * si_loc] for j in range(ns)]),
                    )
                    _copy(
                        m.w3_B,
                        torch.stack([B3[j * moe_i + t * si_loc : j * moe_i + (t + 1) * si_loc] for j in range(ns)]),
                    )
                    A2 = _get(f"{hp}w2.lora_A.weight")  # [r, ns*moe_i] expert-major cols
                    _copy(
                        m.w2_A,
                        torch.stack([A2[:, j * moe_i + t * si_loc : j * moe_i + (t + 1) * si_loc] for j in range(ns)]),
                    )
                    _copy(m.w2_B, _get(f"{hp}w2.lora_B.weight"))
                elif m.kind == "lm_head":
                    t, vloc = meta["tp_rank"], meta["vocab_local"]
                    _copy(m.head_A, _get(f"{hp}lora_A.weight"))
                    _copy(m.head_B, _get(f"{hp}lora_B.weight")[t * vloc : (t + 1) * vloc])
                else:
                    raise ValueError(f"unknown adapter kind {m.kind}")
    logger.info("[inkling-lora] loaded %d lora tensors from %s", n_loaded, adapter_path)
    return n_loaded


class _GatherBatch:
    """Coalesce the export's per-tensor all_gathers into ONE flat all_gather per (tp|ep) group."""

    class _Tok:
        __slots__ = ("batch", "kind", "idx")

        def __init__(self, batch, kind, idx):
            self.batch, self.kind, self.idx = batch, kind, idx

        def get(self):
            return self.batch._resolved[self.kind][self.idx]

    def __init__(self):
        self._reqs = {"tp": [], "ep": []}  # (local, cat_dim)
        self._resolved = {"tp": None, "ep": None}

    def tp(self, local, dim):
        return self._add("tp", local, dim)

    def ep(self, local, dim):
        return self._add("ep", local, dim)

    def _add(self, kind, local, dim):
        self._reqs[kind].append((local, dim))
        return self._Tok(self, kind, len(self._reqs[kind]) - 1)

    def flush(self):
        import torch
        from megatron.core import parallel_state as ps

        groups = {
            "tp": (ps.get_tensor_model_parallel_group, ps.get_tensor_model_parallel_world_size),
            "ep": (ps.get_expert_model_parallel_group, ps.get_expert_model_parallel_world_size),
        }
        n_calls = 0
        for kind, reqs in self._reqs.items():
            if not reqs:
                self._resolved[kind] = []
                continue
            w = groups[kind][1]()
            if w == 1:
                self._resolved[kind] = [local for local, _ in reqs]
                continue
            assert len({local.dtype for local, _ in reqs}) == 1, "mixed adapter dtypes"
            flats = [local.detach().contiguous().reshape(-1) for local, _ in reqs]
            sizes = [f.numel() for f in flats]
            total = sum(sizes)
            flat_local = torch.cat(flats)
            gathered = flat_local.new_empty(w * total)
            torch.distributed.all_gather_into_tensor(gathered, flat_local, group=groups[kind][0]())
            per_rank = gathered.view(w, total)
            resolved = []
            off = 0
            for (local, dim), size in zip(reqs, sizes, strict=False):
                parts = [per_rank[r, off : off + size].view(local.shape) for r in range(w)]
                resolved.append(torch.cat(parts, dim=dim))
                off += size
            self._resolved[kind] = resolved
            n_calls += 1
        return n_calls


_UNPADDED_VOCAB_CACHE: list = []


def _hf_unpadded_vocab_size():
    """True (unpadded) vocab size from the HF config, or None if absent."""
    if not _UNPADDED_VOCAB_CACHE:
        val = None
        try:
            import json as _json
            import os as _os2

            from megatron.training import get_args as _get_args

            with open(_os2.path.join(_get_args().hf_checkpoint, "config.json"), encoding="utf-8") as f:
                cfg = _json.load(f)
            val = (cfg.get("text_config") or cfg).get("unpadded_vocab_size")
        except Exception:
            val = None
        _UNPADDED_VOCAB_CACHE.append(val)
    return _UNPADDED_VOCAB_CACHE[0]


def export_inkling_lora_hf_named(model_chunks):
    """Return (hf_name, full_tensor) for every applied lora param, gathered to full HF layout."""

    import time

    import torch

    t0 = time.perf_counter()
    Adapter = _adapter_cls()
    batch = _GatherBatch()
    plans: list = []

    def emit(name, t):
        plans.append((name, t))

    def emit_lazy(name, fn):
        plans.append((name, fn))

    for chunk in model_chunks:
        mod = chunk
        while hasattr(mod, "module"):
            mod = mod.module
        for m in mod.modules():
            if not isinstance(m, Adapter):
                continue
            hp = m.hf_prefix
            if m.kind == "attn":
                for hf, aten, bten in (
                    ("wq_du", m.wq_A, m.wq_B),
                    ("wk_dv", m.wk_A, m.wk_B),
                    ("wv_dv", m.wv_A, m.wv_B),
                    ("wr_du", m.wr_A, m.wr_B),
                ):
                    emit(f"{hp}{hf}.lora_A.weight", aten)  # replicated [r,H]
                    emit_lazy(f"{hp}{hf}.lora_B.weight", batch.tp(bten, 0).get)  # row-shard -> full
                emit_lazy(f"{hp}wo_ud.lora_A.weight", batch.tp(m.wo_A, 1).get)  # col-shard
                emit(f"{hp}wo_ud.lora_B.weight", m.wo_B)  # replicated
            elif m.kind == "dense_mlp":
                i_loc = m.load_meta["i_loc"]
                emit(f"{hp}gate_up_proj.lora_A.weight", m.fc1_A)
                gate_local, up_local = m.fc1_B[:i_loc], m.fc1_B[i_loc:]
                tg, tu = batch.tp(gate_local, 0), batch.tp(up_local, 0)
                emit_lazy(f"{hp}gate_up_proj.lora_B.weight", lambda g=tg, u=tu: torch.cat([g.get(), u.get()], dim=0))
                emit_lazy(f"{hp}down_proj.lora_A.weight", batch.tp(m.fc2_A, 1).get)
                emit(f"{hp}down_proj.lora_B.weight", m.fc2_B)
            elif m.kind == "experts":
                emit(f"{hp}w1.lora_A.weight", m.w1_A.unsqueeze(0))  # shared [1,r,H]
                emit(f"{hp}w3.lora_A.weight", m.w3_A.unsqueeze(0))
                emit_lazy(f"{hp}w1.lora_B.weight", batch.ep(m.w1_B, 0).get)  # [256,I,r]
                emit_lazy(f"{hp}w3.lora_B.weight", batch.ep(m.w3_B, 0).get)
                emit_lazy(f"{hp}w2.lora_A.weight", batch.ep(m.w2_A, 0).get)  # [256,r,I]
                emit(f"{hp}w2.lora_B.weight", m.w2_B.unsqueeze(0))  # shared [1,H,r]
            elif m.kind == "shared_experts":
                ns = m.load_meta["ns"]
                emit(f"{hp}w1.lora_A.weight", m.w1_A)
                emit(f"{hp}w3.lora_A.weight", m.w3_A)
                t1 = [batch.tp(m.w1_B[j], 0) for j in range(ns)]
                t3 = [batch.tp(m.w3_B[j], 0) for j in range(ns)]
                emit_lazy(f"{hp}w1.lora_B.weight", lambda ts=t1: torch.cat([t.get() for t in ts], dim=0))
                emit_lazy(f"{hp}w3.lora_B.weight", lambda ts=t3: torch.cat([t.get() for t in ts], dim=0))
                t2 = [batch.tp(m.w2_A[j], 1) for j in range(ns)]
                emit_lazy(f"{hp}w2.lora_A.weight", lambda ts=t2: torch.cat([t.get() for t in ts], dim=1))
                emit(f"{hp}w2.lora_B.weight", m.w2_B)
            elif m.kind == "lm_head":
                emit(f"{hp}lora_A.weight", m.head_A)
                tb = batch.tp(m.head_B, 0)

                def _head_b(tb=tb):
                    head_b = tb.get()
                    uv = _hf_unpadded_vocab_size()
                    if uv and uv < head_b.shape[0]:
                        head_b = head_b[:uv]
                    return head_b

                emit_lazy(f"{hp}lora_B.weight", _head_b)
            else:
                raise ValueError(f"unknown adapter kind {m.kind}")

    n_reqs = len(batch._reqs["tp"]) + len(batch._reqs["ep"])
    n_calls = batch.flush()
    out = [
        (name, (prod() if callable(prod) else prod).detach().to(torch.bfloat16).contiguous()) for name, prod in plans
    ]
    if torch.distributed.get_rank() == 0:
        ms = (time.perf_counter() - t0) * 1e3
        logger.info(
            "[inkling-lora] adapter export: %d tensors, %d gathers -> %d flat all_gathers, %.1f ms",
            len(out),
            n_reqs,
            n_calls,
            ms,
        )
    return out


_ATTN_PROJ = {"wq": "wq_du", "wk": "wk_dv", "wv": "wv_dv", "wr": "wr_du"}


def megatron_lora_to_hf_names(megatron_name: str) -> list[str]:
    """Map ONE megatron lora param (global name) to its HF adapter tensor name(s)."""
    name = re.sub(r"^(module\.)+", "", megatron_name)
    if name.startswith("lora_lm_head_adapter."):
        part = name.split(".")[-1]
        return [f"language_model.lm_head.lora_{part[-1]}.weight"]
    m = re.match(r"decoder\.layers\.(\d+)\.(.+)$", name)
    if not m:
        raise ValueError(f"not a Inkling lora param: {megatron_name}")
    lidx, rest = int(m.group(1)), m.group(2)
    hp = f"language_model.layers.{lidx}."
    am = re.match(r"self_attention\.lora_adapter\.(w[qkvro])_([AB])$", rest)
    if am:
        proj, ab = am.groups()
        hf_proj = "wo_ud" if proj == "wo" else _ATTN_PROJ[proj]
        return [f"{hp}attn.{hf_proj}.lora_{ab}.weight"]
    dm = re.match(r"mlp\.lora_adapter\.fc(\d)_([AB])$", rest)
    if dm:
        which, ab = dm.groups()
        hf_mod = "gate_up_proj" if which == "1" else "down_proj"
        return [f"{hp}mlp.{hf_mod}.lora_{ab}.weight"]
    em = re.match(r"mlp\.experts\.lora_adapter\.(w[123])_([AB])$", rest)
    if em:
        w, ab = em.groups()
        return [f"{hp}mlp.experts.{w}.lora_{ab}.weight"]
    sm = re.match(r"mlp\.shared_experts\.lora_adapter\.(w[123])_([AB])$", rest)
    if sm:
        w, ab = sm.groups()
        return [f"{hp}mlp.shared_experts.{w}.lora_{ab}.weight"]
    raise ValueError(f"not a Inkling lora param: {megatron_name}")


def hf_lora_to_megatron_name(hf_name: str) -> str:
    """Inverse of megatron_lora_to_hf_names."""
    if hf_name.startswith("language_model.lm_head.lora_"):
        ab = re.match(r"language_model\.lm_head\.lora_([AB])\.weight$", hf_name).group(1)
        return f"lora_lm_head_adapter.head_{ab}"
    m = re.match(r"language_model\.layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        raise ValueError(f"not a Inkling lora adapter tensor: {hf_name}")
    lidx, rest = int(m.group(1)), m.group(2)
    mp = f"decoder.layers.{lidx}."
    am = re.match(r"attn\.(wq_du|wk_dv|wv_dv|wr_du|wo_ud)\.lora_([AB])\.weight$", rest)
    if am:
        hf_proj, ab = am.groups()
        proj = "wo" if hf_proj == "wo_ud" else {v: k for k, v in _ATTN_PROJ.items()}[hf_proj]
        return f"{mp}self_attention.lora_adapter.{proj}_{ab}"
    dm = re.match(r"mlp\.(gate_up_proj|down_proj)\.lora_([AB])\.weight$", rest)
    if dm:
        hf_mod, ab = dm.groups()
        return f"{mp}mlp.lora_adapter.fc{'1' if hf_mod == 'gate_up_proj' else '2'}_{ab}"
    em = re.match(r"mlp\.experts\.(w[123])\.lora_([AB])\.weight$", rest)
    if em:
        w, ab = em.groups()
        return f"{mp}mlp.experts.lora_adapter.{w}_{ab}"
    sm = re.match(r"mlp\.shared_experts\.(w[123])\.lora_([AB])\.weight$", rest)
    if sm:
        w, ab = sm.groups()
        return f"{mp}mlp.shared_experts.lora_adapter.{w}_{ab}"
    raise ValueError(f"not a Inkling lora adapter tensor: {hf_name}")
