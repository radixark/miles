"""DeepSeek-V4 attention role delegated to SGLang.

Replaces miles' DeepSeekV4Attention under TOP. The trainer runs the serving
attention program over a packed micro-batch as one multi-request prefill:
- FP8 projections through SGLang's TOP FP8 linear (wqkv_a, wq_b, wo_b);
- SGLang's RMSNorm for q_norm, fused per-head q norm + RoPE, fused kv norm +
  RoPE + FP8 SWA store into a private SGLang DS4 pool;
- SGLang's Compressor / C4Indexer modules and backend metadata (dsv4_runtime);
- the TOP sparse attention program (prefill == decode);
- inverse RoPE, per-token-group FP8 quant + DeepGEMM fp8_einsum for wo_a.
Weights are installed by DSV4Checkpoint: the FP8 projections get BF16 masters behind
serving's processed representation (fp8_masters), and
the block output carries the gradient of a PyTorch reference at the served weights (attach): Megatron-style
q/kv norms and RoPE, FP8 KV as straight-through, a reference compressor, and Miles'
differentiable sparse attention over the same selections serving used (SWA window,
C128 schedule, the C4 indexer's own top-k). The indexer is not trained.
"""

from types import SimpleNamespace

import torch
from megatron.core.transformer.module import MegatronModule


def wo_a_layout(payload, scale, *, o_lora_rank):
    """Serving's processed wo_a: the payload with grouped scales as _setup_fp8_wo_a_scales leaves them."""
    return payload, wo_a_scales(scale, groups=payload.shape[0] // o_lora_rank, rank=o_lora_rank, dim=payload.shape[1])


def _require_serving_fp8_wo_a():
    """The trainer runs an FP8 wo_a as the FP8 GEMM; serving otherwise dequantizes it to BF16 at load."""
    from sglang.srt.true_on_policy.dsv4 import fp8_wo_a_program

    if not fp8_wo_a_program():
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires the FP8 wo_a GEMM for an FP8 wo_a; set SGLANG_OPT_FP8_WO_A_GEMM=1 "
            "(explicitly on SM90) for serving and trainer"
        )


def wo_a_scales(scale, *, groups, rank, dim):
    """Blackwell: DeepGEMM's UE8M0 layout; Hopper: the raw FP32 blocks (serving's _FP8_WO_A_UE8M0 decides)."""
    from sglang.srt.models.deepseek_v4 import _FP8_WO_A_UE8M0

    _require_serving_fp8_wo_a()

    raw = scale.float().view(groups, rank // 128, dim // 128)
    if not _FP8_WO_A_UE8M0:
        return raw.contiguous()
    from deep_gemm import transform_sf_into_required_layout

    return transform_sf_into_required_layout(
        raw, mn=rank, k=dim, recipe=(1, 128, 128), num_groups=groups, is_sfa=False
    )


def quantize_wo_a_input(o):
    """Serving's wo_a activation quant and fp8_einsum recipe for this architecture."""
    from sglang.srt.models.deepseek_v4 import _FP8_WO_A_UE8M0

    if _FP8_WO_A_UE8M0:
        from sglang.kernels.ops.quantization.fp8_group_major_quant_ue8m0 import (
            sglang_per_token_group_quant_fp8_dsv4_wo_a,
        )

        return (*sglang_per_token_group_quant_fp8_dsv4_wo_a(o), (1, 1, 128))
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    rows, groups, dim = o.shape
    o_fp8, o_s = sglang_per_token_group_quant_fp8(
        o.reshape(rows * groups, dim).contiguous(), group_size=128, scale_ue8m0=False
    )
    return o_fp8.view(rows, groups, dim), o_s.view(rows, groups, -1), (1, 128, 128)


def _gather_rows(local, group):
    """Concatenate every CP rank's rows in rank order: Miles' all-gather CP chunks the packed stream that way."""
    out = local.new_empty((local.shape[0] * torch.distributed.get_world_size(group), *local.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, local.contiguous(), group=group)
    return out


def _gather_rows_differentiable(local, group):
    """_gather_rows whose backward reduce-scatters, summing each rank's gradient for the rows it owns."""
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(local.contiguous(), group=group))


class _Named(torch.nn.Module):
    """Holds parameters under one submodule name, so they take Megatron's stock paths (``q_layernorm.weight``)."""

    def __init__(self, **params):
        super().__init__()
        for name, param in params.items():
            setattr(self, name, param)


class _PlainWeight:
    """A BF16 weight read as is: the FP8Master interface without a processed form."""

    def __init__(self, param):
        self.master = param

    def processed(self):
        return self.master.detach(), None

    def jacobian_weight(self):
        return self.master


def _wo_a_value(processed, *, o_lora_rank):
    """The BF16 wo_a serving computes with: the payload's dequantization under its grouped scales."""
    from miles_plugins.top.fp8_forward import dequantize_processed

    payload, scale = processed
    return dequantize_processed(payload.view(-1, o_lora_rank, payload.shape[-1]), scale).view_as(payload)


class TopDSV4Attention(MegatronModule):
    def __init__(self, config, submodules=None, layer_number=1, pg_collection=None, **_):
        super().__init__(config=config)
        from miles_plugins.top.dsv4_runtime import dsv4_hf_config, freqs_cis_for

        # All-gather CP (Miles --allgather-cp): each rank holds one contiguous chunk of the micro-batch's packed
        # stream. Attention runs replicated on the gathered stream, exactly as at CP1; everything per-token stays local.
        self.cp_group = None
        if config.context_parallel_size > 1:
            if pg_collection is None:
                from megatron.core import parallel_state

                self.cp_group = parallel_state.get_context_parallel_group()
            else:
                self.cp_group = pg_collection.cp

        self.layer_id = layer_number - 1
        self.cfg = cfg = dsv4_hf_config()
        self.ratio = cfg.compress_ratios[self.layer_id]
        self.eps = cfg.rms_norm_eps
        self.freqs_cis = freqs_cis_for(cfg, self.ratio, torch.cuda.current_device())
        self.compressor = None
        self.indexer = None
        self.installed = False
        # Hands the C4 selection on unchanged; a module, so the dumper can capture it for the VJP's index replay.
        self.top_index_selection = torch.nn.Identity()

    def install(self, ck, prefix, fp8_config):
        """Load checkpoint tensors: BF16 masters behind serving's processed representation."""
        from sglang.srt.layers.attention.dsv4.compressor import Compressor
        from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
        from sglang.srt.layers.layernorm import RMSNorm

        cfg, dev = self.cfg, torch.cuda.current_device()
        t = ck.tensor
        q_norm = RMSNorm(cfg.q_lora_rank, eps=self.eps, weight_dtype=torch.bfloat16).to(dev)
        q_norm.weight.data.copy_(t(prefix + "q_norm.weight"))
        object.__setattr__(self, "q_norm", q_norm)
        if self.ratio:
            compressor = Compressor(
                cfg, self.layer_id, False, self.freqs_cis, compress_ratio=self.ratio, head_dim=512, rotate=False
            ).to(dev)
            _install_compressor(compressor, t, prefix + "compressor.")
            object.__setattr__(self, "compressor", compressor)
        if self.ratio == 4:
            indexer = C4Indexer(cfg, self.layer_id, self.freqs_cis, quant_config=fp8_config).to(dev)
            wq_b = indexer.wq_b
            wq_b.weight.data = t(prefix + "indexer.wq_b.weight")
            wq_b.weight_scale_inv.data = t(prefix + "indexer.wq_b.scale").float()
            wq_b.quant_method.process_weights_after_loading(wq_b)
            indexer.weights_proj.weight.data.copy_(t(prefix + "indexer.weights_proj.weight"))
            _install_compressor(indexer.compressor, t, prefix + "indexer.compressor.")
            object.__setattr__(self, "indexer", indexer)
        self._install_masters(t, prefix)
        self.installed = True

    def _install_masters(self, t, prefix):
        from miles_plugins.top.fp8_forward import TrainableFP8Linear
        from miles_plugins.top.fp8_masters import FP8Master

        # Serving fuses wq_a and wkv rows before the loader's UE8M0 requant.
        self.wqkv_a_linear = TrainableFP8Linear(
            torch.cat([t(prefix + "wq_a.weight"), t(prefix + "wkv.weight")]),
            torch.cat([t(prefix + "wq_a.scale"), t(prefix + "wkv.scale")]),
        )
        # The rest carry Megatron's stock DeepSeek-V4 names, so weight sync converts them with the stock converter.
        self.linear_q_up_proj = TrainableFP8Linear(t(prefix + "wq_b.weight"), t(prefix + "wq_b.scale"))
        self.linear_proj = TrainableFP8Linear(t(prefix + "wo_b.weight"), t(prefix + "wo_b.scale"))
        rank = self.cfg.o_lora_rank
        wo_a = t(prefix + "wo_a.weight")
        if wo_a.dtype == torch.bfloat16:
            # A BF16 wo_a checkpoint (no scale) is served by the BF16 absorb matmul; weight sync sends it
            # unquantized (dsv4_export, wo_a_weight).
            self.wo_a_weight = torch.nn.Parameter(wo_a)
            self._wo_a = _PlainWeight(self.wo_a_weight)
        else:
            # FP8 wo_a keeps checkpoint payloads (format_ue8m0 skips the requant) and serving's
            # _setup_fp8_wo_a_scales layout; it is served by fp8_einsum.
            self._wo_a = FP8Master(
                wo_a,
                t(prefix + "wo_a.scale"),
                lambda w, s: wo_a_layout(w, s, o_lora_rank=rank),
                lambda p: _wo_a_value(p, o_lora_rank=rank),
            )
            self.wo_a_master = self._wo_a.master
        self.q_layernorm = _Named(weight=torch.nn.Parameter(self.q_norm.weight.data))
        self.q_norm.weight = self.q_layernorm.weight
        self.kv_layernorm = _Named(weight=torch.nn.Parameter(t(prefix + "kv_norm.weight")))
        self.core_attention = _Named(attn_sink=torch.nn.Parameter(t(prefix + "attn_sink").float()))
        if self.compressor is not None:
            # Registered so the optimizer owns its parameters; the indexer stays frozen.
            self.compressor.requires_grad_(True)
            self.core_attention.add_module("compressor", self.compressor)

    def _processed(self):
        """Current FP8 weights, from the masters' caches."""
        return (
            self.wqkv_a_linear.processed(),
            self.linear_q_up_proj.processed(),
            self._wo_a.processed(),
            self.linear_proj.processed(),
        )

    def forward(self, hidden_states, attention_mask=None, packed_seq_params=None, **_):
        from miles_plugins.top.dsv4_runtime import get_runtime

        if not self.installed:
            raise RuntimeError("[top] DeepSeek-V4 attention weights were not installed")
        train = torch.is_grad_enabled() and hidden_states.requires_grad
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise NotImplementedError("[top] DeepSeek-V4 attention requires packed THD requests")
        cu = packed_seq_params.cu_seqlens_q.tolist()
        lengths = [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]
        cfg = self.cfg
        rt = get_runtime(cfg, lengths=lengths, device=hidden_states.device)
        s, b, width = hidden_states.shape
        local = hidden_states.reshape(s * b, width)
        if self.cp_group is None:
            x, own = local[: sum(lengths)].contiguous(), None
            fb = rt.prepare(lengths)
        else:
            # cu_seqlens are global (including Miles' pad segment), so the gathered stream is exactly sum(lengths).
            x = _gather_rows(local.detach(), self.cp_group)
            if x.shape[0] != sum(lengths):
                raise ValueError(f"[top] gathered {x.shape[0]} CP rows, global cu_seqlens cover {sum(lengths)}")
            rank = torch.distributed.get_rank(self.cp_group)
            own = (rank * s * b, (rank + 1) * s * b)
            fb = rt.prepare(lengths, own)
        with torch.no_grad():
            full, c4_topk = self._serving_forward(x.detach(), rt, fb, s * b, own)
        if c4_topk is not None:
            c4_topk = self.top_index_selection(c4_topk)
        self.last_c4_topk = c4_topk  # serving's C4 selection, for gradient references and replay checks
        if train:
            from miles_plugins.top.kernels.surrogate import attach, reference_mode

            if self.cp_group is None:
                xr, topk = x, c4_topk
            else:
                # The reference sees every row's keys and every query's selection; its gather is differentiable,
                # so each rank's own rows receive the gradient summed over every rank's queries.
                xr = _gather_rows_differentiable(local, self.cp_group)
                topk = None if c4_topk is None else _gather_rows(c4_topk, self.cp_group)
            with reference_mode():
                ref = self._reference(xr, lengths, rt.positions, topk)
                if own is None:
                    ref = full.new_zeros(full.shape).index_copy(0, torch.arange(x.shape[0], device=x.device), ref)
                else:
                    ref = ref[own[0] : own[1]]
            full = attach(full, ref, f"attention_c{self.ratio}")
        return full.view(s, b, -1), None

    def _serving_forward(self, x, rt, fb, rows, own=None):
        """Serving's attention over ``x``; with ``own`` = (start, stop) every KV-side write covers all of ``x`` and
        only those rows are queried (rows are independent, so they get the bytes the whole stream would)."""
        from deep_gemm import fp8_einsum
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope, fused_rope_inplace
        from sglang.srt.true_on_policy.fp8.fp8_linear import linear

        cfg = self.cfg
        backend, positions = rt.backend, rt.positions
        q_rows = slice(None) if own is None else slice(*own)
        wqkv_a, wq_b, (wo_a, wo_a_scale), wo_b = self._processed()
        qkv_a = linear(x, *wqkv_a)
        q_lora = self.q_norm(qkv_a[q_rows, : cfg.q_lora_rank])
        q = linear(q_lora, *wq_b).view(-1, cfg.num_attention_heads, 512)
        q_out = torch.empty_like(q)
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, rt.query_positions)
        rt.pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=backend.get_swa_out_cache_loc(fb),
            kv=qkv_a[..., cfg.q_lora_rank :],
            kv_weight=self.kv_layernorm.weight.detach(),
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )
        c4_topk = None
        if self.indexer is not None:
            if own is not None:
                # Index keys for the whole stream; the indexer itself then scores only the queried rows.
                backend.forward_indexer_compressor(
                    x=x, forward_batch=fb, layer_id=self.layer_id, compressor=self.indexer.compressor
                )
            self.indexer(
                x=x[q_rows], q_lora=q_lora, forward_batch=fb, attn_backend=backend, skip_compressor=own is not None
            )
            # Request-local logical blocks, shared by all layers: copy before the next layer.
            c4_topk = backend.forward_metadata.core_attn_metadata.c4_sparse_raw_indices[: q_lora.shape[0]].clone()
        if self.compressor is not None:
            backend.forward_core_compressor(x, fb, self.layer_id, self.compressor)
        layer = SimpleNamespace(layer_id=self.layer_id, tp_q_head_num=q_out.shape[1], v_head_dim=512)
        o = backend.forward(
            q=q_out,
            k=q_out,
            v=q_out,
            layer=layer,
            forward_batch=fb,
            compress_ratio=self.ratio,
            attn_sink=self.core_attention.attn_sink.detach(),
            save_kv_cache=False,
        )
        fused_rope_inplace(
            o[..., -cfg.qk_rope_head_dim :], None, self.freqs_cis, positions=rt.query_positions, inverse=True
        )
        groups, rank = cfg.o_groups, cfg.o_lora_rank
        o = o.reshape(o.shape[0], groups, -1)
        if wo_a_scale is None:
            from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant
            from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul

            # Serving's BI mode replaces torch.bmm process-wide; here only for this call (a global aten::bmm
            # override breaks torch.compile's pad_mm autotuning in DS4 backward graphs).
            bmm, torch.bmm = torch.bmm, bmm_batch_invariant
            try:
                out = _apply_wo_a_bf16_matmul(o, wo_a.view(groups, rank, -1), is_decode=False, is_prefill=True)
            finally:
                torch.bmm = bmm
        else:
            o_fp8, o_s, recipe = quantize_wo_a_input(o)
            out = torch.empty(o.shape[0], groups, rank, device=o.device, dtype=torch.bfloat16)
            fp8_einsum("bhr,hdr->bhd", (o_fp8, o_s), (wo_a.view(groups, rank, -1), wo_a_scale), out, recipe=recipe)
        y = linear(out.flatten(1), *wo_b)
        full = y.new_zeros((rows, y.shape[-1]))
        full[: y.shape[0]] = y
        return full, c4_topk

    def _reference(self, x, lengths, positions, c4_topk):
        """The block in PyTorch at the served weights, straight through to the BF16 masters, over serving's selections
        (gradient only)."""
        from miles.kernels.attention.dsa import sparse_attention
        from miles_plugins.models.deepseek_v4.ops.thd_utils import (
            compressed_cu_seqlens,
            get_compress_topk_idxs_thd,
            get_window_topk_idxs_thd,
        )
        from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant, rms_norm_reference

        cfg, eps, rd = self.cfg, self.eps, self.cfg.qk_rope_head_dim
        n = x.shape[0]
        freqs = self.freqs_cis[positions]
        qkv = self.wqkv_a_linear.reference(x)
        q = self.linear_q_up_proj.reference(
            rms_norm_reference(qkv[:, : cfg.q_lora_rank], self.q_layernorm.weight, eps)
        )
        q = q.view(n, cfg.num_attention_heads, -1)
        qf = q.float()
        q = (qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
        q = torch.cat([q[..., :-rd], _rope(q[..., -rd:], freqs)], -1)
        # Serving normalizes in FP32 and quantizes the nope dims to FP8 from FP32.
        kvf = qkv[:, cfg.q_lora_rank :].float()
        kvf = kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + eps) * self.kv_layernorm.weight.float()
        kv = torch.cat([fp8_group_fake_quant(kvf[:, :-rd], 64), _rope(kvf[:, -rd:], freqs)], -1).to(x.dtype)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=x.device, dtype=torch.int32)
        idx = get_window_topk_idxs_thd(cu, window_size=cfg.sliding_window, total_tokens=n)[0]
        if self.ratio:
            kv = torch.cat([kv, self._compressor_reference(x, lengths)])
            cu_c = compressed_cu_seqlens(cu, self.ratio)
            if self.ratio == 4:
                request = torch.repeat_interleave(
                    torch.arange(len(lengths), device=x.device), torch.tensor(lengths, device=x.device)
                )
                comp = torch.where(c4_topk >= 0, c4_topk + cu_c[request].unsqueeze(1) + n, -1)
            else:
                comp = get_compress_topk_idxs_thd(
                    cu,
                    cu_c,
                    ratio=self.ratio,
                    total_tokens=n,
                    max_n_compressed=max(lengths) // self.ratio,
                    kv_offset=n,
                )[0]
            idx = torch.cat([idx, comp], -1)
        width = (idx.shape[-1] + 63) // 64 * 64
        idx = torch.nn.functional.pad(idx, (0, width - idx.shape[-1]), value=-1).int()
        o = sparse_attention(
            q[None], kv[None, :, None], idx[None, :, None], 512**-0.5, attn_sink=self.core_attention.attn_sink
        )[0]
        o = torch.cat([o[..., :-rd], _rope(o[..., -rd:], freqs, inverse=True)], -1)
        groups, rank = cfg.o_groups, cfg.o_lora_rank
        wo_a = self._wo_a.jacobian_weight().view(groups, rank, -1)
        out = torch.einsum("tgd,grd->tgr", o.reshape(n, groups, -1), wo_a)
        return self.linear_proj.reference(out.flatten(1))

    def _compressor_reference(self, x, lengths):
        """SGLang's compressor math: gated softmax pooling (C4 overlaps the previous block),
        FP32 RMSNorm, RoPE at the block start, FP8 nope as straight-through."""
        from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant

        comp, ratio, d, rd = self.compressor, self.ratio, 512, self.cfg.qk_rope_head_dim
        weight, ape = comp.wkv_gate.weight, comp.ape.view(-1, d)
        out, start = [], 0
        for length in lengths:
            blocks = length // ratio
            if blocks:
                ks = x[start : start + blocks * ratio].float() @ weight.float().t()
                if ratio == 4:
                    kv_ov, kv, sc_ov, sc = (t.reshape(blocks, 4, d) for t in ks.split(d, -1))
                    prev_kv = torch.cat([kv_ov.new_zeros(1, 4, d), kv_ov[:-1]])
                    prev_sc = torch.cat([sc_ov.new_full((1, 4, d), torch.finfo(torch.float32).min), sc_ov[:-1]])
                    kv, sc = torch.cat([prev_kv, kv], 1), torch.cat([prev_sc, sc], 1)
                else:
                    kv, sc = (t.reshape(blocks, ratio, d) for t in ks.split(d, -1))
                c = (torch.softmax(sc + ape, dim=1) * kv).sum(1)
                c = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + comp.norm.variance_epsilon) * comp.norm.weight
                freqs = self.freqs_cis[torch.arange(blocks, device=x.device) * ratio]
                out.append(
                    torch.cat(
                        [
                            fp8_group_fake_quant(c[:, :-rd].to(x.dtype).float(), 64).to(x.dtype),
                            _rope(c[:, -rd:], freqs).to(x.dtype),
                        ],
                        -1,
                    )
                )
            start += length
        return torch.cat(out) if out else x.new_zeros(0, d)


def _rope(x, freqs, inverse=False):
    """Interleaved-pair RoPE, out of place: x [rows, ..., 2k], freqs complex [rows, k]."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    if inverse:
        freqs = freqs.conj()
    freqs = freqs.view(freqs.shape[0], *([1] * (xc.ndim - 2)), freqs.shape[-1])
    return torch.view_as_real(xc * freqs).flatten(-2).to(x.dtype)


def _install_compressor(module, tensor, prefix):
    module.load_ape_weight(module.ape, tensor(prefix + "ape").float())
    module.wkv_gate.weight.data.copy_(torch.cat([tensor(prefix + "wkv.weight"), tensor(prefix + "wgate.weight")]))
    module.norm.weight.data.copy_(tensor(prefix + "norm.weight").float())
