"""DeepSeek-V4.1 attention role delegated to SGLang (op ``dsv41_attention``).

The trainer runs serving's V4.1 attention program over a packed micro-batch as one multi-request prefill on its own
SGLang pool and backend (dsv41_runtime), calling what serving's MQALayer calls under TOP, in its order:
32x32 FP8 wq_a / wkv / wq_b / wo_b (fp8_b32_linear), SGLang's RMSNorm for q_norm, fused q RoPE, the fused kv
norm + RoPE + FP8 SWA store, the backend's TOP low-ratio path (KV-source compressor, index layers' selection and
candidates), the TOP sparse core, inverse RoPE and the batch-invariant BF16 wo_a.

The gradient is a PyTorch reference of the same math at the served weights, straight through to the BF16 masters, over
serving's selections (attach): the SWA window, and each compressed layer's request-local top-k as serving chose it. A KV-source layer's reference latent is
carried to the later layers of its ratio, so their gradients reach its compressor. The indexer is not trained.
"""

from types import SimpleNamespace

import torch
from megatron.core.transformer.module import MegatronModule

from miles_plugins.top.binding import CaptureTap, canon_dump

# Reference latents of the KV-source layers, keyed by compress ratio, for the later layers of this forward.
_SOURCE_LATENTS: dict[int, torch.Tensor] = {}


class _AbsorbedB32:
    """wo_a: served as BF16, the exact dequantization of its 32x32 FP8 bytes (at install the checkpoint's, after a
    sync the sync quantizer's); the master is the parameter the optimizer and the sync see."""

    def __init__(self, payload, scale):
        from miles_plugins.top.fp8_b32 import FP8B32Master

        self._fp8 = FP8B32Master(payload, scale)
        self.master = self._fp8.master

    def processed(self):
        return self._fp8.served()

    def jacobian_weight(self):
        return self._fp8.jacobian_weight()


class TopDSV41Attention(MegatronModule):
    def __init__(self, config, submodules=None, layer_number=1, pg_collection=None, **_):
        super().__init__(config=config)
        from miles_plugins.top.dsv4_runtime import dsv4_hf_config, freqs_cis_for

        if config.context_parallel_size > 1:
            raise NotImplementedError("[top] DeepSeek-V4.1 attention has no all-gather CP path")
        self.layer_id = layer_number - 1
        self.cfg = cfg = dsv4_hf_config()
        self.ratio = cfg.compress_ratios[self.layer_id]
        self.eps = cfg.rms_norm_eps
        self.freqs_cis = freqs_cis_for(cfg, self.ratio, torch.cuda.current_device())
        self.is_kv_source = self.layer_id in cfg.kv_source_layer_ids
        self.is_index_source = self.layer_id in cfg.index_source_layer_ids
        self.compressor = None
        self.indexer = None
        self.installed = False
        # Capture points of the per-layer VJP (top_vjp): the compressed blocks this layer attends to, and the KV
        # source's latent as a later layer of its ratio reads it (its gradient reaches the source's compressor).
        self.top_index_selection = torch.nn.Identity()
        self.top_source_latent = CaptureTap()

    def install(self, ck, prefix, fp8_config):
        """Checkpoint tensors: BF16 masters behind serving's bytes; SGLang's compressor and indexer modules."""
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor, DeepseekV41Indexer
        from sglang.srt.layers.layernorm import RMSNorm

        from miles_plugins.top.fp8_b32 import TrainableFP8B32Linear

        cfg, dev, t = self.cfg, torch.cuda.current_device(), ck.tensor
        q_norm = RMSNorm(cfg.q_lora_rank, eps=self.eps, weight_dtype=torch.bfloat16).to(dev)
        q_norm.weight.data.copy_(t(prefix + "q_norm.weight"))
        object.__setattr__(self, "q_norm", q_norm)
        self.q_norm_weight = torch.nn.Parameter(q_norm.weight.data)
        q_norm.weight = self.q_norm_weight
        self.kv_norm = torch.nn.Parameter(t(prefix + "kv_norm.weight"))
        self.attn_sink_param = torch.nn.Parameter(t(prefix + "attn_sink").float())
        pair = lambda name: (t(prefix + name + ".weight"), t(prefix + name + ".scale").float())  # noqa: E731
        self.wq_a_linear = TrainableFP8B32Linear(*pair("wq_a"), name="attn_wq_a")
        self.wq_b_linear = TrainableFP8B32Linear(*pair("wq_b"), name="attn_wq_b")
        self.wkv_linear = TrainableFP8B32Linear(*pair("wkv"), name="attn_wkv")
        self.wo_b_linear = TrainableFP8B32Linear(*pair("wo_b"), name="attn_wo_b")
        self._wo_a = _AbsorbedB32(*pair("wo_a"))
        self.wo_a_master = self._wo_a.master
        if self.is_kv_source:
            compressor = DeepseekV41Compressor(cfg.hidden_size, 512, self.ratio, self.eps, fused_compress=False).to(
                device=dev, dtype=torch.bfloat16
            )
            compressor.wkv.weight.data.copy_(t(prefix + "compressor.wkv.weight"))
            if self.ratio > 1:
                compressor.wgate.weight.data.copy_(t(prefix + "compressor.wgate.weight"))
            compressor.norm.weight.data = t(prefix + "compressor.norm.weight").to(torch.bfloat16)
            # Registered so the optimizer owns its parameters and the sync sends them.
            self.compressor = compressor
        if self.is_index_source:
            indexer = DeepseekV41Indexer(cfg, self.layer_id, 512, fp8_config, prefix="indexer").to(dev)
            indexer.wq_b.weight.data = t(prefix + "indexer.wq_b.weight")
            indexer.wq_b.weight_scale_inv.data = t(prefix + "indexer.wq_b.scale").float()
            indexer.wq_b.quant_method.process_weights_after_loading(indexer.wq_b)
            indexer.weights_proj.weight.data.copy_(t(prefix + "indexer.weights_proj.weight"))
            if indexer.owns_k:
                indexer.wk.weight.data.copy_(t(prefix + "indexer.wk.weight"))
                indexer.k_norm.weight.data = t(prefix + "indexer.k_norm.weight").to(torch.bfloat16)
            indexer.top = True
            indexer.requires_grad_(False)
            # Not a submodule: the indexer is not trained, and its FP8 wq_b is no Megatron parameter.
            object.__setattr__(self, "indexer", indexer)
        self.installed = True

    def forward(self, hidden_states, attention_mask=None, packed_seq_params=None, **_):
        from miles_plugins.top.dsv41_runtime import get_runtime

        if not self.installed:
            raise RuntimeError("[top] DeepSeek-V4.1 attention weights were not installed")
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise NotImplementedError("[top] DeepSeek-V4.1 attention requires packed THD requests")
        cu = packed_seq_params.cu_seqlens_q.tolist()
        lengths = [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]
        rt = get_runtime(self.cfg, lengths=lengths, device=hidden_states.device)
        s, b, width = hidden_states.shape
        local = hidden_states.reshape(s * b, width)
        n = sum(lengths)
        x = local[:n].contiguous()
        if self.layer_id == 0:
            _SOURCE_LATENTS.clear()
        # Every layer of a micro-batch shares one prefill's metadata and pools (prepare caches by lengths).
        fb = rt.prepare(lengths)
        # A later pipeline stage takes the earlier stages' runtime state before its first layer reads it.
        from miles_plugins.top.dsv41_pipeline import restore_runtime

        restore_runtime(rt)
        with torch.no_grad():
            out, topk = self._serving_forward(x.detach(), rt, fb)
        canon_dump("attn_out", out)
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            from miles_plugins.top.kernels.surrogate import attach, checkpointed, reference_mode

            # The KV-source latent is carried to later layers (and captured), so it is computed and read outside the
            # recomputed reference.
            latent = None
            if self.ratio:
                with reference_mode():
                    if self.is_kv_source:
                        _SOURCE_LATENTS[self.ratio] = self._compressor_reference(x, lengths)
                        latent = _SOURCE_LATENTS[self.ratio]
                    else:
                        latent = self.top_source_latent(_SOURCE_LATENTS[self.ratio])

            # Bound now: under pipeline parallelism the next micro-batch prepares the runtime before this backward.
            positions = rt.positions

            def reference(x, latent):
                with reference_mode():
                    return self._reference(x, lengths, positions, topk, latent)

            ref = checkpointed(reference, x, latent)
            out = attach(out, ref, f"attention_v41_c{self.ratio}")
        full = out.new_zeros((s * b, out.shape[-1]))
        full[:n] = out
        return full.view(s, b, -1), None

    def _serving_forward(self, x, rt, fb):
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope, fused_rope_inplace
        from sglang.srt.true_on_policy.dsv41_attention import wo_a
        from sglang.srt.true_on_policy.fp8.fp8_b32_linear import linear

        cfg, backend, positions = self.cfg, rt.backend, rt.positions
        heads, rd = cfg.num_attention_heads, cfg.qk_rope_head_dim
        q_lora = self.q_norm(linear(x, *self.wq_a_linear.processed()))
        q = linear(q_lora, *self.wq_b_linear.processed()).view(-1, heads, 512)
        q_out = torch.empty_like(q)
        fused_q_norm_rope(q, q_out, None, self.freqs_cis, positions)
        rt.pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=backend.get_swa_out_cache_loc(fb),
            kv=linear(x, *self.wkv_linear.processed()),
            kv_weight=self.kv_norm.detach(),
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )
        topk = None
        if self.ratio:
            layer = SimpleNamespace(
                layer_id=self.layer_id,
                compress_ratio=self.ratio,
                compressor=self.compressor,
                indexer=self.indexer,
                freqs_cis=self.freqs_cis,
                rope_head_dim=rd,
            )
            backend._top_low_ratio_sources(layer, x, q_lora, positions, fb)
            # Request-local positions of this layer's selection (its own, or the last index layer's of its ratio).
            topk = backend.forward_metadata.core_metadata.sparse_raw_indices(self.ratio)[: x.shape[0]].clone()
            topk = self.top_index_selection(topk)
        o = backend.forward(
            q=q_out,
            k=q_out,
            v=q_out,
            layer=SimpleNamespace(layer_id=self.layer_id, tp_q_head_num=heads, v_head_dim=512),
            forward_batch=fb,
            compress_ratio=self.ratio,
            attn_sink=self.attn_sink_param.detach(),
            save_kv_cache=False,
        )
        fused_rope_inplace(o[..., -rd:], None, self.freqs_cis, positions=positions, inverse=True)
        groups, rank = cfg.o_groups, cfg.o_lora_rank
        out = wo_a(o.reshape(o.shape[0], groups, -1), self._wo_a.processed().view(groups, rank, -1))
        return linear(out.flatten(1), *self.wo_b_linear.processed()), topk

    def _reference(self, x, lengths, positions, topk, latent):
        """The block in PyTorch at the served weights, straight through to the BF16 masters, over serving's selections
        and the KV source's reference ``latent`` (gradient only)."""
        from miles.kernels.attention.dsa import sparse_attention
        from miles_plugins.models.deepseek_v4.ops.thd_utils import compressed_cu_seqlens, get_window_topk_idxs_thd
        from miles_plugins.top.dsv4_attention import _rope
        from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant, rms_norm_reference

        cfg, eps, rd = self.cfg, self.eps, self.cfg.qk_rope_head_dim
        n = x.shape[0]
        freqs = self.freqs_cis[positions]
        q_lora = rms_norm_reference(self.wq_a_linear.reference(x), self.q_norm_weight, eps)
        q = self.wq_b_linear.reference(q_lora).view(n, cfg.num_attention_heads, -1)
        q = torch.cat([q[..., :-rd], _rope(q[..., -rd:], freqs)], -1)
        kvf = self.wkv_linear.reference(x).float()
        kvf = kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + eps) * self.kv_norm.float()
        kv = torch.cat([fp8_group_fake_quant(kvf[:, :-rd], 64), _rope(kvf[:, -rd:], freqs)], -1).to(x.dtype)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=x.device, dtype=torch.int32)
        idx = get_window_topk_idxs_thd(cu, window_size=cfg.sliding_window, total_tokens=n)[0]
        if self.ratio:
            kv = torch.cat([kv, latent])
            cu_c = compressed_cu_seqlens(cu, self.ratio)
            request = torch.repeat_interleave(
                torch.arange(len(lengths), device=x.device), torch.tensor(lengths, device=x.device)
            )
            idx = torch.cat([idx, torch.where(topk >= 0, topk + cu_c[request].unsqueeze(1) + n, -1)], -1)
        width = (idx.shape[-1] + 63) // 64 * 64
        idx = torch.nn.functional.pad(idx, (0, width - idx.shape[-1]), value=-1).int()
        o = sparse_attention(
            q[None], kv[None, :, None], idx[None, :, None], 512**-0.5, attn_sink=self.attn_sink_param
        )[0]
        o = torch.cat([o[..., :-rd], _rope(o[..., -rd:], freqs, inverse=True)], -1)
        groups, rank = cfg.o_groups, cfg.o_lora_rank
        wo_a = self._wo_a.jacobian_weight().view(groups, rank, -1)
        out = torch.einsum("tgd,grd->tgr", o.reshape(n, groups, -1), wo_a)
        return self.wo_b_linear.reference(out.flatten(1))

    def _compressor_reference(self, x, lengths):
        """V4.1's compressor: FP32 projections, softmax pooling of consecutive pairs (ratio 2) or the projection
        itself (ratio 1), FP32 RMSNorm, RoPE at the group start; the stored rounding as straight-through: serving's
        compressed-KV FP4 round trip (E2M1 per 16 with E4M3 scales, after RoPE), then the FP8 cache's latent."""
        from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_compressed_kv

        from miles_plugins.top.dsv4_attention import _rope
        from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant

        comp, ratio, d, rd = self.compressor, self.ratio, 512, self.cfg.qk_rope_head_dim
        out, start = [], 0
        for length in lengths:
            groups = length // ratio
            if groups:
                xs = x[start : start + groups * ratio]
                if ratio == 1:
                    c = (xs @ comp.wkv.weight.t()).float()
                else:
                    kv = (xs.float() @ comp.wkv.weight.float().t()).view(groups, ratio, d)
                    sc = (xs.float() @ comp.wgate.weight.float().t()).view(groups, ratio, d)
                    c = (torch.softmax(sc, dim=1) * kv).sum(1)
                c = c.to(x.dtype).float()
                c = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + comp.norm.eps) * comp.norm.weight.float()
                freqs = self.freqs_cis[torch.arange(groups, device=x.device) * ratio]
                c = torch.cat([c[:, :-rd].to(x.dtype), _rope(c[:, -rd:].to(x.dtype), freqs)], -1)
                c = c + (fake_quant_compressed_kv(c.detach()) - c.detach())
                out.append(torch.cat([fp8_group_fake_quant(c[:, :-rd], 64), c[:, -rd:].float()], -1).to(x.dtype))
            start += length
        return torch.cat(out) if out else x.new_zeros(0, d)
