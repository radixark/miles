"""Reset NemotronH Mamba2 (conv+SSM) and attention state at packed-document boundaries (FSDP packing).

NemotronH is a Mamba2/attention hybrid; both are stateful and bleed across documents packed into one
forward row, which dominates its train/rollout logprob gap. We derive per-doc boundaries from
position_ids and feed seq_idx to the mixer's un-fused conv/scan kernels + run attention as varlen
flash-attn with cu_seqlens, so each doc stays isolated. Boundaries are stashed from the CausalLM
forward (position_ids don't reach the mixers otherwise). No-op when not packing.

The post-load fixup re-asserts checkpoint weights that transformers' Mamba ``_init_weights`` clobbers.
"""

import functools
import glob
import json
import logging
import os
import sys

import torch

from ..adaptations.packing.boundaries import packed_seq_context

logger = logging.getLogger(__name__)


def _reload_clobbered_from_disk(model, ckpt_path, tol=1e-3) -> int:
    """Reload params whose materialized value differs from the on-disk checkpoint by > ``tol`` (meta-device
    ranks skipped; they get the corrected value via the rank-0 broadcast). Returns the count re-asserted."""
    try:
        from safetensors import safe_open
    except Exception:  # pragma: no cover
        return 0
    files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not files:
        return 0
    index = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shard_of = json.load(f)["weight_map"]
    else:
        shard_of = {}

    reloaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                continue
            shards = [os.path.join(ckpt_path, shard_of[name])] if name in shard_of else files
            for f in shards:
                try:
                    with safe_open(f, framework="pt") as sf:
                        if name not in sf.keys():
                            continue
                        disk = sf.get_tensor(name)
                except Exception:
                    continue
                if disk.shape == param.shape:
                    disk = disk.to(param.dtype)
                    if (param.detach() - disk).abs().max().item() > tol:
                        param.copy_(disk)
                        reloaded += 1
                break
    if reloaded:
        logger.info(
            "[fsdp post_load] re-asserted %d checkpoint param(s) that from_pretrained clobbered "
            "post-load (Mamba _init_weights)",
            reloaded,
        )
    return reloaded


def _inject_seq_idx(fn, seq_idx):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        kwargs["seq_idx"] = seq_idx
        return fn(*args, **kwargs)

    return wrapped


def _patch_mixer_forward(mixer_cls):
    orig = mixer_cls.cuda_kernels_forward
    if getattr(orig, "_nemotron_packing", False):
        return

    @functools.wraps(orig)
    def cuda_kernels_forward(self, hidden_states, *args, **kwargs):
        seq_idx = getattr(self, "_packing_seq_idx", None)
        cache_params = args[0] if args else kwargs.get("cache_params")
        if seq_idx is None or cache_params is not None:
            return orig(self, hidden_states, *args, **kwargs)
        mod = sys.modules[mixer_cls.__module__]
        saved = {}
        for n in ("causal_conv1d_fn", "mamba_chunk_scan_combined"):
            fn = getattr(mod, n, None)
            if fn is not None:
                saved[n] = fn
                setattr(mod, n, _inject_seq_idx(fn, seq_idx))
        was_training = self.training
        self.training = False  # force the un-fused branch (the fused kernel can't take seq_idx correctly)
        try:
            return orig(self, hidden_states, *args, **kwargs)
        finally:
            self.training = was_training
            for n, fn in saved.items():
                setattr(mod, n, fn)

    cuda_kernels_forward._nemotron_packing = True
    mixer_cls.cuda_kernels_forward = cuda_kernels_forward


def _patch_attn_forward(attn_cls):
    orig = attn_cls.forward
    if getattr(orig, "_nemotron_packing", False):
        return

    mod = sys.modules[attn_cls.__module__]
    repeat_kv = getattr(mod, "repeat_kv", None)
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception:  # pragma: no cover
        flash_attn_varlen_func = None

    @functools.wraps(orig)
    def forward(self, hidden_states, *args, **kwargs):
        cu = getattr(self, "_packing_cu_seqlens", None)
        cache = kwargs.get("past_key_value", kwargs.get("cache_params"))
        if cu is None or cache is not None or flash_attn_varlen_func is None or repeat_kv is None:
            return orig(self, hidden_states, *args, **kwargs)
        b, q, _ = hidden_states.size()
        Q = self.q_proj(hidden_states).view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(b, q, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(b, q, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kk = repeat_kv(K, self.num_key_value_groups)
        vv = repeat_kv(V, self.num_key_value_groups)
        qf = Q.transpose(1, 2).reshape(q, self.num_heads, self.head_dim).contiguous()
        kf = kk.transpose(1, 2).reshape(q, self.num_heads, self.head_dim).contiguous()
        vf = vv.transpose(1, 2).reshape(q, self.num_heads, self.head_dim).contiguous()
        ml = self._packing_max_seqlen
        o = flash_attn_varlen_func(
            qf, kf, vf, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=ml, max_seqlen_k=ml, causal=True
        )
        o = o.reshape(b, q, self.num_heads * self.head_dim)
        return self.o_proj(o), None, kwargs.get("past_key_value")

    forward._nemotron_packing = True
    attn_cls.forward = forward


def _patch_causallm_forward(causallm_cls, mixer_cls, attn_cls):
    orig = causallm_cls.forward
    if getattr(orig, "_nemotron_packing", False):
        return
    import inspect

    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        try:
            position_ids = sig.bind(self, *args, **kwargs).arguments.get("position_ids")
        except TypeError:
            position_ids = kwargs.get("position_ids")
        ctx = packed_seq_context(position_ids)
        cu = ctx.cu_seqlens if ctx is not None else None
        si = ctx.seq_idx if ctx is not None else None
        ml = ctx.max_seqlen if ctx is not None else None
        for mod in self.modules():
            if isinstance(mod, mixer_cls):
                mod._packing_seq_idx = si
            elif attn_cls is not None and isinstance(mod, attn_cls):
                mod._packing_cu_seqlens = cu
                mod._packing_max_seqlen = ml
        return orig(self, *args, **kwargs)

    forward._nemotron_packing = True
    causallm_cls.forward = forward


def apply_nemotron_h_sglang_match_patch(model):
    """Reset NemotronH Mamba2 conv+SSM state AND attention at packed-document boundaries. Idempotent;
    no-op for non-NemotronH models and for single-document (unpacked) forwards."""
    mixer_cls = attn_cls = None
    for mod in model.modules():
        cn = type(mod).__name__
        if mixer_cls is None and cn.endswith("Mamba2Mixer") and hasattr(type(mod), "cuda_kernels_forward"):
            mixer_cls = type(mod)
        if attn_cls is None and getattr(mod, "block_type", None) == "attention" and hasattr(mod, "mixer"):
            attn_cls = type(mod.mixer)
    if mixer_cls is None:
        return False
    _patch_mixer_forward(mixer_cls)
    if attn_cls is not None:
        _patch_attn_forward(attn_cls)
    _patch_causallm_forward(type(model), mixer_cls, attn_cls)
    logger.info(
        "[fsdp] NemotronH packed-doc reset applied (mamba seq_idx + attention varlen cu_seqlens); " "mixer=%s attn=%s",
        mixer_cls.__module__,
        attn_cls.__name__ if attn_cls else None,
    )
    return True
