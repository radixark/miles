"""Reset GatedDeltaNet recurrence + conv state at packed-document boundaries (FSDP packing).

GatedDeltaNet's linear-attn recurrence and causal_conv1d run over the whole packed row and bleed
across documents, inflating the train/rollout logprob gap. The decoder layer derives cu_seqlens/seq_idx
from position_ids and stashes them on its GDN submodule, which injects them into both kernels so each
document resets. Runs inside the gradient-checkpointed layer, so boundaries recompute identically on
backward. No-op outside THD packing.
"""

import functools
import logging

from ..adaptations.packing.boundaries import packed_seq_context

logger = logging.getLogger(__name__)


def _inject_kwarg(fn, key, value):
    """Wrap a kernel callable to default a kwarg (cu_seqlens / seq_idx) when unset."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if kwargs.get(key) is None:
            kwargs[key] = value
        return fn(*args, **kwargs)

    return wrapped


def _patch_gdn_forward(gdn_cls):
    orig = gdn_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    # rebind the kernel instance-attrs for the duration of the forward to inject per-doc boundaries
    _INJECT = (
        ("chunk_gated_delta_rule", "cu_seqlens"),
        ("recurrent_gated_delta_rule", "cu_seqlens"),
        ("causal_conv1d_fn", "seq_idx"),
    )

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        cu = getattr(self, "_gdn_cu_seqlens", None)
        si = getattr(self, "_gdn_seq_idx", None)
        if cu is None and si is None:
            return orig(self, *args, **kwargs)
        saved = {}
        for attr, key in _INJECT:
            value = cu if key == "cu_seqlens" else si
            fn = getattr(self, attr, None)
            if fn is not None and value is not None:
                saved[attr] = fn
                setattr(self, attr, _inject_kwarg(fn, key, value))
        try:
            return orig(self, *args, **kwargs)
        finally:
            for attr, fn in saved.items():
                setattr(self, attr, fn)

    forward._gdn_packing = True
    gdn_cls.forward = forward


def _patch_decoder_forward(dl_cls, gdn_cls):
    orig = dl_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        ctx = packed_seq_context(kwargs.get("position_ids"))
        # clear to None on non-packed forwards so stale boundaries can't leak into causal_conv1d
        cu = ctx.cu_seqlens if ctx is not None else None
        si = ctx.seq_idx if ctx is not None else None
        for module in self.modules():
            if isinstance(module, gdn_cls):
                module._gdn_cu_seqlens = cu
                module._gdn_seq_idx = si
        return orig(self, *args, **kwargs)

    forward._gdn_packing = True
    dl_cls.forward = forward


def _find_class(mod, suffix):
    for name in dir(mod):
        if name.endswith(suffix):
            return getattr(mod, name)
    return None


def apply_gateddeltanet_packing_patch():
    """Patch every GatedDeltaNet hybrid arch present (idempotent). Returns True if anything was patched."""
    patched = False
    for mod_name in ("qwen3_5_moe", "qwen3_next"):
        try:
            mod = __import__(f"transformers.models.{mod_name}.modeling_{mod_name}", fromlist=["x"])
        except Exception:
            continue
        gdn_cls = _find_class(mod, "GatedDeltaNet")
        dl_cls = _find_class(mod, "DecoderLayer")
        if gdn_cls is None or dl_cls is None:
            continue
        _patch_gdn_forward(gdn_cls)
        _patch_decoder_forward(dl_cls, gdn_cls)
        patched = True

    if patched:
        logger.info(
            "[fsdp] GatedDeltaNet packing fix applied: cu_seqlens/seq_idx reset the "
            "linear-attn recurrence and causal-conv state per packed document"
        )
    return patched
