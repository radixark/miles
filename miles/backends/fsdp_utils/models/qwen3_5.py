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
from ..kernels.presets import SLOT_CAUSAL_CONV1D, SLOT_GATED_DELTA_RULE

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
        for module in self.modules():
            if isinstance(module, gdn_cls):
                module._gdn_cu_seqlens = ctx.cu_seqlens if ctx is not None else None
                module._gdn_seq_idx = ctx.seq_idx if ctx is not None else None
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
    for mod_name in ("qwen3_5", "qwen3_5_moe", "qwen3_next"):
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


# Per slot: the GatedDeltaNet instance attribute -> the function name in that Hub build. The names
# differ on the recurrent kernel, which transformers stores without the `fused_` prefix.
_GDN_KERNEL_SLOTS = (
    (
        SLOT_GATED_DELTA_RULE,
        {
            "chunk_gated_delta_rule": "chunk_gated_delta_rule",
            "recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
        },
    ),
    (
        SLOT_CAUSAL_CONV1D,
        {
            "causal_conv1d_fn": "causal_conv1d_fn",
            "causal_conv1d_update": "causal_conv1d_update",
        },
    ),
)


def bind_gated_deltanet_hub_kernels(model, args) -> int:
    """Point each GatedDeltaNet's kernel handles at the Hub builds. Returns modules patched.

    ``_patch_gdn_forward`` above injects the packed-document boundaries into whatever callables the
    instance carries, so this only has to replace them. Both slots exist because the native
    fallbacks defeat that injection in different ways:

      * no ``flash-linear-attention`` -> ``self.chunk_gated_delta_rule`` becomes transformers'
        ``torch_chunk_gated_delta_rule``, which ends in ``**kwargs``: the injected ``cu_seqlens`` is
        swallowed and ignored, so the recurrence never resets and nothing warns;
      * no ``causal_conv1d`` -> ``self.causal_conv1d_fn`` is ``None`` and the forward drops to
        ``F.silu(self.conv1d(...))``, which takes no ``seq_idx``, so the wrapper finds nothing to
        wrap and the conv state carries across documents.

    Slots resolve independently: one missing Hub build leaves the other bound.
    """
    from ..kernels.hub import resolve_slot

    modules = [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    if not modules:
        return 0

    bound = []
    for slot, attr_to_function in _GDN_KERNEL_SLOTS:
        functions = resolve_slot(args, slot)
        if not functions:
            continue
        for module in modules:
            for attr, function_name in attr_to_function.items():
                setattr(module, attr, functions[function_name])
        bound.append(slot)

    if not bound:
        return 0

    logger.info(
        "[fsdp hub kernels] GatedDeltaNet %s served from the Hub on %d module(s); "
        "the packed-document reset stays active without the native wheels",
        " + ".join(bound),
        len(modules),
    )
    return len(modules)
