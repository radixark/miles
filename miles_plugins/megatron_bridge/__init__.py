"""Miles plugin package for ``megatron.bridge`` integration.

Importing this package is enough to:

* register miles' bridge subclasses (e.g.
  :class:`~miles_plugins.megatron_bridge.nemotron_h.MilesNemotronHBridge`) via
  ``@MegatronModelBridge.register_bridge`` so ``AutoBridge`` picks them up
  instead of the upstream defaults;
* install general-purpose shims that make ``megatron.bridge`` cooperate with
  miles infrastructure (e.g. ``ReloadableProcessGroup``).

Every shim / registration is wrapped in try/except so an import-time failure of
one model does not prevent other models from working.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _install_bridge_pp_group_unwrap() -> None:
    """Let ``MegatronParamMapping.broadcast_obj_from_pp_rank`` work with
    miles' :class:`~miles.utils.reloadable_process_group.ReloadableProcessGroup`.

    ``broadcast_obj_from_pp_rank`` calls ``broadcast_object_list`` on
    ``self.pp_group``, which goes through ``_world.pg_group_ranks``. Miles wraps
    every ``ProcessGroup`` in ``ReloadableProcessGroup`` for reload-safety; that
    wrapper is not in ``pg_group_ranks`` so ``get_group_rank`` raises
    ``"Group ... is not registered"``. Temporarily swap in the inner group for
    the duration of the broadcast.
    """
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    if getattr(MegatronParamMapping, "_miles_pp_group_unwrap_installed", False):
        return

    _orig = MegatronParamMapping.broadcast_obj_from_pp_rank

    def broadcast_obj_from_pp_rank(self, obj, name=None):
        if not isinstance(self.pp_group, ReloadableProcessGroup):
            return _orig(self, obj, name)
        saved = self.pp_group
        self.pp_group = saved.group
        try:
            return _orig(self, obj, name)
        finally:
            self.pp_group = saved

    MegatronParamMapping.broadcast_obj_from_pp_rank = broadcast_obj_from_pp_rank
    MegatronParamMapping._miles_pp_group_unwrap_installed = True


try:
    _install_bridge_pp_group_unwrap()
except Exception as _e:  # best-effort
    logger.warning("miles bridge shim _install_bridge_pp_group_unwrap not applied: %s", _e)


try:
    from miles_plugins.models.qwen3_vl import install_qwen3_vl_packed_mrope_patch

    install_qwen3_vl_packed_mrope_patch()
except Exception as _e:  # best-effort; Qwen3-VL may be unavailable in some envs
    logger.warning("miles Qwen3-VL THD packed mRoPE patch not applied: %s", _e)


def _gdn_nan_probe(module, hidden_states, out):
    """FIX + light probe for the GDN packed-sequence backward NaN.

    Under qkv_format=thd the packed buffer is padded and the padding is the last cu_seqlens segment,
    so the per-segment GDN backward runs over PADDING tokens whose degenerate backward yields NaN
    gradients (forward stays finite — confirmed: a GDN layer's grad-IN NaN with finite forward+out).
    Those positions are loss-masked, so their gradient *should* be zero — therefore sanitizing the
    layer's input-gradient with nan_to_num restores the correct value (NaN/±inf → 0) and stops the
    NaN from reaching the trainable LoRA adapters. nan_to_num is a no-op on healthy grads."""
    import torch

    ln = getattr(module, "layer_number", getattr(module, "layer_idx", "?"))
    try:
        if isinstance(out, torch.Tensor) and not bool(torch.isfinite(out).all()):
            logger.warning("[GDN-PROBE] FWD-OUT NON-FINITE at GDN layer=%s", ln)
        if isinstance(hidden_states, torch.Tensor) and hidden_states.requires_grad:
            def _sanitize_grad(g, ln=ln):
                if not bool(torch.isfinite(g).all()):
                    logger.warning("[GDN-FIX] sanitized non-finite grad-IN at GDN layer=%s", ln)
                    return torch.nan_to_num(g)
                return g  # healthy → unchanged
            hidden_states.register_hook(_sanitize_grad)
    except Exception:
        pass


def _install_gdn_packed_seq_patch() -> None:
    """Make Megatron's core GatedDeltaNet handle packed sequences (qkv_format=thd).

    In bridge mode the model is built from megatron.core's GatedDeltaNet, which
    raises NotImplementedError on ``packed_seq_params``. Miles passes packed_seq_params
    whenever qkv_format='thd' (the default). Naively nulling it would let the SSM
    recurrence + causal conv bleed across packed-sequence boundaries (state from one
    example leaking into the next) — a silent correctness bug that worsens with packing
    density. Instead: split the packed batch at cu_seqlens and run each sequence through
    the ORIGINAL single-sequence forward independently (initial_state=None per segment →
    no boundary bleed). Reuses the proven non-packed path, so it's correct without
    touching the kernel. Assumes TP=1 / no sequence-parallel, CP=1 (guarded).
    (Raw mode uses miles' own Qwen3_5GatedDeltaNet, which already handles varlen — so
    this is only needed on the bridge path, which is why it lives here.)
    """
    import torch
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    if getattr(GatedDeltaNet.forward, "_miles_varlen_patched", False):
        return
    _orig_forward = GatedDeltaNet.forward

    def forward(self, hidden_states, attention_mask=None, inference_context=None,
                packed_seq_params=None, sequence_len_offset=None, *,
                inference_params=None, **kwargs):
        cu = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu is None:  # not packed (bshd) → original path + lean fla kernel, unchanged
            out, out_bias = _orig_forward(self, hidden_states, attention_mask, inference_context,
                                          None, sequence_len_offset,
                                          inference_params=inference_params, **kwargs)
        else:
            # FIX (option 2, PACKED-ONLY): force the DETERMINISTIC torch GDN kernel. The fla
            # chunk_gated_delta_rule backward is non-deterministic (Megatron's own note: "FLA is not
            # deterministic") and NaNs under thd + full activation recompute — the recomputed forward
            # differs from the original → inconsistent/NaN backward. The pure-torch reference
            # (identical delta-rule math) is CPU-validated finite in fwd AND bwd. Same call signature
            # → drop-in. Slower & MORE MEMORY than fla, so we ONLY pay it on the packed (thd) path;
            # bshd (cu is None, above) keeps the lean fla kernel and its original memory footprint —
            # applying it unconditionally OOM'd bshd at step 1.
            if not getattr(self, "_miles_force_torch_gdr", False):
                try:
                    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
                    self.gated_delta_rule = torch_chunk_gated_delta_rule
                except Exception:
                    pass
                self._miles_force_torch_gdr = True
            sp = getattr(self, "sp_size", 1) or 1
            cp = getattr(self, "cp_size", 1) or 1
            if sp != 1 or cp != 1:
                raise NotImplementedError(
                    "GDN packed-sequence patch supports sp_size==cp_size==1 only "
                    f"(got sp={sp}, cp={cp}); use qkv_format=bshd or extend this patch."
                )
            seq_total = hidden_states.shape[0]  # THD: (total_tokens, batch=1, hidden)
            bounds = [int(x) for x in cu.tolist()]
            if not bounds or bounds[0] != 0:
                bounds = [0] + bounds
            real_end = bounds[-1]  # cu_seqlens[-1] = end of real tokens; [real_end:seq_total] is PADDING
            outs, out_bias = [], None
            for s, e in zip(bounds[:-1], bounds[1:]):
                if e <= s:
                    continue
                out_i, out_bias = _orig_forward(
                    self, hidden_states[s:e], attention_mask, inference_context,
                    None, sequence_len_offset,
                    inference_params=inference_params, **kwargs,
                )
                outs.append(out_i)
            out = torch.cat(outs, dim=0)
            # Do NOT feed trailing PADDING tokens through the GDN kernel — their degenerate
            # backward produces NaN gradients (confirmed origin: a GDN layer's grad-IN NaN with
            # finite forward). Emit zeros for padding positions instead (finite fwd, zero grad;
            # they're loss-masked). Restores the (seq_total, …) shape for downstream layers.
            if real_end < seq_total:
                pad = out.new_zeros((seq_total - real_end,) + tuple(out.shape[1:]))
                out = torch.cat([out, pad], dim=0)
        _gdn_nan_probe(self, hidden_states, out)  # TEMP DIAGNOSTIC
        return out, out_bias

    forward._miles_varlen_patched = True
    GatedDeltaNet.forward = forward


try:
    _install_gdn_packed_seq_patch()
except Exception as _e:  # best-effort; GDN may be unavailable in some envs
    logger.warning("miles GDN THD packed-sequence patch not applied: %s", _e)


def _install_attention_nan_guard() -> None:
    """Sanitize the core-attention output so fully-masked rows can't NaN-poison training.

    Under qkv_format=thd (packed sequences), a degenerate/padding query row can attend to zero
    valid keys → softmax(all −inf) = NaN at that (loss-masked) position. The position doesn't
    affect the loss VALUE (it's masked), but the attention output projection's weight-gradient
    sums grad_output ⊗ core_attn_out over ALL positions, and 0 × NaN = NaN → the whole grad_weight
    becomes NaN → the LoRA adapter (and the policy in SGLang) collapses after one step. (Confirmed
    via anomaly detection: LinearWithGradAccumulation...Backward NaN in grad_weight of linear_proj;
    confirmed fixed by bshd/no-packing.)

    Fix: wrap the core-attention forward to torch.nan_to_num its output. The clamped positions are
    loss-masked so 0 is the correct value, and it makes grad_weight finite (0 × 0 = 0). Backend-
    agnostic; applied to both the TE (flash) and the native core-attention classes. nan_to_num is a
    no-op when the output is already finite, so it costs ~nothing on healthy steps.
    """
    import torch

    def _wrap(cls):
        if cls is None or getattr(cls.forward, "_miles_attn_nan_guarded", False):
            return
        _orig = cls.forward

        def forward(self, *args, **kwargs):
            out = _orig(self, *args, **kwargs)
            if isinstance(out, torch.Tensor):
                return torch.nan_to_num(out)
            if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                return (torch.nan_to_num(out[0]),) + tuple(out[1:])
            return out

        forward._miles_attn_nan_guarded = True
        cls.forward = forward

    patched = []
    try:
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
        _wrap(TEDotProductAttention)
        patched.append("TEDotProductAttention")
    except Exception as _e:  # TE may be unavailable
        logger.warning("attn NaN guard: TEDotProductAttention not patched: %s", _e)
    try:
        from megatron.core.transformer.dot_product_attention import DotProductAttention
        _wrap(DotProductAttention)
        patched.append("DotProductAttention")
    except Exception as _e:
        logger.warning("attn NaN guard: DotProductAttention not patched: %s", _e)
    logger.warning("miles attention NaN guard installed on: %s", patched)


try:
    _install_attention_nan_guard()
except Exception as _e:  # best-effort
    logger.warning("miles attention NaN guard not applied: %s", _e)


# Model-specific bridge subclasses. Each submodule self-installs on import.
# Keep imports here so merely importing ``miles_plugins.megatron_bridge`` is
# enough to pick up every miles bridge (mirrors ``miles_plugins.mbridge``).
try:
    from . import nemotron_h  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    logger.warning("miles nemotron_h plugin failed to load: %s", _e)
