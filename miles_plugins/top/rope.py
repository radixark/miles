"""Fused-only RoPE at Megatron SelfAttention's existing function seam."""


def enforce_fused_rope(args, config):
    from megatron.core.extensions import transformer_engine as te

    from miles_plugins.top.install import install_fused_rope

    if getattr(config, "mrope_section", None) is not None:
        raise NotImplementedError("[top] matched fused RoPE does not admit mRoPE")
    if getattr(config, "fused_single_qkv_rope", False):
        raise NotImplementedError("[top] matched RoPE requires the separate TE fused RoPE call")
    for name in ("fused_apply_rotary_pos_emb", "fused_apply_rotary_pos_emb_thd"):
        if not callable(getattr(te, name, None)):
            raise RuntimeError(f"[top] required TE fused RoPE implementation is unavailable: {name}")
    config.apply_rope_fusion = True
    args.apply_rope_fusion = True
    install_fused_rope()


def apply_fused_rope(
    t,
    freqs,
    config,
    cu_seqlens=None,
    mscale=1.0,
    cp_group=None,
    mla_rotary_interleaved=False,
    inverse=False,
    mla_output_remove_interleaving=False,
    max_seqlen=None,
):
    """Keep the stock SBHD/THD and CP mapping, but never enter unfused dispatch."""
    if not config.apply_rope_fusion:
        raise RuntimeError("[top] fused RoPE was disabled after TOP construction")
    if (
        getattr(config, "mrope_section", None) is not None
        or mscale != 1.0
        or mla_rotary_interleaved
        or (mla_rotary_interleaved is None and config.multi_latent_attention)
        or inverse
        or mla_output_remove_interleaving
    ):
        raise NotImplementedError("[top] requested RoPE expression is not supported by the matched TE fused path")
    from megatron.core.extensions import transformer_engine as te

    if cu_seqlens is None:
        return te.fused_apply_rotary_pos_emb(t, freqs, interleaved=config.rotary_interleaved)
    if cp_group is None:
        raise RuntimeError("[top] packed fused RoPE requires the caller's CP group")
    return te.fused_apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        freqs,
        cp_size=cp_group.size(),
        cp_rank=cp_group.rank(),
        interleaved=config.rotary_interleaved,
    )
