"""Per-arch registry: HF ``config.json`` -> torchtitan ``ModelSpec``.

torchtitan's own flavor factories are hardcoded (a fixed table of named sizes); there is
no ``from_hf_config`` entry point, so this mapper is ours. Every derivable field is
hard-asserted against the flavor-builder output so a checkpoint that doesn't match our
assumptions fails loudly at build time rather than silently producing a wrong model.

fuse_qkv=False is required and is NOT reachable through ``model_registry`` (every
non-debug flavor factory hardcodes ``fuse_qkv=True``): with fused QKV, every
``state_dict()`` call triggers a redistribute-to-Replicate allgather of the whole wqkv
tensor (the per-call save hook that keeps the checkpoint's un-fused wq/wk/wv layout).
Building layers directly with ``fuse_qkv=False`` avoids that hook — required because
weight sync calls into the state dict every training step.
"""

import json
import os


def _load_hf_config(hf_checkpoint: str) -> dict:
    with open(os.path.join(hf_checkpoint, "config.json")) as f:
        return json.load(f)


def spec_from_hf(hf_checkpoint: str, *, attn_backend: str = "flex"):
    """Build a torchtitan ModelSpec from an HF checkpoint's config.json.

    Supports model_type == "qwen3" (dense) and "qwen3_5" (dense, text-only —
    vision encoder is pruned post-build, see model.py's build_and_load_model).
    Returns (spec, hf_config).
    """
    hf = _load_hf_config(hf_checkpoint)
    model_type = hf.get("model_type")
    if model_type == "qwen3":
        return _qwen3_spec_from_hf(hf, attn_backend=attn_backend), hf
    if model_type == "qwen3_5":
        return _qwen3_5_spec_from_hf(hf, attn_backend=attn_backend), hf
    raise NotImplementedError(f"model_type={model_type!r} not yet supported")


def _qwen3_spec_from_hf(hf: dict, *, attn_backend: str):
    import torchtitan.models.qwen3 as q3
    from torchtitan.protocols.model_spec import ModelSpec

    eps = float(hf.get("rms_norm_eps", 1e-6))
    if abs(eps - q3._EPS) > 1e-12:
        raise NotImplementedError(f"rms_norm_eps={eps} != titan qwen3 _EPS={q3._EPS}")

    dim = hf["hidden_size"]
    n_heads = hf["num_attention_heads"]
    n_kv_heads = hf["num_key_value_heads"]
    head_dim = int(hf.get("head_dim") or dim // n_heads)
    vocab_size = hf["vocab_size"]
    tie = bool(hf.get("tie_word_embeddings", False))

    layers = q3._build_qwen3_layers(
        fuse_qkv=False,
        n_layers=hf["num_hidden_layers"],
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_dim=hf["intermediate_size"],
        attn_backend=attn_backend,
        rope=q3.CosSinRoPE.Config(
            dim=head_dim,
            max_seq_len=int(hf.get("max_position_embeddings", 4096)),
            theta=float(hf.get("rope_theta", 1000000.0)),
        ),
    )
    for layer_cfg in layers:
        from torchtitan.models.common.attention import FusedQKVLinear

        assert not isinstance(layer_cfg.attention.qkv_linear, FusedQKVLinear.Config), (
            "fuse_qkv leaked through _build_qwen3_layers(fuse_qkv=False, ...) — "
            "weight sync depends on the un-fused per-call allgather-free state_dict()."
        )

    config = q3.Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=q3._qwen3_norm(dim),
        tok_embeddings=q3.Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=q3._EMBEDDING_INIT
        ),
        lm_head=q3.Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=q3._output_linear_init(dim)
        ),
        layers=layers,
        enable_weight_tying=tie,
    )
    return ModelSpec(
        name="qwen3",
        flavor="hf-mapped",
        model=config,
        parallelize_fn=q3.parallelize_qwen3,
        pipelining_fn=None,
        post_optimizer_build_fn=q3.register_moe_load_balancing_hook,
        state_dict_adapter=q3.Qwen3StateDictAdapter,
    )


def _qwen3_5_spec_from_hf(hf: dict, *, attn_backend: str):
    """qwen3_5 (Qwen3.5, hybrid GDN + full-attention). HF's config nests every
    text-model dimension under ``text_config`` (this is a VL checkpoint format even
    when used text-only); ``vision_config`` is still read here (the model always
    builds a vision_encoder submodule — see model.py's Config — v1 prunes it to
    None post-build for text-only RL, matching torchtitan's own PP-prune pattern,
    which is why every dtype/parallelize path already null-guards vision_encoder).

    No fuse_qkv concern here (unlike qwen3): Qwen35Attention.Config always uses
    separate wq/wk/wv, no fused-QKV variant exists to accidentally reach.

    enable_weight_tying is deliberately left unset (False): torchtitan's qwen3_5
    doesn't implement real weight tying yet (see upstream _4b()'s docstring) even
    though this checkpoint has tie_word_embeddings=true — Qwen35StateDictAdapter.
    from_hf already copies the embedding into a separate lm_head.weight when the
    checkpoint omits it, so loading is correct without a truly aliased tensor.
    """
    import torchtitan.models.qwen3_5 as q35
    from torchtitan.protocols.model_spec import ModelSpec

    text = hf["text_config"]
    eps = float(text.get("rms_norm_eps", 1e-6))
    if abs(eps - q35._EPS) > 1e-12:
        raise NotImplementedError(f"rms_norm_eps={eps} != titan qwen3_5 _EPS={q35._EPS}")

    dim = text["hidden_size"]
    n_layers = text["num_hidden_layers"]
    n_heads = text["num_attention_heads"]
    n_kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    vocab_size = text["vocab_size"]
    full_attention_interval = text["full_attention_interval"]

    rope_params = text["rope_parameters"]
    partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial_rotary_factor)
    mrope_section = list(rope_params["mrope_section"])
    theta = float(rope_params["rope_theta"])
    max_seq_len = int(text.get("max_position_embeddings", 4096))

    layers = q35._build_qwen35_layers(
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=q35.MRoPE.Config(dim=rotary_dim, max_seq_len=max_seq_len, theta=theta, mrope_section=mrope_section),
        hidden_dim=text["intermediate_size"],
        n_key_heads=text["linear_num_key_heads"],
        n_value_heads=text["linear_num_value_heads"],
        key_head_dim=text["linear_key_head_dim"],
        value_head_dim=text["linear_value_head_dim"],
        full_attention_interval=full_attention_interval,
        attn_backend=attn_backend,
    )
    for layer_cfg in layers:
        from torchtitan.models.common.attention import FusedQKVLinear

        if layer_cfg.attention is not None:
            assert not isinstance(layer_cfg.attention.wq, FusedQKVLinear.Config), (
                "unexpected fused QKV in a qwen3_5 full-attention layer"
            )

    vc = hf.get("vision_config", {})
    vision_encoder_config = q35._qwen35_vision_encoder_config(
        dim=vc.get("hidden_size", 1024),
        ffn_dim=vc.get("intermediate_size", 4096),
        num_layers=vc.get("depth", 24),
        num_heads=vc.get("num_heads", 16),
        patch_size=vc.get("patch_size", 16),
        temporal_patch_size=vc.get("temporal_patch_size", 2),
        spatial_merge_size=vc.get("spatial_merge_size", 2),
        out_hidden_size=vc.get("out_hidden_size", dim),
        num_position_embeddings=vc.get("num_position_embeddings", 2304),
    )

    config = q35.Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=q35._offset_norm(dim),
        tok_embeddings=q35.Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=q35._EMBEDDING_INIT
        ),
        lm_head=q35.Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=q35._output_linear_init(dim)
        ),
        layers=layers,
        vision_encoder=vision_encoder_config,
    )
    return ModelSpec(
        name="qwen3_5",
        flavor="hf-mapped",
        model=config,
        parallelize_fn=q35.parallelize_qwen3_5,
        pipelining_fn=None,
        post_optimizer_build_fn=q35.register_moe_load_balancing_hook,
        state_dict_adapter=q35.Qwen35StateDictAdapter,
    )
