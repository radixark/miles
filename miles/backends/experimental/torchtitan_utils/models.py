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

    v1 supports model_type == "qwen3" (dense). Returns (spec, hf_config).
    """
    hf = _load_hf_config(hf_checkpoint)
    model_type = hf.get("model_type")
    if model_type != "qwen3":
        raise NotImplementedError(
            f"model_type={model_type!r} not yet supported (v1: dense qwen3 only)"
        )
    return _qwen3_spec_from_hf(hf, attn_backend=attn_backend), hf


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
