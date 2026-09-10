"""Opt-in GLM experiment hook for Megatron-Core's packed DSA implementation.

The installed Bridge compatibility layer predates Core's packed-sequence and
cross-layer index sharing support. Keep the Bridge checkpoint mappings, use
Core attention, and retain explicit Q/K/V projection forwards so every slot's
KV LoRA participates (the older absorbed path only knows a single adapter).
"""


def packed_sparse_sdpa(
    query, key, value, topk_indices, softmax_scale, mask=None, varlen_starts=None, varlen_ends=None, key_positions=None
):
    """Apply the exact selected-key mask with memory-efficient SDPA.

    Core's reference kernel gathers [tokens, topk, heads, head_dim] K/V tensors
    into its autograd graph. At 8K that exceeds the memory left by this model.
    A dense selection mask is smaller; SDPA avoids materializing attention
    scores. The compute is dense, so this is a correctness/capacity recipe,
    not a benchmark of the fused sparse kernel's throughput.
    """
    import torch
    from megatron.core.transformer.experimental_attention_variant import dsa_layout, dsa_masking
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query, was_thd = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    value, _ = dsa_layout.ensure_sbhd(value, "value")
    sq, batch, heads, _ = query.shape
    sk = key.shape[0]
    row_mask, starts, ends, positions = dsa_masking.prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=sk,
        b=batch,
        device=query.device,
    )
    masks = []
    for bi in range(batch):
        raw = topk_indices[bi].to(device=query.device, dtype=torch.int64)
        indices = raw.clamp(0, sk - 1)
        valid, bias = dsa_masking.gather_sparse_topk_validity_and_bias(
            idx_topk=indices,
            valid_t=(raw >= 0) & (raw < sk),
            bi=bi,
            s0=0,
            s1=sq,
            row_mask=row_mask,
            varlen_starts=starts,
            varlen_ends=ends,
            key_positions=positions,
            dtype=query.dtype,
        )
        selected = torch.zeros_like(indices, dtype=query.dtype) if bias is None else bias.to(query.dtype)
        selected = selected.masked_fill(~valid, float("-inf"))
        dense = torch.full((sq, sk), float("-inf"), dtype=query.dtype, device=query.device)
        # Invalid (-1) entries clamp to 0: amax prevents them from erasing a
        # valid key-0 selection in the same row.
        dense.scatter_reduce_(1, indices, selected, reduce="amax", include_self=True)
        masks.append(dense)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(
            query.permute(1, 2, 0, 3),
            key.permute(1, 2, 0, 3),
            value.permute(1, 2, 0, 3),
            attn_mask=torch.stack(masks).unsqueeze(1),
            dropout_p=0.0,
            scale=softmax_scale,
            enable_gqa=key.shape[2] != heads,
        )
    output = output.permute(2, 0, 1, 3).reshape(sq, batch, -1)
    return output.squeeze(1) if was_thd else output


def native_dsa_spec(config, backend=None):
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as eav
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    config.dsa_indexer_topk_freq = config.dsa_index_topk_freq
    config.dsa_indexer_skip_topk_offset = config.dsa_index_skip_topk_offset
    spec = eav.get_dsa_module_spec_for_backend(config=config, backend=backend)
    spec.module = MLASelfAttention
    return spec


def install(args):
    from megatron.bridge.models.glm5 import cross_layer_dsa_dispatch
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
    from megatron.core.transformer.transformer_config import TransformerConfig

    assert args.dsa_attention_backend == "megatron"
    assert args.context_parallel_size == args.pipeline_model_parallel_size == 1
    assert args.qkv_format == "thd"
    if not hasattr(DSAttention, "_get_index_share_topk_holder") or (
        "dsa_indexer_topk_freq" not in TransformerConfig.__dataclass_fields__
    ):
        raise RuntimeError("This pressure recipe requires Megatron-Core with native packed DSA and index sharing")
    cross_layer_dsa_dispatch.get_glm5_crosslayer_dsa_spec = native_dsa_spec
    from megatron.core.transformer.experimental_attention_variant import dsa

    dsa.unfused_dsa_fn = packed_sparse_sdpa
