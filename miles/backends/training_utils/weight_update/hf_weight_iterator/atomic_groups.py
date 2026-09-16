"""HF-namespace atomic update groups: HF weights an sglang loader fuses into
one engine parameter, which therefore must arrive in the same load call."""

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import AtomicUpdateGroup

# sglang's deepseek_v4 load_weights cache-and-concats each pair into one engine
# param (wqkv_a / compressor.wkv_gate / indexer.compressor.wkv_gate) and
# hard-asserts no half-arrived pair remains at end of call.
_DEEPSEEK_V4_GROUPS = [
    AtomicUpdateGroup(key, suffixes)
    for key, suffixes in [
        ("wqkv_a", (".self_attn.wq_a.weight", ".self_attn.wkv.weight")),
        (
            "compressor_wkv_gate",
            (".self_attn.compressor.wkv.weight", ".self_attn.compressor.wgate.weight"),
        ),
        (
            "indexer_compressor_wkv_gate",
            (
                ".self_attn.indexer.compressor.wkv.weight",
                ".self_attn.indexer.compressor.wgate.weight",
            ),
        ),
    ]
]


def get_hf_atomic_update_groups(model_name: str, *, q_lora_rank: int | None = None) -> list[AtomicUpdateGroup]:
    """Atomic groups for a model. inkling registers none: its fusions happen
    inside the converter, and its engine-side loads are split-safe."""
    model_name = model_name.lower()
    if "deepseekv4" in model_name:
        return list(_DEEPSEEK_V4_GROUPS)
    if "inkling" in model_name:
        return []
    if q_lora_rank is not None:
        # sglang's deepseek family cache-and-concats q_a_proj + kv_a_proj_with_mqa
        # into fused_qkv_a_proj_with_mqa; a half-arrived pair is a silent no-op.
        return [
            AtomicUpdateGroup(
                key="q_lora_a_proj",
                suffixes=(
                    ".self_attn.q_a_proj.weight",
                    ".self_attn.kv_a_proj_with_mqa.weight",
                ),
            )
        ]
    return []
