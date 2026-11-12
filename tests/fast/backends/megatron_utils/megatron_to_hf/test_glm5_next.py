import torch

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf


def test_exported_names_are_the_checkpoint_names():
    """The text stack sits under model.language_model., as in GLM-5.3-Flash's checkpoint; the head does not."""
    args = MegatronArgsNamespace(hidden_size=4, num_attention_heads=2, num_query_groups=2, kv_channels=None)
    weight = torch.zeros(4, 4)
    expected = {
        "module.module.decoder.layers.0.self_attention.kda.q_proj.weight": [
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ],
        "module.module.decoder.layers.3.self_attention.wq_b.weight": [
            "model.language_model.layers.3.self_attn.indexer.wq_b.weight"
        ],
        "module.module.decoder.layers.1.mlp_hyper_connection.mapping_proj.weight": [
            "model.language_model.layers.1.hc_ffn_fn"
        ],
        "module.module.decoder.layers.4.mlp.experts.linear_fc1.weight7": [
            "model.language_model.layers.4.mlp.experts.7.gate_proj.weight",
            "model.language_model.layers.4.mlp.experts.7.up_proj.weight",
        ],
        "module.module.embedding.word_embeddings.weight": ["model.language_model.embed_tokens.weight"],
        "module.module.decoder.final_layernorm.weight": ["model.language_model.norm.weight"],
        "module.module.output_layer.weight": ["lm_head.weight"],
    }
    for name, hf_names in expected.items():
        assert [hf_name for hf_name, _ in convert_glm5_next_to_hf(args, name, weight)] == hf_names, name
