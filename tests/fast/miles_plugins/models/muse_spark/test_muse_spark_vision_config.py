"""MuseSparkVisionConfig.from_hf reads the released checkpoint config (``muse_spark``, nested sub-configs)."""

import pytest

from miles_plugins.models.muse_spark.vision.navit import MuseSparkVisionConfig


def _layer_types(num_layers=50, factor=4):
    return [
        "full_attention" if index == num_layers - 1 or (index + 1) % factor == 0 else "window_attention"
        for index in range(num_layers)
    ]


def _released_config():
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": {"hidden_size": 8192, "rms_norm_eps": 1e-5, "normalize_tok_embeddings": None},
        "vision_config": {
            "model_type": "muse_spark_vision",
            "hidden_act": "gelu",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "merge_size": 2,
            "num_attention_heads": 16,
            "num_hidden_layers": 50,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "layer_types": _layer_types(),
        },
    }


def test_released_config():
    config = MuseSparkVisionConfig.from_hf(_released_config())

    assert (config.hidden_size, config.num_hidden_layers, config.num_attention_heads) == (1536, 50, 16)
    assert (config.intermediate_size, config.vision_output_dim, config.projector_input_dim) == (8960, 6144, 4096)
    assert (config.image_size, config.patch_size, config.patch_temporal, config.downsample_factor) == (448, 14, 2, 2)
    assert config.llm_hidden_size == 8192 and config.rope_theta == 10000.0 and config.norm_eps == 1e-5
    # SGLang's omni model always RMS-normalizes the projected rows, whatever the text config says.
    assert config.normalize_embeddings is True and config.embedding_norm_eps == 1e-5
    assert config.sparse_attention_factor == 4
    assert [config.uses_full_attention(index) for index in range(50)] == [
        kind == "full_attention" for kind in _layer_types()
    ]


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_layer_types_become_the_sparse_attention_factor(factor):
    hf_config = _released_config()
    hf_config["vision_config"]["layer_types"] = _layer_types(factor=factor)

    assert MuseSparkVisionConfig.from_hf(hf_config).sparse_attention_factor == factor


def test_layer_types_outside_the_navit_rule_are_rejected():
    hf_config = _released_config()
    hf_config["vision_config"]["layer_types"] = ["full_attention"] + ["window_attention"] * 49

    with pytest.raises(ValueError, match="window/full attention pattern"):
        MuseSparkVisionConfig.from_hf(hf_config)


def test_projector_width_must_match_the_merged_tower():
    hf_config = _released_config()
    hf_config["out_hidden_size"] = 4096

    with pytest.raises(ValueError, match="out_hidden_size"):
        MuseSparkVisionConfig.from_hf(hf_config)
