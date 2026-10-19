"""The released GLM-5.3-Flash FP8 checkpoint leaves KDA, the indexer, and kv_b in BF16.

Weight updates have to keep that split. Quantizing a skipped parameter would
hand SGLang an FP8 tensor where the checkpoint stored BF16.
"""

from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
    _build_layer_skip_predicate,
    quantize_params_fp8,
)

_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": [
        "lm_head",
        "model.layers.11.self_attn.indexer.wq_b",
        "model.layers.11.self_attn.kv_b_proj",
        "weights_proj",
    ],
}


def test_fp8_skip_matches_full_paths_and_short_tokens():
    is_skipped = _build_layer_skip_predicate(tuple(_CONFIG["modules_to_not_convert"]), ())

    assert is_skipped("lm_head")
    assert is_skipped("model.layers.11.self_attn.kv_b_proj")
    assert is_skipped("model.layers.11.self_attn.indexer.weights_proj")
    assert not is_skipped("model.layers.10.mlp.experts")
    # A layer-11 skip must not swallow layer 110.
    assert not is_skipped("model.layers.110.self_attn.kv_b_proj")


def test_quantize_params_fp8_keeps_skipped_parameters():
    weight = torch.zeros(4, 4)
    megatron_name = "module.module.decoder.layers.11.self_attention.linear_kv_up_proj.weight"
    converted = [("model.layers.11.self_attn.kv_b_proj.weight", weight)]

    out = quantize_params_fp8(SimpleNamespace(backend=SimpleNamespace()), megatron_name, converted, _CONFIG)

    assert out == [("model.layers.11.self_attn.kv_b_proj.weight", weight)]
