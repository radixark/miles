from types import SimpleNamespace

import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor


def _model_cls(**config_fields):
    actor = object.__new__(FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(**config_fields)
    return actor.get_model_cls()


def test_native_vlm_routes_to_image_text_to_text():
    # Qwen3-VL: multimodal, no auto_map, resolves through the transformers registry.
    assert _model_cls(model_type="qwen3_vl", vision_config={"depth": 24}) is AutoModelForImageTextToText


def test_remote_code_multimodal_without_i2t_in_auto_map_routes_to_causal_lm():
    # Kimi-K2.5 ships a vision_config but its remote code only maps AutoModelForCausalLM,
    # so asking for AutoModelForImageTextToText raises "Unrecognized configuration".
    cls = _model_cls(
        model_type="kimi_k25",
        vision_config={"init_pos_emb_height": 64},
        auto_map={
            "AutoConfig": "configuration_kimi.KimiK25Config",
            "AutoModel": "modeling_kimi.KimiK25Model",
            "AutoModelForCausalLM": "modeling_kimi.KimiK25ForCausalLM",
        },
    )
    assert cls is AutoModelForCausalLM


def test_remote_code_multimodal_declaring_i2t_routes_to_image_text_to_text():
    cls = _model_cls(
        model_type="some_remote_vlm",
        vision_config={"depth": 8},
        auto_map={"AutoModelForImageTextToText": "modeling_x.XForConditionalGeneration"},
    )
    assert cls is AutoModelForImageTextToText


def test_text_only_remote_code_routes_to_causal_lm():
    cls = _model_cls(model_type="some_remote_lm", auto_map={"AutoModelForCausalLM": "modeling_x.XForCausalLM"})
    assert cls is AutoModelForCausalLM


def test_native_causal_lm_resolves_concrete_class():
    assert _model_cls(model_type="qwen3") is transformers.Qwen3ForCausalLM
