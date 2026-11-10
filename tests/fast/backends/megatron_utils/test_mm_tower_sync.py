"""--mm-tower-sync: which frozen towers the Megatron weight update re-sends."""

from types import SimpleNamespace

from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    is_tower_key,
    mm_tower_names,
)
from miles.utils.args.custom_function import CustomFunctionConfig


def test_flag_names_the_towers_and_default_is_off():
    assert mm_tower_names(
        SimpleNamespace(mm_tower_sync=["visual"], custom_model_provider_path=CustomFunctionConfig(path="x.y"))
    ) == ("visual",)
    assert (
        mm_tower_names(SimpleNamespace(mm_tower_sync=[], custom_model_provider_path=CustomFunctionConfig(path="x.y")))
        == ()
    )
    assert (
        mm_tower_names(
            SimpleNamespace(
                mm_tower_sync=None, custom_model_provider_path=CustomFunctionConfig(path="my_plugins.qwen_vl_provider")
            )
        )
        == ()
    )


def test_inkling_provider_implies_its_two_towers():
    args = SimpleNamespace(
        mm_tower_sync=None,
        custom_model_provider_path=CustomFunctionConfig(path="miles_plugins.inkling_mm_model_provider"),
    )
    assert mm_tower_names(args) == ("visual", "audio")
    # an explicit flag wins over the implied default
    args.mm_tower_sync = ["visual"]
    assert mm_tower_names(args) == ("visual",)


def test_no_provider_and_no_flag_is_off():
    assert mm_tower_names(SimpleNamespace(mm_tower_sync=None, custom_model_provider_path=None)) == ()


def test_tower_keys_match_as_path_parts():
    assert is_tower_key("model.visual.blocks.0.attn.proj.weight", ("visual",))
    assert is_tower_key("visual.merger.ln_q.weight", ("visual",))
    assert not is_tower_key("model.language_model.layers.0.visual_gate.weight", ("visual",))
    assert not is_tower_key("model.visual.blocks.0.attn.proj.weight", ("audio",))
