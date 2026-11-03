"""Bridge single LoRA's --save / --lora-adapter-path glue around the adapter library."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from miles.backends.megatron_utils.lora.checkpoint import single_lora_parametrization


def _base_model(hf_checkpoint: str) -> str:
    slots = MagicMock()
    args = SimpleNamespace(
        hf_checkpoint=hf_checkpoint, lora_rank=8, lora_alpha=16.0, lora_type="lora", experts_shared_outer_loras=False
    )
    single_lora_parametrization(slots, args)
    return slots.parametrization.call_args.kwargs["base_model"]


def test_one_local_model_under_another_path_string_is_the_same_base(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "mount").symlink_to(model)
    expected = os.path.realpath(model)
    assert _base_model(str(model)) == _base_model(f"{model}/") == _base_model(str(tmp_path / "mount")) == expected


def test_a_hub_id_is_kept():
    assert _base_model("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
