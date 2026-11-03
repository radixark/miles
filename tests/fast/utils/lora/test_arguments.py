import subprocess
import sys
from argparse import Namespace

import pytest
from transformers import AutoConfig, LlamaConfig

from miles.utils.lora.arguments import _assert_adapter_layout_defined, _resolve_lora_targets


class TestLoraTargetOrdering:
    def test_exclusions_expand_to_stable_hf_module_order_without_changing_selectors(self) -> None:
        """Resolved adapter targets have reproducible order after exclusions expand module names."""
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        args = Namespace(
            target_modules="all-linear",
            exclude_modules="q_proj",
            multi_lora=False,
            megatron_to_hf_mode="bridge",
            lora_type="canonical_lora",
        )

        hf_targets = _resolve_lora_targets(args, config)

        expected = [
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.1.mlp.down_proj",
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.1.self_attn.v_proj",
        ]
        assert hf_targets == expected
        assert args.target_modules == "all-linear"
        assert args.exclude_modules == "q_proj"


def test_lora_arguments_import_no_megatron_to_hf():
    """megatron_to_hf loads every converter and SGLang's quantization kernels, which probe the GPU on ROCm (#4201)."""
    code = "import sys, miles.utils.lora.arguments; sys.exit('miles.backends.megatron_utils.megatron_to_hf' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


_NO_SNAPSHOTS = dict(save_hf=None, eval_hf_dir=None, save=None, lora_adapter_path=None)


@pytest.mark.parametrize(
    "flag, value",
    [("save", "/ckpt"), ("save_hf", "/hf/{rollout_id}"), ("eval_hf_dir", "/eval"), ("lora_adapter_path", "/adapter")],
)
def test_adapter_slots_need_the_families_hf_lora_layout(flag, value) -> None:
    """Native trains glm4, which has no HF LoRA layout; saving or loading its adapter would crash building the slots."""
    args = Namespace(**(_NO_SNAPSHOTS | {flag: value}))
    with pytest.raises(AssertionError, match="model_type='glm4', which miles does not define"):
        _assert_adapter_layout_defined(args, AutoConfig.for_model("glm4"))
    _assert_adapter_layout_defined(args, AutoConfig.for_model("qwen3"))


def test_a_run_without_adapter_snapshots_trains_any_family() -> None:
    _assert_adapter_layout_defined(Namespace(**_NO_SNAPSHOTS), AutoConfig.for_model("glm4"))
