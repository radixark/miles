import subprocess
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
from transformers import LlamaConfig

from miles.utils.lora import arguments as lora_arguments
from miles.utils.lora.arguments import _resolve_lora_targets


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


def _bridge_args_without_layout(monkeypatch, *, multi_lora=False, **overrides):
    hf_config = SimpleNamespace(model_type="gpt2", to_dict=lambda: {"model_type": "gpt2"})
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: hf_config)
    monkeypatch.setattr(
        lora_arguments, "validate_multi_lora_args", lambda args: setattr(args, "multi_lora", multi_lora)
    )
    monkeypatch.setattr(lora_arguments, "_resolve_lora_targets", lambda args, config: ["h.0.attn.c_attn"])
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora.target_modules.check_fused_lora_selection", lambda *a, **k: None
    )
    fields = dict(
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        hf_checkpoint="/hf",
        lora_rank=8,
        lora_adapter_path=None,
        lora_type="lora",
        experts_shared_outer_loras=False,
        save_hf=None,
        eval_hf_dir=None,
        eval_uses_snapshots=False,
        save=None,
    )
    return Namespace(**(fields | overrides))


def _validate(args):
    lora_arguments.validate_lora_args(args)
    lora_arguments.validate_lora_args_after_megatron(args)


def test_only_hf_exports_and_multi_lora_need_the_hf_lora_layout(monkeypatch):
    """The adapter slots, which need the layout, are built only for what reads them: HF snapshot exports and
    multi-LoRA's sampler exports; a plain --save needs none."""
    _validate(_bridge_args_without_layout(monkeypatch, save="/save"))
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, save_hf="/hf_out"))


def test_an_eval_hf_dir_without_eval_snapshots_needs_no_layout(monkeypatch):
    """Only eval snapshots export to --eval-hf-dir; the actor publishes nothing for it otherwise."""
    _validate(_bridge_args_without_layout(monkeypatch, eval_hf_dir="/eval"))
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, eval_hf_dir="/eval", eval_uses_snapshots=True))


def test_multi_lora_bridge_refuses_a_model_without_hf_lora_layout(monkeypatch):
    """Multi-LoRA builds the adapter slots its sampler exports read, even without --save-hf / --save."""
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, multi_lora=True))


def _canonical_gdn_args(monkeypatch, **overrides):
    # a model with a layout, whose canonical adapters target one GDN in_proj_qkv by component
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(lora_arguments, "has_hf_lora_targets", lambda config: True)
    fields = dict(
        hf_checkpoint="/hf",
        lora_type="canonical_lora",
        lora_adapter_targets=[f"model.layers.0.linear_attn.in_proj_{c}" for c in "qkvz"],
        save_hf=None,
        eval_hf_dir=None,
        eval_uses_snapshots=False,
        multi_lora=False,
    )
    return Namespace(**(fields | overrides))


def test_canonical_gdn_components_refuse_hf_exports_at_launch(monkeypatch):
    """A PEFT adapter cannot name in_proj_q/k/v with their own lora_A, so the export would fail at every save."""
    lora_arguments._assert_adapter_layout_defined(_canonical_gdn_args(monkeypatch))
    # multi-LoRA reads the slots but writes no PEFT view
    lora_arguments._assert_adapter_layout_defined(_canonical_gdn_args(monkeypatch, multi_lora=True))
    with pytest.raises(AssertionError, match=r"no PEFT adapter can name: use --lora-type lora"):
        lora_arguments._assert_adapter_layout_defined(_canonical_gdn_args(monkeypatch, save_hf="/hf_out"))
    with pytest.raises(AssertionError, match=r"\['in_proj_k', 'in_proj_q', 'in_proj_v'\]"):
        lora_arguments._assert_adapter_layout_defined(
            _canonical_gdn_args(monkeypatch, eval_hf_dir="/eval", eval_uses_snapshots=True)
        )
