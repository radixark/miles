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


def test_only_runs_exporting_hf_snapshots_need_the_hf_lora_layout(monkeypatch):
    """The adapter slots, which need the layout, are built only for the HF snapshot exports that read them."""
    _validate(_bridge_args_without_layout(monkeypatch, save="/save"))
    _validate(_bridge_args_without_layout(monkeypatch, multi_lora=True))
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, save_hf="/hf_out"))


def test_an_eval_hf_dir_without_eval_snapshots_needs_no_layout(monkeypatch):
    """Only eval snapshots export to --eval-hf-dir; the actor publishes nothing for it otherwise."""
    _validate(_bridge_args_without_layout(monkeypatch, eval_hf_dir="/eval"))
    with pytest.raises(AssertionError, match="HF LoRA target layout"):
        _validate(_bridge_args_without_layout(monkeypatch, eval_hf_dir="/eval", eval_uses_snapshots=True))


_TINY_QWEN3_5_TEXT = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
)


def _gdn_export_args(monkeypatch, targets, **overrides):
    from transformers import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**_TINY_QWEN3_5_TEXT)
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: config)
    fields = dict(
        hf_checkpoint="/hf",
        lora_type="canonical_lora",
        hf_lora_targets=targets,
        lora_rank=4,
        lora_alpha=8.0,
        experts_shared_outer_loras=False,
        save_hf="/hf_out",
        eval_hf_dir=None,
        eval_uses_snapshots=False,
    )
    return Namespace(**(fields | overrides))


@pytest.mark.parametrize("targets", [["in_proj_qkv"], ["in_proj_*"], ["*.linear_attn.in_proj_*"]])
def test_canonical_gdn_components_refuse_hf_exports_at_launch(monkeypatch, targets):
    """The check resolves the targets to the base's modules, so a wildcard selecting in_proj_qkv is caught too."""
    with pytest.raises(AssertionError, match=r"no PEFT adapter can name: use --lora-type lora"):
        lora_arguments._assert_adapter_layout_defined(_gdn_export_args(monkeypatch, targets))
    # no HF export: nothing to refuse
    lora_arguments._assert_adapter_layout_defined(_gdn_export_args(monkeypatch, targets, save_hf=None))
    # one shared lora_A per in_proj_qkv has a PEFT form
    lora_arguments._assert_adapter_layout_defined(_gdn_export_args(monkeypatch, targets, lora_type="lora"))


def _gpt_oss_export_args(monkeypatch, **overrides):
    from transformers import GptOssConfig

    config = GptOssConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: config)
    fields = dict(
        hf_checkpoint="/hf",
        lora_type="lora",
        hf_lora_targets=["gate_up_proj", "down_proj"],
        lora_rank=4,
        lora_alpha=8.0,
        experts_shared_outer_loras=True,
        save_hf="/hf_out",
        eval_hf_dir=None,
        eval_uses_snapshots=False,
    )
    return Namespace(**(fields | overrides))


def test_a_packed_expert_shared_outer_adapter_refuses_hf_exports_at_launch(monkeypatch):
    """Bridge names a packed family's per-expert side once for all experts, so the export keeps only the last."""
    with pytest.raises(AssertionError, match="would keep only the last expert's per-expert side"):
        lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch))
    lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch, experts_shared_outer_loras=False))
    lora_arguments._assert_adapter_layout_defined(_gpt_oss_export_args(monkeypatch, save_hf=None))
