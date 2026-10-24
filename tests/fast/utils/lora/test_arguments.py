from argparse import Namespace

from transformers import LlamaConfig

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

        hf_targets, adapter_targets = _resolve_lora_targets(args, config)

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
        assert adapter_targets == expected
        assert args.target_modules == "all-linear"
        assert args.exclude_modules == "q_proj"
