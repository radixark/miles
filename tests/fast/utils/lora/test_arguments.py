from argparse import Namespace

import pytest
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


class TestLoraWildcardLeafTargets:
    @staticmethod
    def _resolve(target_modules: str) -> list[str]:
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        args = Namespace(
            target_modules=target_modules,
            exclude_modules=None,
            multi_lora=False,
            megatron_to_hf_mode="bridge",
            lora_type="lora",
        )
        return _resolve_lora_targets(args, config)

    def test_bare_glob_leaf_expands_to_concrete_leaves(self) -> None:
        """SGLang serves targets by literal leaf, so a glob leaf becomes the leaves it selects."""
        assert self._resolve("*_proj") == [
            "down_proj",
            "gate_proj",
            "up_proj",
            "k_proj",
            "o_proj",
            "q_proj",
            "v_proj",
        ]

    def test_glob_leaf_keeps_its_prefix(self) -> None:
        assert self._resolve("*.self_attn.[qk]_proj") == ["*.self_attn.k_proj", "*.self_attn.q_proj"]

    def test_prefix_only_glob_is_kept(self) -> None:
        assert self._resolve("*.self_attn.q_proj,o_proj") == ["*.self_attn.q_proj", "o_proj"]

    def test_glob_leaf_spanning_segments_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="cannot be rewritten as concrete leaves"):
            self._resolve("model.layers.0.*")
