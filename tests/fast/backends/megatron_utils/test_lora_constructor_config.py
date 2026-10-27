import pytest
from tests.fast.backends.megatron_utils.lora_constructor_fixtures import _BridgeConstructors, bridge_constructors
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils.lora.slots import create_multi_lora_instance
from miles.backends.megatron_utils.lora.utils import create_lora_instance

__all__ = ["bridge_constructors"]


class TestLoraConstructorConfig:
    @pytest.mark.parametrize("kind", ["lora", "canonical_lora"])
    def test_miles_selects_the_adapter_and_preserves_explicit_rank_dropout_and_initialization(
        self, kind: str, bridge_constructors: _BridgeConstructors
    ) -> None:
        """Miles forwards explicit LoRA settings to the selected SDK constructor boundary."""
        args = make_trainer_args(
            lora_type=kind,
            lora_rank=4,
            lora_alpha=12,
            lora_dropout=0.25,
            lora_A_init_method="normal",
            lora_B_init_method="zero",
            experts_shared_outer_loras=False,
        )

        adapter = create_lora_instance(args, target_modules=["linear_qkv"])

        expected_type = {"lora": bridge_constructors.lora, "canonical_lora": bridge_constructors.canonical_lora}[kind]
        assert type(adapter) is expected_type
        assert adapter.dim == 4
        assert adapter.alpha == 12
        assert adapter.dropout == 0.25
        assert adapter.lora_A_init_method == "normal"
        assert adapter.lora_B_init_method == "zero"
        assert adapter.target_modules == ["linear_qkv"]
        if kind == "lora":
            assert adapter.share_expert_adapters is False
            assert adapter.experts_shared_outer_loras is False

    def test_multi_adapter_config_preserves_slot_count_and_dropout(
        self, bridge_constructors: _BridgeConstructors
    ) -> None:
        """Miles forwards slot count and initialization settings to the Multi-LoRA SDK boundary."""
        args = make_trainer_args(
            multi_lora_n_adapters=3,
            lora_rank=4,
            lora_alpha=12,
            lora_dropout=0.25,
            lora_A_init_method="normal",
            lora_B_init_method="zero",
        )

        adapter = create_multi_lora_instance(args, target_modules=["linear_qkv"])

        assert type(adapter) is bridge_constructors.multi_lora
        assert adapter.n_adapters == 3
        assert adapter.dim == 4
        assert adapter.alpha == 12
        assert adapter.dropout == 0.25
        assert adapter.lora_A_init_method == "normal"
        assert adapter.lora_B_init_method == "zero"
        assert adapter.target_modules == ["linear_qkv"]

    def test_canonical_adapter_rejects_the_shared_expert_mode(self, bridge_constructors: _BridgeConstructors) -> None:
        """Canonical LoRA cannot silently accept the standard adapter's shared-expert layout."""
        args = make_trainer_args(
            lora_type="canonical_lora", experts_shared_outer_loras=True, lora_B_init_method="zero"
        )

        with pytest.raises(AssertionError, match="requires the standard LoRA adapter type"):
            create_lora_instance(args, target_modules=["linear_qkv"])
