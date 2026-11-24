from collections.abc import Callable

import pytest
from tests.fast.backends.megatron_utils.test_dsa_arguments import _args, megatron_defaults

__all__ = ["megatron_defaults"]


class TestMegatronDefaultConsumers:
    @pytest.mark.parametrize(
        "optimizer,disabled,slots,distributed",
        [
            (None, False, 0, True),
            ("ADAM", False, 0, True),
            ("muon", False, 0, False),
            ("adam", True, 0, False),
            ("adam", False, 2, False),
        ],
    )
    def test_optimizer_and_adapter_modes_select_the_supported_gradient_transport(
        self, megatron_defaults: Callable, optimizer: str | None, disabled: bool, slots: int, distributed: bool
    ) -> None:
        """Muon, disabled optimization, and per-slot adapters cannot accidentally select distributed Adam state."""
        args = _args(
            dsa_impl="miles",
            optimizer=optimizer,
            debug_disable_optimizer=disabled,
            multi_lora_n_adapters=slots,
            fp16=False,
            seq_length=128,
            vocab_size=None,
            padded_vocab_size=None,
            tokenizer_model="hf",
            tokenizer_type="HuggingFaceTokenizer",
            hf_checkpoint="hf",
        )

        result = megatron_defaults(args)

        assert result.use_distributed_optimizer is distributed
        assert result.bf16 is True
        assert result.rope_type == "yarn"

    def test_unsupported_true_on_policy_rejects_before_backend_defaults_are_required(
        self, megatron_defaults: Callable
    ) -> None:
        """Unsupported true-on-policy selection fails before backend initialization proceeds."""
        with pytest.raises(NotImplementedError, match="not supported on the megatron backend"):
            megatron_defaults(_args(true_on_policy_mode=True))
