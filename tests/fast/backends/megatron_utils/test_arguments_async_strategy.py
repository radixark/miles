import pytest

pytest.importorskip("megatron.training.arguments")

from argparse import Namespace

from miles.backends.megatron_utils import arguments as megatron_arguments


def _args(**overrides) -> Namespace:
    """A Namespace carrying only the fields set_default_megatron_args touches."""
    base = dict(
        true_on_policy_mode=False,
        optimizer="adam",
        debug_disable_optimizer=False,
        multi_lora_n_adapters=0,
        fp16=False,
        seq_length=4096,
        async_save=True,
        async_strategy="nvrx",
        multi_latent_attention=False,
        rope_type="rope",
        vocab_size=None,
        padded_vocab_size=None,
        tokenizer_model="/models/qwen3",
        tokenizer_type="HuggingFaceTokenizer",
        hf_checkpoint="/models/qwen3",
    )
    base.update(overrides)
    return Namespace(**base)


def test_async_save_falls_back_to_mcore_without_nvrx(monkeypatch) -> None:
    """Without nvidia-resiliency-ext, --async-save downgrades to Megatron's in-tree saver."""
    monkeypatch.setattr(megatron_arguments, "_has_nvrx_async_ckpt_support", lambda: False)

    args = megatron_arguments.set_default_megatron_args(_args())

    assert args.async_strategy == "mcore"


def test_async_save_keeps_nvrx_when_available(monkeypatch) -> None:
    """With nvidia-resiliency-ext installed the NVRx strategy is left alone."""
    monkeypatch.setattr(megatron_arguments, "_has_nvrx_async_ckpt_support", lambda: True)

    args = megatron_arguments.set_default_megatron_args(_args())

    assert args.async_strategy == "nvrx"


def test_explicit_mcore_strategy_is_preserved(monkeypatch) -> None:
    """An explicitly requested mcore strategy is never probed for or rewritten."""
    monkeypatch.setattr(
        megatron_arguments,
        "_has_nvrx_async_ckpt_support",
        lambda: pytest.fail("NVRx availability must not be probed for a non-nvrx strategy"),
    )

    args = megatron_arguments.set_default_megatron_args(_args(async_strategy="mcore"))

    assert args.async_strategy == "mcore"


def test_sync_save_strategy_is_left_to_megatron(monkeypatch) -> None:
    """Without --async-save, Megatron's own validation owns the strategy value."""
    monkeypatch.setattr(
        megatron_arguments,
        "_has_nvrx_async_ckpt_support",
        lambda: pytest.fail("NVRx availability must not be probed when async save is off"),
    )

    args = megatron_arguments.set_default_megatron_args(_args(async_save=False))

    assert args.async_strategy == "nvrx"


def test_missing_async_strategy_attribute_is_tolerated(monkeypatch) -> None:
    """Older Megatron builds expose no async_strategy at all; the probe stays out of the way."""
    monkeypatch.setattr(
        megatron_arguments,
        "_has_nvrx_async_ckpt_support",
        lambda: pytest.fail("NVRx availability must not be probed without an async_strategy"),
    )
    args = _args()
    del args.async_strategy

    result = megatron_arguments.set_default_megatron_args(args)

    assert not hasattr(result, "async_strategy")
