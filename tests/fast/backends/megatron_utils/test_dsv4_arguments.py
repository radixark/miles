from argparse import ArgumentParser, Namespace

import pytest

from miles_plugins.models.deepseek_v4.arguments import (
    DSV4_SPEC_MODULE,
    add_dsv4_arguments,
    is_dsv4_model,
    normalize_dsv4_args,
)


def _parse(*argv: str) -> Namespace:
    parser = ArgumentParser()
    add_dsv4_arguments(parser)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--dsa-kernel-backend", default=None)
    parser.add_argument("--spec", nargs="*", default=[DSV4_SPEC_MODULE, "get_dsv4_spec"])
    return parser.parse_args(argv)


def test_only_the_dsv4_spec_triggers_normalization():
    assert is_dsv4_model(_parse())

    other = _parse()
    other.spec = ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]
    assert not is_dsv4_model(other)

    unspecced = _parse()
    unspecced.spec = None
    assert not is_dsv4_model(unspecced)


def test_model_shape_reaches_the_megatron_fields():
    args = _parse("--dsv4-compress-ratios", "0", "4", "128")
    normalize_dsv4_args(args)

    assert args.csa_window_size == 128
    assert args.csa_compress_ratios == [0, 4, 128]
    assert args.csa_compress_rotary_base == 160000
    assert args.o_groups == 8
    assert args.o_lora_rank == 1024
    assert args.moe_n_hash_layers == 3
    assert args.num_residual_streams == 4
    assert args.mhc_sinkhorn_iterations == 20


def test_impl_selects_the_attention_variant():
    miles = _parse()
    normalize_dsv4_args(miles)
    assert miles.experimental_attention_variant == "dsv4"

    megatron = _parse("--dsv4-impl", "megatron")
    normalize_dsv4_args(megatron)
    assert megatron.experimental_attention_variant == "dsv4_hybrid"
    assert megatron.enable_hyper_connections


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--dsv4-impl", "megatron", "--tensor-model-parallel-size", "8"), "tensor-model-parallel-size 1"),
        (("--dsv4-impl", "megatron", "--dsa-kernel-backend", "tilelang"), "does not support"),
        (("--dsa-kernel-backend", "cudnn"), "ignores cuDNN"),
    ],
    ids=["megatron-needs-tp1", "megatron-rejects-tilelang", "miles-rejects-cudnn"],
)
def test_unsupported_combinations_fail_at_parse_time(argv, message):
    """Megatron would only assert deep inside config post-init, or silently mis-run."""
    with pytest.raises(ValueError, match=message):
        normalize_dsv4_args(_parse(*argv))
