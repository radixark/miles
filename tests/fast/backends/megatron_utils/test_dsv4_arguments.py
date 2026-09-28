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


@pytest.fixture(autouse=True)
def default_indexer_impl(monkeypatch):
    monkeypatch.delenv("V4_INDEXER_IMPL", raising=False)


def test_qat_defaults_remain_legacy():
    args = _parse()
    assert args.dsv4_kv_qat == args.dsv4_index_qat == "legacy"
    normalize_dsv4_args(args)


@pytest.mark.parametrize("kv_mode", ["legacy", "off", "fp8_ue8m0"])
@pytest.mark.parametrize("index_mode", ["legacy", "off", "fp8_ue8m0", "fp8_dynamic"])
def test_qat_modes_parse_independently(kv_mode, index_mode):
    args = _parse("--dsv4-impl", "miles", "--dsv4-kv-qat", kv_mode, "--dsv4-index-qat", index_mode)
    normalize_dsv4_args(args)
    assert args.dsv4_kv_qat == kv_mode
    assert args.dsv4_index_qat == index_mode


@pytest.mark.parametrize("flag,value", [("--dsv4-kv-qat", "fp8_dynamic"), ("--dsv4-index-qat", "fp8")])
def test_unknown_qat_modes_fail_parsing(flag, value):
    with pytest.raises(SystemExit):
        _parse(flag, value)


@pytest.mark.parametrize("flag", ["--dsv4-kv-qat", "--dsv4-index-qat"])
def test_native_megatron_rejects_explicit_qat_policy(flag):
    with pytest.raises(ValueError, match="require --dsv4-impl miles"):
        normalize_dsv4_args(_parse(flag, "off"))


def test_non_v4_indexer_rejects_explicit_index_policy(monkeypatch):
    monkeypatch.setenv("V4_INDEXER_IMPL", "megatron")
    with pytest.raises(ValueError, match="require V4_INDEXER_IMPL=tilelang"):
        normalize_dsv4_args(_parse("--dsv4-impl", "miles", "--dsv4-index-qat", "off"))
    # KV-only overrides do not reconfigure the alternative indexer.
    normalize_dsv4_args(_parse("--dsv4-impl", "miles", "--dsv4-kv-qat", "off"))


def test_missing_qat_attributes_preserve_old_programmatic_callers():
    args = _parse()
    del args.dsv4_kv_qat, args.dsv4_index_qat
    normalize_dsv4_args(args)
    assert args.experimental_attention_variant == "dsv4_hybrid"


def test_only_the_dsv4_spec_triggers_normalization():
    assert is_dsv4_model(_parse())

    other = _parse()
    other.spec = ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]
    assert not is_dsv4_model(other)

    unspecced = _parse()
    unspecced.spec = None
    assert not is_dsv4_model(unspecced)


def test_impl_selects_the_attention_variant():
    default = _parse()
    normalize_dsv4_args(default)
    assert default.experimental_attention_variant == "dsv4_hybrid"

    miles = _parse("--dsv4-impl", "miles")
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
        (("--dsv4-impl", "miles", "--dsa-kernel-backend", "cudnn"), "ignores cuDNN"),
    ],
    ids=["megatron-needs-tp1", "megatron-rejects-tilelang", "miles-rejects-cudnn"],
)
def test_unsupported_combinations_fail_at_parse_time(argv, message):
    """Megatron would only assert deep inside config post-init, or silently mis-run."""
    with pytest.raises(ValueError, match=message):
        normalize_dsv4_args(_parse(*argv))
