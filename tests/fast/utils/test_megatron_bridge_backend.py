from types import SimpleNamespace

import pytest

from miles.utils.megatron_bridge_utils import apply_dsa_backend_args


@pytest.mark.parametrize(
    "option,explicit,expected",
    [("tilelang", None, "tilelang"), ("megatron", None, "none"), ("tilelang", "cudnn", "cudnn")],
)
def test_kernel_only_bridge_provider_uses_backend_explicit_selection(
    option: str, explicit: str | None, expected: str
) -> None:
    """Kernel-only providers use backend explicit selection ahead of the Miles fallback."""
    provider = SimpleNamespace(dsa_kernel_backend="original")
    args = SimpleNamespace(backend=SimpleNamespace(dsa_kernel_backend=explicit), dsa_attention_backend=option)

    apply_dsa_backend_args(provider, args)

    assert provider.dsa_kernel_backend == expected


def test_attention_capable_bridge_provider_preserves_separate_attention_selection() -> None:
    """An attention-capable provider receives the Miles selection without a kernel override."""
    provider = SimpleNamespace(dsa_kernel_backend="original", dsa_attention_backend="original")
    args = SimpleNamespace(backend=SimpleNamespace(dsa_kernel_backend="cudnn"), dsa_attention_backend="tilelang")

    apply_dsa_backend_args(provider, args)

    assert provider.dsa_attention_backend == "tilelang"
    assert provider.dsa_kernel_backend == "original"
