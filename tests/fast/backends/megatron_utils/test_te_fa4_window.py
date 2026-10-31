import importlib
import sys
from types import ModuleType

import pytest

from miles.backends.megatron_utils.te_fa4_window import patch_te_flash_attn_4_window, te_window_to_fa4

TE_BACKENDS = "transformer_engine.pytorch.attention.dot_product_attention.backends"


def _fa4_resolve_causal_local_window(causal, window_size_left, window_size_right):
    """flash_attn/cute/interface.py::_resolve_causal_local_window as of flash-attn-4 4.0.0b30..4.0.0b36."""
    if (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left < 0
        and window_size_right < 0
    ):
        window_size_left = None
        window_size_right = None
    if causal:
        window_size_right = 0
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        ((-1, 0), (None, 0)),
        ((-1, -1), (None, None)),
        ((128, 0), (128, 0)),
        ((128, 128), (128, 128)),
        ((0, 0), (0, 0)),
        ((None, None), (None, None)),
        (None, None),
    ],
)
def test_te_window_to_fa4_maps_only_negative_sides_to_none(window_size, expected) -> None:
    """A TE -1 side becomes FA4's None; sliding-window sizes and None pass through."""
    assert te_window_to_fa4(window_size) == expected


def test_te_causal_window_resolves_to_causal_only_after_translation() -> None:
    """FA4 reads TE's raw (-1, 0) as a local window with left=-1; the translated window is causal."""
    assert _fa4_resolve_causal_local_window(True, -1, 0) == (False, True, -1, 0)
    assert _fa4_resolve_causal_local_window(True, *te_window_to_fa4((-1, 0))) == (True, False, None, None)


@pytest.mark.parametrize(("causal", "window_size"), [(True, (128, 0)), (False, (128, 128)), (False, (-1, -1))])
def test_translation_keeps_fa4_resolution_of_other_te_windows(causal, window_size) -> None:
    """Sliding windows and TE's no-mask window resolve exactly as FA4 resolved them before."""
    assert _fa4_resolve_causal_local_window(causal, *te_window_to_fa4(window_size)) == (
        _fa4_resolve_causal_local_window(causal, *window_size)
    )


def _fake_te_backends(monkeypatch: pytest.MonkeyPatch, *, with_fa4: bool) -> tuple[ModuleType, list[dict]]:
    calls = []

    def fa4_func(*args, **kwargs):
        calls.append(kwargs)
        return args

    backends = ModuleType(TE_BACKENDS)
    backends.flash_attn_func_v4 = fa4_func if with_fa4 else None
    backends.flash_attn_varlen_func_v4 = fa4_func if with_fa4 else None
    monkeypatch.setitem(sys.modules, TE_BACKENDS, backends)
    return backends, calls


@pytest.mark.parametrize("entry_point", ["flash_attn_func_v4", "flash_attn_varlen_func_v4"])
def test_patch_translates_the_window_te_passes_to_fa4(monkeypatch: pytest.MonkeyPatch, entry_point: str) -> None:
    """After patching, both TE FA4 entry points hand FA4 None for TE's unbounded sides."""
    backends, calls = _fake_te_backends(monkeypatch, with_fa4=True)

    patch_te_flash_attn_4_window()
    output = getattr(backends, entry_point)("q", "k", "v", causal=True, window_size=(-1, 0), num_splits=1)
    getattr(backends, entry_point)("q", "k", "v", causal=True, window_size=(128, 0))
    getattr(backends, entry_point)("q", "k", "v", causal=True)

    assert output == ("q", "k", "v")
    assert calls == [
        {"causal": True, "window_size": (None, 0), "num_splits": 1},
        {"causal": True, "window_size": (128, 0)},
        {"causal": True},
    ]


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second patch leaves the already wrapped entry points in place."""
    backends, _ = _fake_te_backends(monkeypatch, with_fa4=True)

    patch_te_flash_attn_4_window()
    wrapped = backends.flash_attn_varlen_func_v4
    patch_te_flash_attn_4_window()

    assert backends.flash_attn_varlen_func_v4 is wrapped
    assert wrapped.__wrapped__.__name__ == "fa4_func"


def test_patch_leaves_te_without_fa4_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """TE built without flash-attn-4 keeps its None entry points."""
    backends, _ = _fake_te_backends(monkeypatch, with_fa4=False)

    patch_te_flash_attn_4_window()

    assert backends.flash_attn_func_v4 is None
    assert backends.flash_attn_varlen_func_v4 is None


def test_patch_without_transformer_engine_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Transformer Engine the patch returns quietly."""
    monkeypatch.setitem(sys.modules, TE_BACKENDS, None)
    with pytest.raises(ImportError):
        importlib.import_module(TE_BACKENDS)

    patch_te_flash_attn_4_window()
