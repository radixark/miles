"""The deep_ep.Buffer.__init__ wrapper restores the memory saver's region flag instead of forcing it on (#807)."""

import importlib
import sys
import types

import pytest

_PACKAGE = "miles.backends.megatron_utils"


def _reimport_against(monkeypatch, deep_ep, tms):
    """Import the package once more with fake deep_ep / torch_memory_saver in place, so its import-time wrapper
    binds to the fakes. The caller's fixture puts the real module back afterwards."""
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", types.SimpleNamespace(torch_memory_saver=tms))
    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    sys.modules.pop(_PACKAGE, None)
    return importlib.import_module(_PACKAGE)


@pytest.fixture
def wrapped_buffer():
    """A fake deep_ep.Buffer wrapped by the package's import-time hook, plus the recorded region-flag writes.
    Restores the real package module and its parent attribute on teardown so later tests see the real one."""
    parent = importlib.import_module("miles.backends")
    original_module = sys.modules.get(_PACKAGE)
    original_attr = getattr(parent, "megatron_utils", None)
    calls: list[bool] = []
    state = {"interesting": True}

    def set_region(value):
        state["interesting"] = bool(value)
        calls.append(bool(value))

    cdll = types.SimpleNamespace(
        tms_get_interesting_region=lambda: state["interesting"], tms_set_interesting_region=set_region
    )
    tms = types.SimpleNamespace(_impl=types.SimpleNamespace(_binary_wrapper=types.SimpleNamespace(cdll=cdll)))

    class Buffer:
        def __init__(self, *args, **kwargs):
            self.args = args

    deep_ep = types.ModuleType("deep_ep")
    deep_ep.Buffer = Buffer
    try:
        yield Buffer, calls, state, tms, deep_ep
    finally:
        if original_module is not None:
            sys.modules[_PACKAGE] = original_module
        else:
            sys.modules.pop(_PACKAGE, None)
        if original_attr is not None:
            parent.megatron_utils = original_attr


@pytest.mark.parametrize("initial", [True, False])
def test_init_restores_the_region_flag_it_found(wrapped_buffer, monkeypatch, initial):
    Buffer, calls, state, tms, deep_ep = wrapped_buffer
    _reimport_against(monkeypatch, deep_ep, tms)
    state["interesting"] = initial
    calls.clear()

    Buffer(1, 2)

    assert calls == [False, initial]
    assert state["interesting"] is initial


@pytest.mark.parametrize("initial", [True, False])
def test_init_restores_the_flag_when_the_wrapped_init_raises(wrapped_buffer, monkeypatch, initial):
    Buffer, calls, state, tms, deep_ep = wrapped_buffer

    def boom(self, *args, **kwargs):
        raise RuntimeError("init failed")

    Buffer.__init__ = boom
    _reimport_against(monkeypatch, deep_ep, tms)
    state["interesting"] = initial
    calls.clear()

    with pytest.raises(RuntimeError, match="init failed"):
        Buffer()

    assert calls == [False, initial]
    assert state["interesting"] is initial


def test_init_is_a_plain_passthrough_when_the_saver_is_not_initialized(wrapped_buffer, monkeypatch):
    Buffer, calls, state, tms, deep_ep = wrapped_buffer
    tms._impl = None
    _reimport_against(monkeypatch, deep_ep, tms)

    Buffer(3)

    assert calls == []
