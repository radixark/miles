"""The FSDP Triton attention bridge must survive SGLang relocating its kernels.

SGLang moved ``extend_attention_fwd_unified`` from
``sglang.srt.layers.attention.triton_ops`` to ``sglang.kernels.ops.attention``, and the
relocated kernel takes explicit KV scales. The bridge resolves module and signature
lazily, so both layouts keep working.
"""

import builtins
import functools
import sys
import types

import pytest

from miles.backends.fsdp_utils.sglang_attn_bridge import hf_sglang_triton_patch as bridge

RELOCATED_MODULE = "sglang.kernels.ops.attention.extend_attention"
LEGACY_MODULE = "sglang.srt.layers.attention.triton_ops.extend_attention"


def _relocated_kernel(q, o, k_buffer, v_buffer, k_scale, v_scale, qo_indptr, kv_indptr, kv_indices, prefix_lens,
                      max_len_extend, is_causal=True):
    return "relocated"


def _legacy_kernel(q, o, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, prefix_lens,
                   max_len_extend, is_causal=True):
    return "legacy"


def _install_kernel(monkeypatch, module_name, kernel):
    module = types.ModuleType(module_name)
    module.extend_attention_fwd_unified = kernel
    monkeypatch.setitem(sys.modules, module_name, module)
    return kernel


@pytest.fixture(autouse=True)
def _reset_lookup_cache():
    bridge._get_extend_attention_fwd.cache_clear()
    yield
    bridge._get_extend_attention_fwd.cache_clear()


def test_relocated_kernel_is_used_with_bf16_identity_kv_scales(monkeypatch):
    kernel = _install_kernel(monkeypatch, RELOCATED_MODULE, _relocated_kernel)

    resolved = bridge._get_extend_attention_fwd()

    assert isinstance(resolved, functools.partial)
    assert resolved.func is kernel
    assert resolved.keywords == {"k_scale": 1.0, "v_scale": 1.0}


def test_legacy_kernel_is_used_when_the_relocated_module_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, RELOCATED_MODULE, None)
    kernel = _install_kernel(monkeypatch, LEGACY_MODULE, _legacy_kernel)

    resolved = bridge._get_extend_attention_fwd()

    assert resolved is kernel


def test_a_missing_nested_dependency_is_not_masked_by_the_fallback(monkeypatch):
    """Only the kernel module itself may be missing; anything else must surface."""
    monkeypatch.setitem(sys.modules, RELOCATED_MODULE, None)
    _install_kernel(monkeypatch, LEGACY_MODULE, _legacy_kernel)
    missing = f"{RELOCATED_MODULE}.third_party"
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == RELOCATED_MODULE:
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(ModuleNotFoundError, match="third_party"):
        bridge._get_extend_attention_fwd()
