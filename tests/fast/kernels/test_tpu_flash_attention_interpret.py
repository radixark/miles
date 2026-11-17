"""miles.kernels.tpu.flash_attention on CPU: the segment block skipping, and the Pallas kernel in interpret mode
against dense attention. Guards the private splash-attention calls at the TPU image's JAX version."""

import functools
import importlib
import importlib.util
import inspect
import math
import sys
from types import ModuleType

import numpy as np
import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

jax = pytest.importorskip("jax")
jnp = jax.numpy


# the functions the module hands torch_tpu's Pallas bridge, recorded by the stub
_BRIDGED: list = []


def _import_flash_attention():
    if importlib.util.find_spec("torch_tpu") is not None:
        return importlib.import_module("miles.kernels.tpu.flash_attention")
    pallas = ModuleType("torch_tpu._internal.pallas")
    pallas.custom_jax_kernel = lambda fn, name: _BRIDGED.append(fn) or fn
    internal = ModuleType("torch_tpu._internal")
    internal.pallas = pallas
    torch_tpu = ModuleType("torch_tpu")
    torch_tpu._internal = internal
    with pytest.MonkeyPatch.context() as mp:
        for module in (torch_tpu, internal, pallas):
            mp.setitem(sys.modules, module.__name__, module)
        return importlib.import_module("miles.kernels.tpu.flash_attention")


fa = _import_flash_attention()


def _segment_ids(lengths: list[int]) -> np.ndarray:
    return np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)


def _random_segment_ids(seq_len: int, seed: int) -> np.ndarray:
    starts = np.random.default_rng(seed).random(seq_len) < 0.004
    starts[0] = False
    return np.cumsum(starts).astype(np.int32)


_SEQ_LEN, _BLOCK = 2048, 256  # an 8 x 8 block grid
_MASK_CASES = {
    "random": _random_segment_ids(_SEQ_LEN, seed=0),
    "ends_on_block_boundary": _segment_ids([3 * _BLOCK, 300, _SEQ_LEN - 3 * _BLOCK - 300]),
    "one_segment": _segment_ids([_SEQ_LEN]),
    "many_tiny_segments": _segment_ids([7] * (_SEQ_LEN // 7) + [_SEQ_LEN % 7]),
}


@pytest.mark.parametrize("is_dkv", [False, True], ids=["fwd", "dkv"])
@pytest.mark.parametrize("segment_ids", _MASK_CASES.values(), ids=_MASK_CASES.keys())
def test_skip_blocks_zeroes_exactly_the_blocks_sharing_no_segment(segment_ids, is_dkv):
    num_heads, num_blocks = 2, _SEQ_LEN // _BLOCK
    _, kernel = fa._kernel(_SEQ_LEN, num_heads, _BLOCK)
    causal = kernel.dkv_mask_info if is_dkv else kernel.fwd_mask_info

    info = fa._skip_blocks_across_segments(causal, jnp.asarray(segment_ids), _BLOCK, _BLOCK, is_dkv=is_dkv)

    blocks = segment_ids.reshape(num_blocks, _BLOCK)
    shares_segment = np.array(
        [[np.intersect1d(blocks[i], blocks[j]).size > 0 for j in range(num_blocks)] for i in range(num_blocks)]
    )
    causal_mask = np.asarray(causal.block_mask)  # [1, q blocks, kv blocks]: the heads share one causal mask
    assert causal_mask.shape[-2:] == (num_blocks, num_blocks)
    np.testing.assert_array_equal(np.asarray(info.block_mask), np.where(shares_segment, causal_mask, 0))
    # kept blocks keep the causal kernel's next-block indices, skipped ones point at block 0; the rest is untouched
    kept = np.asarray(info.block_mask) != 0
    np.testing.assert_array_equal(np.asarray(info.data_next), np.where(kept, np.asarray(causal.data_next), 0))
    for field in set(causal._fields) - {"block_mask", "data_next"}:
        assert (getattr(info, field) is None) == (getattr(causal, field) is None), field
        if getattr(causal, field) is not None:
            np.testing.assert_array_equal(np.asarray(getattr(info, field)), np.asarray(getattr(causal, field)))


def _dense_attention(q, k, v, segment_ids):
    """Causal attention within each segment, in float32 at full matmul precision. q: [B, Hq, L, D]."""
    k, v = (jnp.repeat(x, q.shape[1] // k.shape[1], axis=1) for x in (k, v))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision="highest") / math.sqrt(q.shape[-1])
    positions = jnp.arange(q.shape[2])
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    mask = same_segment[:, None] & (positions[:, None] >= positions[None, :])
    probs = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", probs, v, precision="highest")


def test_interpreted_kernel_matches_dense_attention():
    batch, q_heads, kv_heads, seq_len, head_dim = 2, 4, 2, 2 * fa.BLOCK, 128
    # row 0: segments cross the block boundary at 1024; row 1: a segment ends on it, so its row skips block (1, 0).
    # Both end in a padding segment.
    segment_ids = jnp.asarray(np.stack([_segment_ids([700, 900, 300, 148]), _segment_ids([1024, 1000, 24])]))
    keys = jax.random.split(jax.random.key(0), 4)
    q, grad_out = (jax.random.normal(key, (batch, q_heads, seq_len, head_dim)) for key in keys[:2])
    k, v = (jax.random.normal(key, (batch, kv_heads, seq_len, head_dim)) for key in keys[2:])

    out, lse = jax.jit(functools.partial(fa._forward, interpret=True))(q, k, v, segment_ids)
    grads = jax.jit(functools.partial(fa._backward, interpret=True))(q, k, v, out, lse, grad_out, segment_ids)

    ref_out, ref_vjp = jax.vjp(lambda q, k, v: _dense_attention(q, k, v, segment_ids), q, k, v)
    # float32 throughout and full-precision matmuls on CPU, so the kernel differs from the reference only in summation
    # order and the online-softmax rescaling: <= 5e-6 absolute on values up to ~7. A wrong mask, head mapping or scale
    # is off by O(1).
    for name, got, want in zip(("out", "dq", "dk", "dv"), (out, *grads), (ref_out, *ref_vjp(grad_out)), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=5e-5, err_msg=name)


@pytest.mark.skipif(importlib.util.find_spec("torch_tpu") is not None, reason="the stub records the bridged functions")
def test_bridged_kernels_take_tensors_only():
    """The bridge traces every parameter, so a defaulted flag (`interpret=False`) reaches Pallas as a tracer."""
    assert len(_BRIDGED) == 2
    for fn in _BRIDGED:
        params = inspect.signature(fn).parameters.values()
        assert all(p.default is inspect.Parameter.empty for p in params), inspect.signature(fn)
        assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params), inspect.signature(fn)
