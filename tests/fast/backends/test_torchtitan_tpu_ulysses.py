"""Ulysses CP around the TPU flash attention, on two gloo ranks with a dense stand-in for the kernel.

The all-to-alls must hand the kernel every token of its heads, and the segment ids stay global, so the CP output and
gradients equal one rank attending over the whole sequence.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.fast.dist_utils import init_gloo, run_multiprocess

pytest.importorskip("torchtitan")

# three packed sequences and a padding tail; the 2-rank shard boundary (token 32) falls inside the second
_LENGTHS = (20, 26, 10, 8)
_HEADS, _KV_HEADS, _DIM = 4, 2, 8


def _dense_segment_causal_attention(self, q, k, v, *, attention_metadata, **kwargs):
    """[tokens, heads, dim] attention, causal within each segment: what the kernel computes."""
    segment_ids = attention_metadata
    causal = torch.ones(len(segment_ids), len(segment_ids), dtype=torch.bool).tril()
    mask = (segment_ids[:, None] == segment_ids[None, :]) & causal
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, enable_gqa=True
    )
    return out.transpose(0, 1)


def _inputs():
    generator = torch.Generator().manual_seed(0)
    num_tokens = sum(_LENGTHS)
    q, grad_out = (torch.randn(num_tokens, _HEADS, _DIM, generator=generator) for _ in range(2))
    k, v = (torch.randn(num_tokens, _KV_HEADS, _DIM, generator=generator) for _ in range(2))
    positions = torch.cat([torch.arange(n) for n in _LENGTHS])
    return q, k, v, grad_out, positions


def _forward_backward(attention, q, k, v, grad_out, segment_ids):
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    out = attention(q, k, v, attention_metadata=segment_ids)
    out.backward(grad_out)
    return out.detach(), q.grad, k.grad, v.grad


def _worker(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    from miles.backends.torchtitan_utils import config

    # the kernel needs jax and torch_tpu; the CP plumbing around it does not
    config.TpuFlashInnerAttention.__init__ = lambda self, _config: config.InnerAttention.__init__(self)
    config.TpuFlashInnerAttention.forward = _dense_segment_causal_attention
    config.spmd_mesh_group = lambda axis: torch.distributed.group.WORLD

    q, k, v, grad_out, positions = _inputs()
    segment_ids = config.TpuFlashInnerAttention.Config().build_attention_metadata(positions)
    expected = _forward_backward(config.TpuFlashInnerAttention(None), q, k, v, grad_out, segment_ids)

    shard = slice(rank * len(positions) // world_size, (rank + 1) * len(positions) // world_size)
    ulysses = config.UlyssesTpuFlashInnerAttention(None)
    got = _forward_backward(ulysses, q[shard], k[shard], v[shard], grad_out[shard], segment_ids)

    for name, actual, full in zip(("out", "dq", "dk", "dv"), got, expected, strict=True):
        torch.testing.assert_close(actual, full[shard], msg=lambda m, name=name: f"{name}: {m}")


def test_ulysses_cp_equals_attention_over_the_whole_sequence():
    run_multiprocess(_worker, world_size=2)
