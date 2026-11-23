"""DeepSeek-V4.1 across pipeline stages: the carried state arrives bit for bit and its gradients travel back exactly;
a boundary carrying a crossing no run qualified is refused."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.top import dsv41_pipeline
from miles_plugins.top.dsv41_pipeline import _LATENT, _PRE, INBOUND, _Pack, _Unpack, crossing


def _send(streams, pre, latent, runtime):
    packed = _Pack.apply(streams, [(_PRE, 0), (_LATENT, 2)], runtime, pre, latent)
    received = packed.detach().clone().requires_grad_()
    return packed, received


def test_state_arrives_bit_for_bit_and_gradients_travel_back_exactly():
    g = torch.Generator().manual_seed(0)
    streams = torch.randn(6, 1, 512, generator=g).bfloat16().requires_grad_()
    pre = torch.randn(6, 4, generator=g).requires_grad_()
    latent = torch.randn(3, 512, generator=g).bfloat16().requires_grad_()
    runtime = [
        (1000, torch.randint(-100, 100, (5, 7), generator=g, dtype=torch.int8)),
        (1200, torch.randint(-1, 9, (6, 4), generator=g, dtype=torch.int32)),
        (9000, torch.rand(6, 3, generator=g) > 0.5),
    ]
    packed, received = _send(streams, pre, latent, runtime)
    INBOUND.clear()
    out_streams, out_pre, out_latent = _Unpack.apply(received)
    assert INBOUND.keys == [(_PRE, 0), (_LATENT, 2)]
    assert torch.equal(out_streams, streams) and torch.equal(out_pre, pre) and torch.equal(out_latent, latent)
    for (key, t), (k, r) in zip(runtime, INBOUND.runtime, strict=True):
        assert key == k and r.dtype == t.dtype and torch.equal(r, t)
    # The next stage's loss, then its input gradient sent back through the sender's pack.
    w = torch.randn(512, generator=g)
    loss = (out_streams.float() * w).sum() + (out_pre * 3).square().sum() + (out_latent.float() * w).sum()
    loss.backward()
    packed.backward(received.grad)
    direct = [x.detach().clone().requires_grad_() for x in (streams, pre, latent)]
    ((direct[0].float() * w).sum() + (direct[1] * 3).square().sum() + (direct[2].float() * w).sum()).backward()
    for sent, ref in zip((streams, pre, latent), direct, strict=True):
        assert torch.equal(sent.grad, ref.grad)


def test_an_unused_segment_returns_a_zero_gradient():
    streams = torch.ones(2, 1, 512).bfloat16().requires_grad_()
    pre, latent = torch.ones(2, 4).requires_grad_(), torch.ones(1, 512).bfloat16().requires_grad_()
    packed, received = _send(streams, pre, latent, [])
    out_streams, _, _ = _Unpack.apply(received)
    out_streams.float().sum().backward()
    packed.backward(received.grad)
    assert torch.equal(latent.grad, torch.zeros_like(latent)) and torch.equal(pre.grad, torch.zeros_like(pre))


def _slice():
    return SimpleNamespace(
        num_hidden_layers=8,
        compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 6],
        candidate_source_layer_id=4,
    )


def _full():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
    )


def test_crossings_of_the_slice_and_the_full_model():
    assert crossing(_slice(), 3) == {"pre", "ratio2"}
    assert crossing(_slice(), 5) == {"pre", "ratio1", "candidates"}
    assert crossing(_slice(), 4) == {"pre"}
    full = _full()
    assert {b: crossing(full, b) for b in range(5, 40, 5)} == {
        5: {"pre", "ratio2"},
        10: {"pre", "ratio2"},
        15: {"pre", "ratio2"},
        20: {"pre"},
        25: {"pre", "ratio1", "candidates"},
        30: {"pre", "ratio1", "candidates"},
        35: {"pre", "ratio1", "candidates"},
    }


def test_an_unqualified_boundary_is_refused(monkeypatch):
    """A boundary inside both ratios' reach (a layout no run qualified) is refused instead of sent."""
    cfg = _slice()
    cfg.compress_ratios = [0, 0, 2, 1, 2, 1, 1, 1, 0, 0, 0]
    cfg.kv_source_layer_ids, cfg.index_source_layer_ids = [2, 3], [2, 3]
    monkeypatch.setattr("miles_plugins.top.dsv4_runtime.dsv4_hf_config", lambda: cfg)
    layer = SimpleNamespace(layer_number=5)
    model = SimpleNamespace(pre_process=False, post_process=True, decoder=SimpleNamespace(layers=[layer]))
    with pytest.raises(NotImplementedError, match="not qualified"):
        dsv41_pipeline.install(model)
