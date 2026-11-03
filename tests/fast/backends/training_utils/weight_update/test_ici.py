"""The ICI sender against the layout, store keys and tags that sglang_tpu's shared-slice receiver reads."""

import json
import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols import ici
from miles.platforms.tpu_shared_slice import SharedSlice, SharedSliceSpec


class _FakeBackend:
    def __init__(self, *, engine_receives: bool = True):
        self.sends: list[tuple[int, int]] = []
        self._engine_receives = engine_receives

    def send(self, tensors, dst, tag):
        self.sends.append((dst, tag))
        return SimpleNamespace(is_completed=lambda: self._engine_receives, wait=lambda: None)


class _FakeStore:
    def __init__(self):
        self.values: dict[str, str] = {}

    def set(self, key, value):
        self.values[key] = value


class _RecordingApiClient:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            return {"success": True}

        return method


def _connect(monkeypatch, *, rank: int, failed_on_another_rank: bool = False, engine_receives: bool = True):
    def all_reduce(tensor, group):
        if failed_on_another_rank:
            tensor.fill_(1)

    monkeypatch.setattr(ici.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(ici.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(ici.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(ici, "get_gloo_group", lambda: None)
    spec = SharedSliceSpec(rank_offset=0, size=4, store_host="store", store_port=1)
    shared_slice = SharedSlice(spec=spec, backend=_FakeBackend(engine_receives=engine_receives), store=_FakeStore())
    monkeypatch.setattr(ici.current_platform, "shared_slice", shared_slice)
    calls: list[tuple[str, dict]] = []
    protocol = ici.UpdateWeightFromIci(Namespace(ft_components=[], update_weight_engine_request_timeout=0.05))
    protocol.connect(
        [_RecordingApiClient(calls)],
        [2],
        [0],
        ["cell-0"],
        parallel_state=None,
        placement=ici.UpdateWeightFromIci.required_placement,
        selector="all",
    )
    return protocol, shared_slice, calls


def _receive(packed: torch.Tensor, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """sglang_tpu.shared_slice.SharedSliceReceiver's view of a packed bucket, from the request's dtypes and shapes."""
    received, offset = [], 0
    for tensor in tensors:
        num_bytes = tensor.numel() * tensor.dtype.itemsize
        received.append(packed[offset // 2 : (offset + num_bytes) // 2].view(tensor.dtype).view(tensor.shape))
        offset += max(1, math.ceil(num_bytes / 4096)) * 4096
    assert packed.numel() * 2 == offset, "the engine allocates a receive buffer of another size"
    return received


def test_the_engine_views_every_tensor_back_out_of_the_packed_bucket():
    """A tensor at another offset than the engine computes reaches it as a neighbour's bytes."""
    generator = torch.Generator().manual_seed(0)
    tensors = [
        torch.randn(3, 5, generator=generator).to(torch.bfloat16),
        torch.randn(4, 6, generator=generator).t(),
        torch.empty(0, 4, dtype=torch.bfloat16),
        torch.arange(3000, dtype=torch.int64),
    ]

    packed = ici.pack_bucket(tensors)

    assert packed.dtype == torch.int16
    for received, sent in zip(_receive(packed, tensors), tensors, strict=True):
        assert torch.equal(received, sent)


def test_rank_1_sends_to_engine_rank_1_under_tags_that_never_reset(monkeypatch):
    """Engine TP rank i reads trainer rank i's names under (src, dst, tag), its tag counting from 1 for the
    receiver's whole life; a wrong key or tag leaves the engine waiting."""
    protocol, shared_slice, calls = _connect(monkeypatch, rank=1)

    protocol.send_bucket([("a", torch.zeros(2)), ("b", torch.ones(3))])
    protocol.begin_sync(weight_version=2, iter_buckets=None)
    protocol.send_bucket([("a", torch.zeros(2))])

    assert shared_slice.backend.sends == [(3, 1), (3, 2)]
    assert shared_slice.store.values == {
        "bucket_names:1:3:1": json.dumps(["a", "b"]),
        "bucket_names:1:3:2": json.dumps(["a"]),
    }
    assert calls == [], "only rank 0 calls the engine"


def test_rank_0_binds_the_engine_receiver_then_posts_each_bucket(monkeypatch):
    """The bucket request must name the group bound to the engine's shared-slice receiver."""
    protocol, shared_slice, calls = _connect(monkeypatch, rank=0)

    protocol.send_bucket([("a", torch.zeros(2, dtype=torch.bfloat16))])

    (init_name, init), (update_name, update) = calls
    assert (init_name, init["backend"]) == ("init_weights_update_group", "tpu_shared_slice")
    assert (update_name, update["names"], update["group_name"]) == (
        "update_weights_from_distributed",
        ["a"],
        init["group_name"],
    )
    assert shared_slice.backend.sends == [(2, 1)]


def test_every_rank_raises_when_the_engine_fails_a_bucket(monkeypatch):
    """Only rank 0 hears from the engine; a rank that carried on would block in its next send."""
    protocol, _, _ = _connect(monkeypatch, rank=1, failed_on_another_rank=True)

    with pytest.raises(RuntimeError, match="failed on another trainer rank"):
        protocol.send_bucket([("a", torch.zeros(2))])


def test_a_send_the_engine_never_receives_times_out(monkeypatch):
    """An engine that rejected the request posts no receive; torch_tpu would then block the send for 30 minutes."""
    protocol, _, _ = _connect(monkeypatch, rank=1, engine_receives=False)
    buffer_holder = protocol._unreceived_buffers

    with pytest.raises(TimeoutError, match="did not receive bucket 1"):
        protocol.send_bucket([("a", torch.zeros(2))])

    assert len(buffer_holder) == 1, "the posted send's buffer must stay alive"
