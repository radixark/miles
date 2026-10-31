"""The ICI sender's wire contract with sglang_tpu's shared-slice receiver: layout, names, tags, pairing."""

import json
from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.ici import UpdateWeightFromIci
from miles.platforms import tpu_shared_slice

_SHARED = tpu_shared_slice.SharedSlice(rank_offset=0, size=8, store_host="localhost", store_port=29600)


class _Store:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


class _Backend:
    def __init__(self) -> None:
        self.sends: list[tuple[torch.Tensor, int, int]] = []

    def send(self, tensors, dst, tag):
        [tensor] = tensors
        self.sends.append((tensor, dst, tag))
        return self

    def wait(self) -> None:
        pass


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def init_weights_update_group(self, **kwargs) -> dict:
        self.calls.append(("init_weights_update_group", kwargs))
        return {"success": True}

    async def update_weights_from_distributed(self, **kwargs) -> dict:
        self.calls.append(("update_weights_from_distributed", kwargs))
        return {"success": True}


def _unpack(packed: torch.Tensor, dtypes: list[torch.dtype], shapes: list[torch.Size]) -> list[torch.Tensor]:
    """What the engine does with a received buffer."""
    numels = [shape.numel() for shape in shapes]
    regions = tpu_shared_slice.packed_regions(numels, dtypes)
    return [
        packed[offset // 2 : (offset + numel * dtype.itemsize) // 2].view(dtype).view(shape)
        for (offset, _), dtype, shape, numel in zip(regions, dtypes, shapes, numels, strict=True)
    ]


@pytest.fixture
def membership():
    membership = tpu_shared_slice._Membership(backend=_Backend(), store=_Store(), slice_rank=1)
    with patch.object(tpu_shared_slice, "_membership", membership):
        yield membership


def _connected_protocol(engine: _Engine, *, rank: int) -> UpdateWeightFromIci:
    protocol = UpdateWeightFromIci(Namespace(update_weight_engine_request_timeout=10, ft_components=()))
    with (
        patch.object(tpu_shared_slice, "spec", return_value=_SHARED),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.get_rank", return_value=rank),
    ):
        protocol.connect([engine], [4], [4], ["cell-0"], None, None, "all")
    return protocol


def test_regions_are_whole_aligned_blocks():
    regions = tpu_shared_slice.packed_regions(
        [1, 2048, 2049, 0], [torch.float32, torch.bfloat16, torch.bfloat16, torch.float32]
    )
    # 4 B -> one block, 4096 B -> one block, 4098 B -> two blocks, empty -> still one block
    assert regions == [(0, 4096), (4096, 4096), (8192, 8192), (16384, 4096)]


def test_a_packed_bucket_views_back_bitwise():
    tensors = [torch.randn(3, 5, dtype=torch.bfloat16), torch.randn(7, dtype=torch.float32), torch.randn(2, 4096)]
    packed = tpu_shared_slice.pack(tensors)
    assert packed.dtype == torch.int16
    for got, want in zip(
        _unpack(packed, [t.dtype for t in tensors], [t.shape for t in tensors]), tensors, strict=True
    ):
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_each_send_posts_its_own_names_under_a_fresh_tag(membership):
    """Ranks hold different experts, so each posts its own names, keyed by the tag of its send."""
    tpu_shared_slice.send(torch.zeros(2048, dtype=torch.int16), 5, names=["a", "b"])
    tpu_shared_slice.send(torch.zeros(2048, dtype=torch.int16), 5, names=["c"])
    assert [(dst, tag) for _, dst, tag in membership.backend.sends] == [(5, 1), (5, 2)]
    assert membership.store.values == {
        tpu_shared_slice.bucket_names_key(1, 5, 1): json.dumps(["a", "b"]),
        tpu_shared_slice.bucket_names_key(1, 5, 2): json.dumps(["c"]),
    }


def test_trainer_rank_i_sends_to_engine_tp_rank_i(membership):
    engine = _Engine()
    protocol = _connected_protocol(engine, rank=1)
    assert [name for name, _ in engine.calls] == []  # only rank 0 talks to the engine
    bucket = [("model.layers.0.mlp.experts.4.down_proj.weight", torch.randn(3, 5, dtype=torch.bfloat16))]
    with patch("torch.distributed.get_rank", return_value=1):
        protocol.send_bucket(bucket)
    [(packed, dst, _)] = membership.backend.sends
    assert dst == 5  # the engine's TP ranks follow the trainer's 4 in the slice
    [got] = _unpack(packed, [torch.bfloat16], [torch.Size([3, 5])])
    assert torch.equal(got, bucket[0][1])


def test_rank_0_opens_the_group_and_posts_each_bucket(membership):
    engine = _Engine()
    protocol = _connected_protocol(engine, rank=0)
    with patch("torch.distributed.get_rank", return_value=0):
        protocol.send_bucket([("lm_head.weight", torch.zeros(2, 3, dtype=torch.bfloat16))])
    [(init, init_kwargs), (update, update_kwargs)] = engine.calls
    assert (init, init_kwargs["backend"], init_kwargs["world_size"]) == (
        "init_weights_update_group",
        tpu_shared_slice.ENGINE_BACKEND_NAME,
        8,
    )
    assert (update, update_kwargs["names"], update_kwargs["group_name"]) == (
        "update_weights_from_distributed",
        ["lm_head.weight"],
        "miles-ici",
    )


def test_an_engine_that_does_not_fill_the_slice_is_rejected():
    protocol = UpdateWeightFromIci(Namespace(update_weight_engine_request_timeout=10, ft_components=()))
    with (
        patch.object(tpu_shared_slice, "spec", return_value=_SHARED),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.get_rank", return_value=0),
        pytest.raises(ValueError, match="pairs trainer rank i with TP rank i"),
    ):
        protocol.connect([_Engine(), _Engine()], [2, 2], [4, 6], ["cell-0", "cell-1"], None, None, "all")
