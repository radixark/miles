import sys
import types
from types import SimpleNamespace


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = True

    def __init__(self, *, named_tensors=None, flattened_tensor=None, metadata=None):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self._flattened_tensor = ("flattened", tuple(name for name, _ in named_tensors))
            self._metadata = tuple(name for name, _ in named_tensors)
            return

        self._flattened_tensor = flattened_tensor
        self._metadata = metadata

    def get_flattened_tensor(self):
        return self._flattened_tensor

    def get_metadata(self):
        return self._metadata


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(value, output_str):
        assert output_str is True
        return value


_fake_sglang = types.ModuleType("miles.backends.megatron_utils.sglang")
_fake_sglang.FlattenedTensorBucket = _FakeFlattenedTensorBucket
_fake_sglang.MultiprocessingSerializer = _FakeMultiprocessingSerializer
sys.modules.setdefault("miles.backends.megatron_utils.sglang", _fake_sglang)

_fake_common = types.ModuleType("miles.backends.megatron_utils.update_weight.common")
_fake_common._check_weight_sync_results = lambda *args, **kwargs: None
_fake_common.begin_weight_update = lambda *args, **kwargs: None
_fake_common.end_weight_update = lambda *args, **kwargs: None
sys.modules.setdefault("miles.backends.megatron_utils.update_weight.common", _fake_common)


class _FakeHfWeightIteratorBase:
    @staticmethod
    def create(*args, **kwargs):
        return None


_fake_hf_weight_iterator_base = types.ModuleType(
    "miles.backends.megatron_utils.update_weight.hf_weight_iterator_base"
)
_fake_hf_weight_iterator_base.HfWeightIteratorBase = _FakeHfWeightIteratorBase
sys.modules.setdefault(
    "miles.backends.megatron_utils.update_weight.hf_weight_iterator_base",
    _fake_hf_weight_iterator_base,
)

_fake_broadcast = types.ModuleType(
    "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
)
_fake_broadcast.connect_rollout_engines_from_distributed = lambda *args, **kwargs: None
_fake_broadcast.disconnect_rollout_engines_from_distributed = lambda *args, **kwargs: None
_fake_broadcast.update_weights_from_distributed = lambda *args, **kwargs: []
sys.modules.setdefault(
    "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast",
    _fake_broadcast,
)

from miles.backends.megatron_utils.update_weight import update_weight_from_tensor as update_weight  # noqa: E402


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return f"ref-{len(self.calls)}"


class _FakeEngine:
    def __init__(self):
        self.update_weights_from_tensor = _FakeRemoteMethod()


def _install_fakes(monkeypatch, gathered):
    state = SimpleNamespace(local_object=None)

    def gather_object(obj, object_gather_list, dst, group):
        state.local_object = obj
        if object_gather_list is not None:
            object_gather_list[:] = gathered

    fake_dist = SimpleNamespace(
        get_rank=lambda: 0,
        get_world_size=lambda group=None: len(gathered),
        gather_object=gather_object,
    )
    monkeypatch.setattr(update_weight, "dist", fake_dist)
    monkeypatch.setattr(update_weight, "FlattenedTensorBucket", _FakeFlattenedTensorBucket)
    monkeypatch.setattr(update_weight, "MultiprocessingSerializer", _FakeMultiprocessingSerializer)
    monkeypatch.setattr(update_weight.torch.cuda, "current_device", lambda: "cuda:0")
    monkeypatch.setattr(
        update_weight.torch,
        "empty",
        lambda size, dtype, device: {"size": size, "dtype": dtype, "device": device},
    )

    return state


def test_empty_colocated_bucket_still_participates_in_gather(monkeypatch):
    state = _install_fakes(monkeypatch, gathered=[[], []])
    engine = _FakeEngine()

    refs, long_lived_tensors = update_weight._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=3,
    )

    assert state.local_object == []
    assert refs == []
    assert long_lived_tensors == []
    assert engine.update_weights_from_tensor.calls == []


def test_source_rank_pads_empty_colocated_bucket_entries(monkeypatch):
    remote_serialized_bucket = {"flattened_tensor": ("remote",), "metadata": ("remote_weight",)}
    state = _install_fakes(monkeypatch, gathered=[[], [remote_serialized_bucket]])
    engine = _FakeEngine()

    refs, long_lived_tensors = update_weight._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert state.local_object == []
    assert refs == ["ref-1"]
    assert len(long_lived_tensors) == 1
    empty_bucket = long_lived_tensors[0]
    assert empty_bucket["metadata"] == []
    assert empty_bucket["flattened_tensor"] == {"size": 0, "dtype": update_weight.torch.uint8, "device": "cuda:0"}

    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": [empty_bucket, remote_serialized_bucket],
            "load_format": "flattened_bucket",
            "weight_version": "7",
        }
    ]
