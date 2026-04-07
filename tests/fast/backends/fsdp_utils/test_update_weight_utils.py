import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _install_sglang_stubs():
    serializer = type(
        "MultiprocessingSerializer",
        (),
        {"serialize": staticmethod(lambda data, output_str=True: "serialized")},
    )
    bucket = type(
        "FlattenedTensorBucket",
        (),
        {
            "__init__": lambda self, named_tensors: setattr(self, "named_tensors", named_tensors),
            "get_metadata": lambda self: {"names": [name for name, _ in self.named_tensors]},
            "get_flattened_tensor": lambda self: self.named_tensors[0][1],
        },
    )

    stub_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
        "sglang.srt.utils.patch_torch": types.ModuleType("sglang.srt.utils.patch_torch"),
        "sglang.srt.weight_sync": types.ModuleType("sglang.srt.weight_sync"),
        "sglang.srt.weight_sync.tensor_bucket": types.ModuleType("sglang.srt.weight_sync.tensor_bucket"),
    }
    stub_modules["sglang.srt.utils"].MultiprocessingSerializer = serializer
    stub_modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    stub_modules["sglang.srt.weight_sync.tensor_bucket"].FlattenedTensorBucket = bucket

    for name, module in stub_modules.items():
        sys.modules.setdefault(name, module)


_install_sglang_stubs()

_MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "miles" / "backends" / "fsdp_utils" / "update_weight_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("miles_fsdp_update_weight_utils_test", _MODULE_PATH)
uw = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(uw)


class _RemoteCall:
    def __init__(self, fn):
        self.remote = fn


class _FakeEngine:
    def __init__(self, event_log, update_result=None):
        self.pause_generation = _RemoteCall(lambda: event_log.append("pause"))
        self.flush_cache = _RemoteCall(lambda: event_log.append("flush"))
        self.continue_generation = _RemoteCall(lambda: event_log.append("continue"))
        self.update_weights_from_tensor = _RemoteCall(lambda **_: update_result)


class _DummyUpdater(uw.UpdateWeight):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.updated_buckets = []

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock, engine_gpu_counts=None, engine_gpu_offsets=None):
        self.rollout_engines = rollout_engines

    def update_bucket_weights(self, named_tensors, weight_version=None):
        self.updated_buckets.append((weight_version, [name for name, _ in named_tensors]))


@pytest.fixture
def _common_monkeypatch(monkeypatch):
    monkeypatch.setattr(uw.ray, "get", lambda x: x)
    monkeypatch.setattr(uw.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(uw.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(uw, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)


def test_update_weights_pauses_and_resumes_engines(_common_monkeypatch):
    args = SimpleNamespace(update_weight_buffer_size=1 << 30)
    model = torch.nn.Linear(2, 2, bias=False)
    events = []
    updater = _DummyUpdater(args, model)
    updater.connect_rollout_engines([_FakeEngine(events)], None)

    updater.update_weights()

    assert events == ["pause", "flush", "continue"]
    assert updater.weight_version == 1
    assert updater.updated_buckets


def test_tensor_update_surfaces_dict_failure(_common_monkeypatch, monkeypatch):
    monkeypatch.setattr(uw, "monkey_patch_torch_reductions", lambda: None)

    class _Bucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_metadata(self):
            return {"names": [name for name, _ in self.named_tensors]}

        def get_flattened_tensor(self):
            return self.named_tensors[0][1]

    monkeypatch.setattr(uw, "FlattenedTensorBucket", _Bucket)
    monkeypatch.setattr(uw.MultiprocessingSerializer, "serialize", lambda data, output_str=True: "serialized")
    monkeypatch.setattr(uw.dist, "get_world_size", lambda group: 1)

    def _gather_object(obj, object_gather_list, dst, group):
        object_gather_list[0] = obj

    monkeypatch.setattr(uw.dist, "gather_object", _gather_object)

    args = SimpleNamespace(update_weight_buffer_size=1 << 30)
    updater = uw.UpdateWeightFromTensor(args, torch.nn.Linear(2, 2, bias=False))
    updater.rollout_engines = []
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = "group"
    updater._ipc_engine = _FakeEngine([], update_result={"success": False, "message": "boom"})

    with pytest.raises(RuntimeError, match="boom"):
        updater.update_bucket_weights([("weight", torch.ones(2, 2))], weight_version=7)
