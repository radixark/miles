from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.utils import MultiprocessingSerializer

from miles.backends.training_utils.weight_update.protocols import cuda_ipc, lora_reload
from miles.backends.training_utils.weight_update.protocols.cuda_ipc import FlattenedTensorBucket
from miles.backends.training_utils.weight_update.protocols.lora_reload import UpdateLoraFromTensorsReload
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_sync_sends_base


class _Engine:
    """Records the LoRA calls one engine receives."""

    def __init__(self):
        self.calls = []

    async def unload_lora_adapter(self, *, lora_name):
        self.calls.append(("unload", lora_name))
        return {"success": True}

    async def load_lora_adapter_from_tensors(self, **kwargs):
        self.calls.append(("load", kwargs))
        return {"success": True}


@pytest.fixture
def reload_protocol(monkeypatch):
    for module in (cuda_ipc, lora_reload):
        monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 1)
        monkeypatch.setattr(module.dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(module.dist, "new_group", lambda ranks, backend: object())
    monkeypatch.setattr(lora_reload.dist, "barrier", lambda group=None: None)

    def gather_object(obj, object_gather_list, dst, group):
        object_gather_list[0] = obj

    monkeypatch.setattr(lora_reload.dist, "gather_object", gather_object)
    versions = []
    monkeypatch.setattr(lora_reload, "maybe_pause_engines", lambda args, updaters: None)
    monkeypatch.setattr(lora_reload, "maybe_resume_engines", lambda args, updaters: None)
    monkeypatch.setattr(lora_reload, "set_weight_version", lambda updaters, version: versions.append(version))

    args = Namespace(
        rollout_num_gpus_per_engine=1,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_adapter_targets=["qkvr"],
        lora_adapter_path=None,
        lora_base_cpu_backup=True,
        lora_train_only=False,
        colocate=True,
        offload_rollout=True,
        check_weight_update_equal=False,
        check_lora_weight_equal=True,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
    )
    args.lora_sync_sends_base = lora_sync_sends_base(args)
    protocol = UpdateLoraFromTensorsReload(args)
    engine = _Engine()
    protocol.connect(
        rollout_engines=[engine],
        engine_gpu_counts=[1],
        engine_gpu_offsets=[0],
        engine_cell_ids=["engine"],
        parallel_state=SimpleNamespace(),
        placement=protocol.required_placement,
        selector="all",
    )
    return protocol, engine, versions


def _adapter(seed):
    gen = torch.Generator().manual_seed(seed)
    return [
        ("base_model.model.layers.0.attn.wq_du.lora_A.weight", torch.randn(16, 32, generator=gen)),
        ("base_model.model.layers.0.attn.wq_du.lora_B.weight", torch.randn(64, 16, generator=gen)),
    ]


def _sync(protocol, adapter, version):
    protocol.send_bucket([(f"{LORA_ADAPTER_NAME}:{name}", tensor) for name, tensor in adapter])
    protocol.finalize(version)


def test_first_sync_loads_and_later_syncs_replace_the_adapter(reload_protocol):
    protocol, engine, versions = reload_protocol
    first, second = _adapter(0), _adapter(1)
    _sync(protocol, first, 1)
    _sync(protocol, second, 2)

    assert [call[0] for call in engine.calls] == ["load", "unload", "load"]
    assert engine.calls[1] == ("unload", LORA_ADAPTER_NAME)
    assert versions == [1, 2]

    load = engine.calls[2][1]
    assert load["lora_name"] == LORA_ADAPTER_NAME and load["load_format"] == "flattened_bucket"
    assert load["config_dict"]["r"] == 16 and load["config_dict"]["lora_alpha"] == 32
    # one serialized bucket per engine rank, holding the second adapter exactly
    (blob,) = load["serialized_named_tensors"]
    data = MultiprocessingSerializer.deserialize(blob)
    bucket = FlattenedTensorBucket(flattened_tensor=data["flattened_tensor"], metadata=data["metadata"])
    received = dict(bucket.reconstruct_tensors())
    assert received.keys() == dict(second).keys()
    assert all(torch.equal(received[name], tensor) for name, tensor in second)
    assert load["expected_checksums"].keys() == received.keys()


def test_a_base_tensor_is_rejected(reload_protocol):
    protocol, _, _ = reload_protocol
    with pytest.raises(AssertionError, match="non-adapter"):
        protocol.send_bucket([("model.llm.layers.0.attn.wo_ud.weight", torch.zeros(2, 2))])


def test_a_failed_load_raises(reload_protocol):
    protocol, engine, _ = reload_protocol

    async def fail(**kwargs):
        return {"success": False, "error_message": "rank too large"}

    engine.load_lora_adapter_from_tensors = fail
    with pytest.raises(RuntimeError, match="rank too large"):
        _sync(protocol, _adapter(0), 1)
