from types import SimpleNamespace

import pytest
import torch.distributed as dist
from tests.fast.dist_utils import find_free_port, init_gloo

from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind


@pytest.fixture
def legacy_source(monkeypatch):
    init_gloo(0, 1, port=find_free_port())
    monkeypatch.setattr(
        checkpoint, "classify_adapter_source", lambda *_, **__: AdapterSourceKind.LEGACY_TINKER_CHECKPOINT
    )
    monkeypatch.setattr(checkpoint, "_read_legacy_meta", lambda *_: {})
    metadata = SimpleNamespace(state_dict_metadata={"layer.adapter.w": None})
    monkeypatch.setattr(checkpoint.dcp, "FileSystemReader", lambda _: SimpleNamespace(read_metadata=lambda: metadata))
    monkeypatch.setattr(
        checkpoint, "_slot_weights_sharded_state_dict", lambda *_: {"w": SimpleNamespace(key="layer.adapters.0.w")}
    )
    yield
    dist.destroy_process_group()


def _with_dp_size(monkeypatch, size):
    groups = SimpleNamespace(dp_cp=SimpleNamespace(size=lambda: size), expt_dp=None)
    monkeypatch.setattr(checkpoint.ProcessGroupCollection, "use_mpu_process_groups", lambda **_: groups)


@pytest.mark.parametrize("dp_size, load_optimizer", [(1, True), (2, False)])
def test_a_legacy_checkpoint_loads_where_its_optimizer_ownership_is_unambiguous(
    legacy_source, monkeypatch, dp_size, load_optimizer
):
    _with_dp_size(monkeypatch, dp_size)
    assert checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer, run_shapes={}) is None


def test_a_legacy_checkpoint_with_optimizer_state_is_refused_under_data_parallelism(legacy_source, monkeypatch):
    _with_dp_size(monkeypatch, 2)
    mismatch = checkpoint.find_load_mismatch(None, 0, "/ckpt", None, None, load_optimizer=True, run_shapes={})
    assert "optimizer=False" in mismatch
