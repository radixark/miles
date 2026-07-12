import importlib
import sys
import types
from argparse import Namespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def disk_delta_updater(monkeypatch):
    common_name = "miles.backends.megatron_utils.update_weight.common"
    mixin_name = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin"
    delta_name = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta"

    common = types.ModuleType(common_name)
    common._check_weight_sync_results = MagicMock()

    mixin = types.ModuleType(mixin_name)

    class DistBucketedWeightUpdateMixin:
        def _init_lora(self, *, is_lora, **_kwargs):
            self.is_lora = is_lora

    mixin.DistBucketedWeightUpdateMixin = DistBucketedWeightUpdateMixin

    monkeypatch.setitem(sys.modules, common_name, common)
    monkeypatch.setitem(sys.modules, mixin_name, mixin)
    monkeypatch.delitem(sys.modules, delta_name, raising=False)
    return importlib.import_module(delta_name).UpdateWeightFromDiskDelta


def test_rollout_engine_connection_freshness(tmp_path, disk_delta_updater):
    updater = disk_delta_updater(
        args=Namespace(
            update_weight_disk_dir=str(tmp_path),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="sha256",
            custom_update_weight_post_write_path=None,
        ),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
    )

    assert not updater.is_rollout_engines_fresh()

    rollout_engines = [object()]
    updater.connect_rollout_engines(rollout_engines, object())
    assert updater.rollout_engines is rollout_engines
    assert updater.is_rollout_engines_fresh()

    updater.mark_engine_connection_stale()
    assert not updater.is_rollout_engines_fresh()

    updater.connect_rollout_engines(rollout_engines, object())
    assert updater.is_rollout_engines_fresh()
