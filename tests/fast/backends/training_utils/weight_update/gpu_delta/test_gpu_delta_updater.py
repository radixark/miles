from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.protocols.gpu_delta.protocol import UpdateWeightFromGpuDelta


def _updater(monkeypatch, requires_export):
    updater = object.__new__(updater_module.WeightUpdater)
    protocol = Mock(spec=UpdateWeightFromGpuDelta)
    protocol.begin_sync.return_value = True
    protocol.requires_export = requires_export
    protocol.needs_base_resync_for_lora = False
    protocol.use_weight_update_session = False
    protocol.is_sender = True
    protocol.group_name = "test"
    protocol.finalize.return_value = None
    protocol.pop_metrics.return_value = {"perf/update_weights_gpu_delta_s": 1.0}
    updater.protocol = protocol
    updater.weight_version = 3
    updater.is_lora = False
    updater.args = SimpleNamespace(check_lora_weight_equal=False)
    updater.weights_getter = Mock(return_value={"w": object()})
    updater._hf_weight_iterator = Mock()
    updater._hf_weight_iterator.iter_hf_weights.return_value = [[("w", object())]]
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(updater_module.dist, "barrier", Mock())
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(updater_module, "tqdm", Mock())
    return updater, protocol


@pytest.mark.parametrize("requires_export", [False, True])
def test_checkpoint_target_reuse_skips_all_weight_export(monkeypatch, requires_export):
    updater, protocol = _updater(monkeypatch, requires_export)

    assert updater.update_weights(9) == {"perf/update_weights_gpu_delta_s": 1.0}

    assert updater.weights_getter.call_count == int(requires_export)
    assert updater._hf_weight_iterator.iter_hf_weights.call_count == int(requires_export)
    assert protocol.send_bucket.call_count == int(requires_export)
    protocol.after_base_weights.assert_called_once_with()
    protocol.begin_sync.assert_called_once_with(4, updater._iter_base_buckets, 9)
    protocol.finalize.assert_called_once_with(4)
    protocol.after_engines_resumed.assert_called_once_with()
    assert updater.weight_version == 4
