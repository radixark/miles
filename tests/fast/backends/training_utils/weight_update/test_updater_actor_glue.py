"""The two actor-facing helpers every backend's update_weights is built from."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"


def _make_updater() -> WeightUpdater:
    protocol = SimpleNamespace(
        required_placement=MagicMock(),
        supports_lora=False,
        is_sender=True,
        connect=MagicMock(),
    )
    iterator = MagicMock()
    iterator.weight_update_selector = "all"
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
        return WeightUpdater(
            Namespace(),
            [MagicMock()],
            weights_getter=lambda: None,
            model_name="qwen",
            quantization_config=None,
            iterator_factory=lambda *a, **k: iterator,
            parallel_state=MagicMock(),
            is_lora=False,
        )


def _info(engines, hashes):
    return SimpleNamespace(
        rollout_engines=engines,
        engine_gpu_counts=[1] * len(engines),
        engine_gpu_offsets=list(range(len(engines))),
        snapshot_cell_id_to_hashes=hashes,
    )


class _Engine:
    def __init__(self, version):
        self.version = version

    async def get_weight_version(self):
        return self.version


def test_reconnect_happens_once_per_cell_snapshot():
    updater = _make_updater()
    engines = [object(), object()]
    with patch(f"{_UPDATER_MODULE}.dist"), patch(f"{_UPDATER_MODULE}.get_gloo_group"):
        first = updater.reconnect_if_needed(_info(engines, {"cell-0": "a"}))
        second = updater.reconnect_if_needed(_info(engines, {"cell-0": "a"}))
        third = updater.reconnect_if_needed(_info(engines, {"cell-0": "b"}))
    assert (first, second, third) == (True, False, True)
    assert updater.protocol.connect.call_count == 2
    assert updater.protocol.connect.call_args_list[1].args[1:3] == ([1, 1], [0, 1])


def test_engine_version_mismatch_is_an_error():
    updater = _make_updater()
    updater.weight_version = 3
    with pytest.raises(RuntimeError, match="Weight version mismatch"):
        updater.verify_engine_version([_Engine(7)])
    updater.verify_engine_version([_Engine(3)])


def test_engine_version_is_not_checked_before_the_first_published_sync():
    """A protocol may decline round one to capture its baseline; the engine then has no version to match."""
    updater = _make_updater()
    updater.verify_engine_version([_Engine(7)])
    updater.verify_engine_version([])
