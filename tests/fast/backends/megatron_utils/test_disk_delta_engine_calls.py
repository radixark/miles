from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta"


class _RecordingApiClient:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            return {"success": True}

        return method


def _make_updater(calls: list[tuple[str, dict]]) -> UpdateWeightFromDiskDelta:
    updater = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    updater.args = Namespace(
        update_weight_local_checkpoint_dir="/local/ckpt",
        update_weight_disk_dir="/shared/delta",
        pause_generation_mode="retract",
        check_weight_update_equal=False,
    )
    updater.rollout_engines = [_RecordingApiClient(calls)]
    updater.weight_version = 7
    updater._post_write_hook = None
    updater._version_dir = "/shared/delta/v7"
    return updater


def test_reload_engines_pulls_with_both_checkpoint_dirs_then_reloads():
    """The reload pull carries both checkpoint dirs the deleted engine wrapper used to inject."""
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater._reload_engines()

    assert [name for name, _kwargs in calls] == [
        "pull_weights",
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert calls[1][1] == {"mode": "retract"}
    assert calls[0][1] == {
        "target_version": 7,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }
    assert calls[3][1] == {"model_path": "/local/ckpt", "weight_version": "7"}


def test_in_place_pause_mode_skips_the_flush():
    """in_place pause mode does not flush."""
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)
    updater.args.pause_generation_mode = "in_place"

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater._reload_engines()

    assert "flush_cache" not in [name for name, _kwargs in calls]


def test_non_source_rank_issues_no_requests():
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 1
        updater._reload_engines()

    assert calls == []


def test_baseline_capture_pulls_with_both_checkpoint_dirs(tmp_path):
    """The baseline pull carries both checkpoint dirs too."""
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)
    updater.delta_dir = str(tmp_path / "delta")
    updater.args.hf_checkpoint = "/fake/hf"
    updater._snapshot = {}
    updater._for_each_hf_bucket = lambda bucket_func: None

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name: None),
    ):
        dist_mock.get_rank.return_value = 0
        updater._capture_baseline()

    assert [name for name, _kwargs in calls] == ["pull_weights", "get_weight_version"]
    assert calls[0][1] == {
        "target_version": 0,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }


def test_baseline_capture_reloads_the_pulled_checkpoint_when_equality_is_checked(tmp_path):
    """check_weight_update_equal makes the baseline reload the base checkpoint it just pulled."""
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)
    updater.delta_dir = str(tmp_path / "delta")
    updater.args.hf_checkpoint = "/fake/hf"
    updater.args.check_weight_update_equal = True
    updater._snapshot = {}
    updater._for_each_hf_bucket = lambda bucket_func: None

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name: None),
    ):
        dist_mock.get_rank.return_value = 0
        updater._capture_baseline()

    assert [name for name, _kwargs in calls] == ["pull_weights", "update_weights_from_disk"]
    assert calls[1][1] == {"model_path": "/local/ckpt", "weight_version": "7"}


def test_non_source_rank_waits_for_baseline_engine_reload(tmp_path):
    """A non-source rank waits until rank zero finishes the baseline engine reload."""
    updater = _make_updater([])
    updater.delta_dir = str(tmp_path / "delta")
    updater.args.hf_checkpoint = "/fake/hf"
    updater._snapshot = {}
    updater._for_each_hf_bucket = lambda bucket_func: None

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name: None),
    ):
        dist_mock.get_rank.return_value = 1
        updater._capture_baseline()

    assert dist_mock.barrier.call_count == 2
