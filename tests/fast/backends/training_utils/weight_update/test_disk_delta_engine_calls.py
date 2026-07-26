from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RecordingApiClient:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            return {"success": True}

        return method


def _make_protocol(calls: list[tuple[str, dict]]) -> UpdateWeightFromDiskDelta:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace(
        update_weight_local_checkpoint_dir="/local/ckpt",
        update_weight_disk_dir="/shared/delta",
        pause_generation_mode="retract",
        check_weight_update_equal=False,
    )
    protocol.rollout_engines = [_RecordingApiClient(calls)]
    protocol._post_write_hook = None
    protocol._version_dir = "/shared/delta/v7"
    return protocol


def _patch_dist(rank: int):
    dist_mock = patch(f"{_MODULE}.dist")
    gloo_mock = patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock())
    return dist_mock, gloo_mock, rank


def test_reload_engines_pulls_with_both_checkpoint_dirs_then_reloads():
    """The reload pull carries both checkpoint dirs the deleted engine wrapper used to inject."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(7)

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
    protocol = _make_protocol(calls)
    protocol.args.pause_generation_mode = "in_place"

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(7)

    assert "flush_cache" not in [name for name, _kwargs in calls]


def test_non_source_rank_issues_no_requests():
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 1
        protocol._reload_engines(7)

    assert calls == []


def _capture_baseline(protocol: UpdateWeightFromDiskDelta, tmp_path) -> None:
    protocol.delta_dir = str(tmp_path / "delta")
    protocol.args.hf_checkpoint = "/fake/hf"
    protocol._snapshot = {}
    protocol.is_sender = False

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name, **kwargs: None),
    ):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        protocol._capture_baseline(lambda materialize: [])


def test_baseline_capture_pulls_with_both_checkpoint_dirs(tmp_path):
    """The baseline pull carries both checkpoint dirs too."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    _capture_baseline(protocol, tmp_path)

    assert [name for name, _kwargs in calls] == ["pull_weights", "get_weight_version"]
    assert calls[0][1] == {
        "target_version": 0,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }


def test_baseline_capture_reloads_the_pulled_checkpoint_when_equality_is_checked(tmp_path):
    """check_weight_update_equal makes the baseline reload the base checkpoint it just pulled."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)
    protocol.args.check_weight_update_equal = True

    _capture_baseline(protocol, tmp_path)

    assert [name for name, _kwargs in calls] == ["pull_weights", "update_weights_from_disk"]
    assert calls[1][1] == {"model_path": "/local/ckpt", "weight_version": "0"}
