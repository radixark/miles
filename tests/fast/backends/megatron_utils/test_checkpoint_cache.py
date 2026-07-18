import json
from pathlib import Path

import pytest

from miles.backends.megatron_utils import checkpoint_cache
from miles.backends.megatron_utils.checkpoint_cache import prepare_rank_local_checkpoint


def _make_checkpoint(root: Path) -> None:
    release = root / "release"
    release.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("release")
    (release / ".metadata").write_bytes(b"metadata")
    (release / "common.pt").write_bytes(b"common")
    (release / "__0_0.distcp").write_bytes(b"rank-zero")
    (release / "__1_0.distcp").write_bytes(b"rank-one")


def test_prepare_rank_local_checkpoint_copies_only_requested_rank(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(checkpoint_cache, "_MIN_FREE_BYTES", 0)
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    _make_checkpoint(source)

    target = Path(prepare_rank_local_checkpoint(str(source), str(cache), rank=1))

    assert (target / "release" / "__1_0.distcp").read_bytes() == b"rank-one"
    assert not (target / "release" / "__0_0.distcp").exists()
    assert (target / "release" / ".metadata").read_bytes() == b"metadata"
    assert json.loads((target / ".miles_cache_manifest.json").read_text())["files"]

    sentinel_mtime = (target / ".miles_cache_manifest.json").stat().st_mtime_ns
    assert prepare_rank_local_checkpoint(str(source), str(cache), rank=1) == str(target)
    assert (target / ".miles_cache_manifest.json").stat().st_mtime_ns == sentinel_mtime


def test_prepare_rank_local_checkpoint_rejects_mismatched_world_size(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(checkpoint_cache, "_MIN_FREE_BYTES", 0)
    source = tmp_path / "source"
    _make_checkpoint(source)

    with pytest.raises(FileNotFoundError, match="same world size and parallel topology"):
        prepare_rank_local_checkpoint(str(source), str(tmp_path / "cache"), rank=2)
