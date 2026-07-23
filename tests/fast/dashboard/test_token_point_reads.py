"""Token-view point reads: parquet mirror + mmap'd train shards."""

import polars as pl
import pytest
import torch
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader


@pytest.fixture
def dumped(tmp_path):
    dump_dummy_run(tmp_path, steps=1)
    return tmp_path


def test_save_writes_the_parquet_mirror(dumped):
    frame = pl.read_parquet(dumped / "dashboard_columns" / "rollout_0.parquet")
    reader = DumpReader(dumped)
    joined = reader.joined(0)
    assert sorted(frame["sample_index"]) == sorted(s.index for s in joined.samples)
    sample = joined.samples[0]
    row = frame.filter(pl.col("sample_index") == sample.index).row(0, named=True)
    assert row["tokens"] == list(sample.tokens)
    assert row["response_length"] == sample.response_length


def test_tokens_does_not_touch_the_rollout_pt(dumped, monkeypatch):
    reader = DumpReader(dumped)
    real_load = torch.load

    def guarded(path, *args, **kwargs):
        assert "rollout_data" not in str(path), f"point read full-loaded {path}"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    payload = reader.tokens(0, 0, start=0, end=8)
    assert payload["total_len"] > 0 and len(payload["token_ids"]) == 8


def test_missing_mirror_is_rebuilt(dumped):
    mirror = dumped / "dashboard_columns" / "rollout_0.parquet"
    mirror.unlink()
    reader = DumpReader(dumped)
    payload = reader.tokens(0, 0, start=0, end=4)
    assert mirror.exists() and payload["token_ids"]


def test_stale_schema_is_rebuilt(dumped):
    mirror = dumped / "dashboard_columns" / "rollout_0.parquet"
    pl.DataFrame({"sample_index": [0]}).write_parquet(mirror)
    payload = DumpReader(dumped).tokens(0, 0, start=0, end=4)
    assert payload["token_ids"]
    assert "tokens" in pl.read_parquet_schema(mirror)


def test_unknown_sample_raises(dumped):
    with pytest.raises(KeyError):
        DumpReader(dumped).tokens(0, 10_000)


def test_train_row_values_match_joined(dumped):
    reader = DumpReader(dumped)
    joined = reader.joined(0)
    sample_index = joined.samples[0].index
    lazy = reader._train_row_lazy(0, sample_index)
    full = joined.train_rows[sample_index]
    for field in ("log_probs", "ref_log_probs", "advantages"):
        lazy_v, full_v = getattr(lazy, field), getattr(full, field)
        assert (lazy_v is None) == (full_v is None)
        if lazy_v is not None:
            assert torch.equal(lazy_v, full_v)
