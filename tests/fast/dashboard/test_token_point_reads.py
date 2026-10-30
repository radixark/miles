"""Token-view point reads: parquet mirror + mmap'd train shards."""

import threading
import time

import polars as pl
import pytest
import torch
from tests.fast.dashboard.dummy_dump import blank_samples, dump_dummy_run

from miles.dashboard.dump_reader import DumpReader


@pytest.fixture
def dumped(tmp_path):
    dump_dummy_run(tmp_path, steps=1)
    return tmp_path


def _mirror(reader: DumpReader):
    return reader.cache_dir / "columns" / "rollout_0.parquet"


def test_the_first_point_read_writes_the_parquet_mirror(dumped):
    reader = DumpReader(dumped)
    reader.tokens(0, 0, start=0, end=1)
    frame = pl.read_parquet(_mirror(reader))
    joined = reader.joined(0)
    assert sorted(frame["sample_index"]) == sorted(s.index for s in joined.samples)
    sample = joined.samples[0]
    row = frame.filter(pl.col("sample_index") == sample.index).row(0, named=True)
    assert row["tokens"] == list(sample.tokens)
    assert row["response_length"] == sample.response_length


def test_tokens_does_not_touch_the_rollout_pt(dumped, monkeypatch):
    reader = DumpReader(dumped)
    reader.tokens(0, 0, start=0, end=1)  # the first read builds the mirror from the .pt
    real_load = torch.load

    def guarded(path, *args, **kwargs):
        assert "rollout_data" not in str(path), f"point read full-loaded {path}"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    payload = reader.tokens(0, 0, start=0, end=8)
    assert payload["total_len"] > 0 and len(payload["token_ids"]) == 8


def test_missing_mirror_is_rebuilt(dumped):
    reader = DumpReader(dumped)
    payload = reader.tokens(0, 0, start=0, end=4)
    assert _mirror(reader).exists() and payload["token_ids"]


def test_stale_schema_is_rebuilt(dumped):
    reader = DumpReader(dumped)
    mirror = _mirror(reader)
    mirror.parent.mkdir(parents=True)
    pl.DataFrame({"sample_index": [0]}).write_parquet(mirror)
    payload = reader.tokens(0, 0, start=0, end=4)
    assert payload["token_ids"]
    assert "tokens" in pl.read_parquet_schema(mirror)


def test_the_mirror_goes_to_the_cache_dir_so_a_read_only_dump_still_serves(dumped, tmp_path_factory):
    """A dump the viewer cannot write to must still serve token views when ``--cache-dir`` points elsewhere."""
    before = sorted(path for path in dumped.rglob("*"))
    reader = DumpReader(dumped, cache_dir=tmp_path_factory.mktemp("cache"))

    assert reader.tokens(0, 0, start=0, end=4)["token_ids"]
    assert _mirror(reader).exists()
    assert sorted(path for path in dumped.rglob("*")) == before


def test_a_rewritten_rollout_rebuilds_its_mirror(dumped):
    """A rerun that rewrites a step's ``.pt`` must not leave the token view serving the old samples."""
    DumpReader(dumped).tokens(0, 0, start=0, end=1)
    blank_samples(dumped, 0)

    with pytest.raises(KeyError, match="unknown sample_index"):
        DumpReader(dumped).tokens(0, 0, start=0, end=1)


def test_building_one_steps_mirror_does_not_hold_up_another_step(tmp_path, monkeypatch):
    """A slow first read of a new step must not block token views of a step that is already built."""
    dump_dummy_run(tmp_path, steps=2)
    reader = DumpReader(tmp_path)
    new_index, ready_index = reader.joined(0).samples[0].index, reader.joined(1).samples[0].index
    reader.tokens(1, ready_index, start=0, end=1)
    building, release = threading.Event(), threading.Event()
    real_load = reader._torch_load

    def held_load(path, **kwargs):
        if path.name == "0.pt" and "rollout_data" in str(path):
            building.set()
            release.wait(timeout=10)
        return real_load(path, **kwargs)

    monkeypatch.setattr(reader, "_torch_load", held_load)
    builder = threading.Thread(target=lambda: reader.tokens(0, new_index, start=0, end=1))
    builder.start()
    assert building.wait(timeout=10)
    reader_of_ready_step = threading.Thread(target=lambda: reader.tokens(1, ready_index, start=0, end=1))
    reader_of_ready_step.start()
    reader_of_ready_step.join(timeout=5)
    finished_while_building = not reader_of_ready_step.is_alive()
    release.set()
    builder.join(timeout=10)

    assert finished_while_building, "a ready step waited for another step's mirror build"


def test_concurrent_first_reads_build_the_mirror_once(dumped, monkeypatch):
    """Two handlers reading a new step at once must not both rebuild its mirror through the same temp file."""
    reader = DumpReader(dumped)
    loads: list[str] = []
    real_load = reader._torch_load

    def slow_load(path, **kwargs):
        if "rollout_data" in str(path):
            loads.append(str(path))
            time.sleep(0.2)
        return real_load(path, **kwargs)

    monkeypatch.setattr(reader, "_torch_load", slow_load)
    payloads: list[dict] = []
    threads = [threading.Thread(target=lambda: payloads.append(reader.tokens(0, 0, start=0, end=4))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(payloads) == 2 and all(payload["token_ids"] for payload in payloads)
    assert len(loads) == 1, "both handlers rebuilt the mirror"


def test_unknown_sample_raises(dumped):
    with pytest.raises(KeyError):
        DumpReader(dumped).tokens(0, 10_000)


def test_train_row_values_match_joined(dumped):
    reader = DumpReader(dumped)
    joined = reader.joined(0)
    sample_index = joined.samples[0].index
    rollout_columns = reader._rollout_columns(0, sample_index, sample_occurrence=0, evaluation=False)
    lazy = reader._train_row_lazy(0, sample_index, rollout_columns)
    full = joined.train_rows[(sample_index, 0)]
    for field in ("log_probs", "ref_log_probs", "advantages"):
        lazy_v, full_v = getattr(lazy, field), getattr(full, field)
        assert (lazy_v is None) == (full_v is None)
        if lazy_v is not None:
            assert torch.equal(lazy_v, full_v)
