import json
import os
import statistics
import time

import polars as pl
import pytest
from tests.fast.dashboard.dummy_dump import blank_samples, dump_dummy_run

from miles.dashboard.dump_reader import DumpReader

REMOVED = (3,)  # within-step positions marked remove_sample=True by the fixture


@pytest.fixture
def reader(tmp_path):
    dump_dummy_run(tmp_path, steps=2, dp_size=2, tp_dup=2, with_eval=True, remove_sample_indices=REMOVED)
    return DumpReader(tmp_path)


def _df_row(df, sample_index):
    matches = df.filter(df["sample_index"] == sample_index)
    assert matches.height == 1
    return matches.row(0, named=True)


def test_summary_matches_hand_computed(reader):
    joined = reader.load_joined(0)
    df = reader.summary(0)
    assert df.height == len(joined.samples)

    for sample in joined.samples:
        entry = _df_row(df, sample.index)
        row = joined.train_rows[(sample.index, 0)]
        assert entry["group_index"] == sample.group_index
        assert entry["response_length"] == sample.response_length
        assert entry["raw_reward"] == sample.reward
        assert entry["normalized_reward"] == pytest.approx(row.reward)
        assert entry["dumped_rank"] == row.rank

        mask = row.loss_mask > 0
        if mask.any():
            expected = (row.log_probs - row.rollout_log_probs).abs()[mask]
            assert entry["mean_abs_lp_diff"] == pytest.approx(float(expected.mean()), rel=1e-5)
            assert entry["max_abs_lp_diff"] == pytest.approx(float(expected.max()), rel=1e-5)
            assert entry["mean_imp_ratio"] == pytest.approx(
                float((row.log_probs - row.rollout_log_probs).exp()[mask].mean()), rel=1e-5
            )
            assert entry["mean_entropy"] == pytest.approx(float(row.entropy[mask].mean()), rel=1e-5)


def test_summary_removed_sample_has_no_masked_stats(reader):
    df = reader.summary(0)
    entry = _df_row(df, REMOVED[0])  # step 0: Sample.index == within-step position
    assert entry["remove_sample"] is True
    for column in ("mean_entropy", "mean_abs_lp_diff", "mean_imp_ratio", "adv_mean", "return_mean"):
        assert entry[column] is None, column


def test_summary_parquet_cache_hit_and_invalidation(reader, monkeypatch):
    df = reader.summary(0)
    assert (reader.cache_dir / "rollout_0.parquet").exists()

    def boom(self, path):
        raise RuntimeError("recompute attempted")

    monkeypatch.setattr(DumpReader, "_torch_load", boom)
    reader._joined_cache.clear()
    assert reader.summary(0).equals(df)  # served from parquet, no dump loads

    stamp = time.time() + 5
    os.utime(next(reader.train_dir.glob("0_*.pt")), (stamp, stamp))
    with pytest.raises(RuntimeError, match="recompute attempted"):
        reader.summary(0)  # source changed -> cache invalidated -> recompute


def test_groups(reader):
    joined = reader.load_joined(0)
    groups_df = reader.groups(0)
    rewards_by_group = {}
    for sample in joined.samples:
        rewards_by_group.setdefault(sample.group_index, []).append(sample.reward)
    assert groups_df.height == len(rewards_by_group)

    for entry in groups_df.iter_rows(named=True):
        rewards = rewards_by_group[entry["group_index"]]
        assert entry["n"] == len(rewards)
        assert entry["reward_mean"] == pytest.approx(statistics.mean(rewards))
        assert entry["reward_std"] == pytest.approx(statistics.stdev(rewards))
        assert entry["zero_std"] == (statistics.stdev(rewards) <= 1e-12)


def test_step_aggregates(reader):
    aggregates = reader.step_aggregates()
    assert aggregates["rollout_id"].to_list() == [0, 1]
    for entry in aggregates.iter_rows(named=True):
        df = reader.summary(entry["rollout_id"])
        assert entry["n_samples"] == df.height
        assert entry["reward_mean"] == pytest.approx(df["raw_reward"].mean())
        assert entry["mean_abs_lp_diff"] is not None


def test_tokens_full_range(reader):
    joined = reader.load_joined(0)
    sample = joined.samples[0]
    payload = reader.tokens(0, sample.index)
    prompt_len = len(sample.tokens) - sample.response_length

    assert payload["total_len"] == len(sample.tokens)
    assert payload["prompt_len"] == payload["response_offset"] == prompt_len
    assert payload["token_ids"] == sample.tokens
    assert len(payload["train_log_probs"]) == sample.response_length
    assert len(payload["imp_ratio"]) == sample.response_length
    assert payload["rollout_log_probs"] == pytest.approx(sample.rollout_log_probs)
    row = joined.train_rows[(sample.index, 0)]
    assert payload["lp_diff"][0] == pytest.approx(float(row.log_probs[0] - row.rollout_log_probs[0]))


def test_tokens_null_the_stats_where_the_loss_is_masked(reader):
    """mask=0 positions hold placeholders the engine never scored: they serialize
    as null, never as numbers a chart would mistake for data."""
    masked = reader.tokens(0, REMOVED[0])
    assert masked["loss_mask"] and set(masked["loss_mask"]) == {0}
    for key in ("train_log_probs", "rollout_log_probs", "lp_diff", "imp_ratio", "ref_log_probs", "advantages"):
        assert masked[key] == [None] * len(masked["loss_mask"]), key

    row = reader.load_joined(0).train_rows[(0, 0)]
    unmasked = reader.tokens(0, 0)
    assert set(unmasked["loss_mask"]) == {1}
    assert unmasked["train_log_probs"] == pytest.approx([float(v) for v in row.log_probs])


def test_tokens_window_straddles_response_boundary(reader):
    sample = reader.load_joined(0).samples[0]
    prompt_len = len(sample.tokens) - sample.response_length

    payload = reader.tokens(0, sample.index, start=prompt_len - 2, end=prompt_len + 3)
    assert len(payload["token_ids"]) == 5
    assert payload["response_offset"] == 2
    assert len(payload["train_log_probs"]) == 3  # stats only cover the response overlap

    prompt_only = reader.tokens(0, sample.index, start=0, end=2)
    assert prompt_only["train_log_probs"] == []
    assert prompt_only["response_offset"] == 2

    clamped = reader.tokens(0, sample.index, start=0, end=10**6)
    assert clamped["end"] == len(sample.tokens)

    with pytest.raises(ValueError, match="empty token range"):
        reader.tokens(0, sample.index, start=5, end=5)

    with pytest.raises(KeyError, match="unknown sample_index"):
        reader.tokens(0, 10**6)


def test_tokens_decode_via_dumped_tokenizer(reader):
    payload = reader.tokens(0, 0)
    assert payload["token_text"] == [f"t{tid}" for tid in payload["token_ids"]]


def test_tokens_eval_sample_has_rollout_side_only(reader):
    payload = reader.tokens(0, 0, evaluation=True)
    assert payload["rollout_log_probs"] is not None
    for key in ("train_log_probs", "lp_diff", "imp_ratio", "entropy", "advantages", "loss_mask"):
        assert payload[key] is None, key


def test_tokens_without_entropy_or_tokenizer(tmp_path):
    dump_dummy_run(tmp_path, steps=1, with_entropy=False, with_eval=False, with_tokenizer=False)
    reader = DumpReader(tmp_path)
    payload = reader.tokens(0, 0)
    assert payload["entropy"] is None and payload["ref_entropy"] is None
    assert payload["train_log_probs"] is not None
    assert payload["token_text"] is None
    assert _df_row(reader.summary(0), 0)["mean_entropy"] is None


def test_joined_lru_eviction(tmp_path):
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    reader = DumpReader(tmp_path, tensor_lru=1)
    first = reader.joined(0)
    assert reader.joined(0) is first  # cache hit
    reader.joined(1)
    assert list(reader._joined_cache) == [(1, False)]  # 0 evicted


@pytest.mark.skipif(
    "MILES_DASHBOARD_REALDATA_DIR" not in os.environ,
    reason="set MILES_DASHBOARD_REALDATA_DIR to a real --dump-details dir",
)
def test_realdata_views(tmp_path):
    reader = DumpReader(os.environ["MILES_DASHBOARD_REALDATA_DIR"], cache_dir=tmp_path)
    df = reader.summary(0)
    assert df.height == 256
    assert df["truncated"].cast(int).sum() == 112  # measured on qwen30b-dash step 0
    assert df["mean_abs_lp_diff"].null_count() == 0
    assert df["mean_entropy"].null_count() == 0

    groups_df = reader.groups(0)
    assert groups_df.height == 32  # 256 samples / 8 per prompt

    sample_index = int(df["sample_index"][0])
    payload = reader.tokens(0, sample_index, start=0, end=64)
    assert len(payload["token_ids"]) == 64
    assert payload["token_text"] is not None

    aggregates = reader.step_aggregates()
    assert aggregates.height == 5


def test_summary_and_tokens_survive_dump_without_log_probs(tmp_path):
    """A 744B-scale run may not dump train log_probs at all: everything
    derived from them degrades to None instead of KeyError -> HTTP 404
    (disagg report 2026-07-14)."""
    import torch

    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    from miles.dashboard.dump_reader import DumpReader

    dump_dummy_run(tmp_path)
    for shard in (tmp_path / "train_data").glob("0_*.pt"):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        assert payload["rollout_data"].pop("log_probs") is not None  # must actually remove it
        torch.save(payload, shard)

    reader = DumpReader(tmp_path)
    df = reader.summary(0)
    assert df["mean_abs_lp_diff"].is_null().all()
    assert df["mean_imp_ratio"].is_null().all()
    assert df["reward"].null_count() < df.height  # the rest of the summary is intact
    assert reader.step_aggregates().height >= 1  # the metrics page path


def test_staleness_and_agentic_columns(reader):
    import polars as pl

    df = reader.summary(0)
    agentic = df.filter(pl.col("sample_index") % 3 == 0)
    plain = df.filter(pl.col("sample_index") % 3 != 0)
    # dummy dump: every third sample is two-turn with mixed versions + one tool message
    assert agentic["mixed_version"].all() and agentic["turns"].min() == 2
    assert agentic["tool_calls"].min() == 1
    assert not plain["mixed_version"].any() and plain["turns"].max() == 1
    assert plain["tool_calls"].is_null().all()  # string prompts: nothing to count
    assert (agentic["weight_version"].cast(pl.Int64) - agentic["weight_version_min"] == 1).all()

    aggregates = reader.step_aggregates()
    assert 0 < aggregates["mixed_version_frac"][0] < 1


def test_groups_null_rewards_are_not_zero_std(tmp_path):
    # missing train dumps leave raw_reward null for every sample; that is
    # absent data, not a degenerate group
    reader = DumpReader(tmp_path)
    reader.summary = lambda rollout_id, evaluation=False: pl.DataFrame(
        {
            "group_index": [0, 0, 1, 1],
            "raw_reward": pl.Series([None] * 4, dtype=pl.Float64),
            "response_length": [1, 2, 3, 4],
            "truncated": [False] * 4,
        }
    )
    assert not reader.groups(0)["zero_std"].any()


def test_summary_columns_declaration_matches_reality(reader):
    # SUMMARY_COLUMNS only shapes the no-sample case, so nothing else would
    # notice it drifting: pin it against the columns a real summary produces
    assert list(DumpReader.SUMMARY_COLUMNS) == reader.summary(0).columns
    assert list(DumpReader.SUMMARY_COLUMNS) == reader.summary(0, evaluation=True).columns


def test_step_with_no_samples_reads_as_empty_not_broken(tmp_path):
    """A step can be dumped with no samples at all. Its summary must still
    carry the full schema, or every view below it fails on a missing column
    instead of reporting an empty step -- and because step_aggregates() walks
    every step, one such step would take the whole metrics page down with it."""
    dump_dummy_run(tmp_path, steps=3, with_eval=False)
    blank_samples(tmp_path, 1)  # a MIDDLE step, so step_aggregates must walk past it
    reader = DumpReader(tmp_path)

    empty = reader.summary(1)
    assert empty.height == 0
    assert empty.columns == list(DumpReader.SUMMARY_COLUMNS)
    assert empty["sample_index"].to_list() == []  # the trajectories endpoint indexes this

    assert reader.groups(1).height == 0
    assert "zero_std" in reader.groups(1).columns

    aggregates = reader.step_aggregates()
    assert aggregates["rollout_id"].to_list() == [0, 1, 2]
    blank = aggregates.filter(pl.col("rollout_id") == 1).row(0, named=True)
    assert blank["n_samples"] == 0
    assert blank["reward_mean"] is None
    # the steps on either side are still measured normally
    assert aggregates.filter(pl.col("rollout_id") == 2).row(0, named=True)["n_samples"] == reader.summary(2).height


def test_empty_step_survives_the_parquet_cache(tmp_path):
    # the all-null frame is written to parquet and read back on the next call
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    blank_samples(tmp_path, 1)
    assert DumpReader(tmp_path, cache_dir=tmp_path / "c").summary(1).columns == list(DumpReader.SUMMARY_COLUMNS)
    reloaded = DumpReader(tmp_path, cache_dir=tmp_path / "c").summary(1)  # served from the parquet written above
    assert reloaded.height == 0
    assert reloaded.columns == list(DumpReader.SUMMARY_COLUMNS)


def test_trajectory_occurrences_follow_the_full_sample_order(tmp_path):
    """The trajectory sidecar only holds samples that recorded a conversation,
    so file positions cannot number occurrences: when the first of two TITO
    leaves sharing an index has no conversation, the recorded one is
    occurrence 1 in summary/tokens and must be occurrence 1 here too. The
    writer persists that numbering in each row."""
    dump_dummy_run(tmp_path, steps=1, duplicate_first_sample_index=True)
    sidecar = tmp_path / "trajectory" / "0.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    index0 = [row for row in rows if int(row["sample_index"]) == 0]
    assert [row["sample_occurrence"] for row in index0] == [0, 1]  # full-list numbering, persisted
    # leaf 0 recorded no conversation: only the occurrence-1 row remains
    rows.remove(index0[0])
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))

    reader = DumpReader(tmp_path)
    assert int(reader.trajectory_messages(0, 0, sample_occurrence=1)["sample_index"]) == 0
    with pytest.raises(KeyError):
        reader.trajectory_messages(0, 0, sample_occurrence=0)  # the conversationless leaf


def test_trajectory_legacy_sidecar_falls_back_to_file_positions(tmp_path):
    """Sidecars written before the occurrence column keep working, numbered by
    file position (the best available for them)."""
    dump_dummy_run(tmp_path, steps=1, duplicate_first_sample_index=True)
    sidecar = tmp_path / "trajectory" / "0.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    for row in rows:
        row.pop("sample_occurrence")
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))

    reader = DumpReader(tmp_path)
    assert int(reader.trajectory_messages(0, 0, sample_occurrence=1)["sample_index"]) == 0


def test_pre_fix_columnless_cache_is_not_served(tmp_path):
    """A cache dir populated before the no-sample schema fix holds a columnless
    parquet for the aborted step, and its mtime stamps still match the dump.
    Only SUMMARY_VERSION tells it apart from a valid cache, so the schema fix
    had to ship with a bump past 4 -- the version that wrote those caches."""
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    blank_samples(tmp_path, 1)
    reader = DumpReader(tmp_path, cache_dir=tmp_path / "c")
    reader.cache_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([], strict=False).write_parquet(reader.cache_dir / "rollout_1.parquet")
    stale = {**reader._source_stamps(1, evaluation=False), "_summary_version": 4}
    (reader.cache_dir / "rollout_1.sources.json").write_text(json.dumps(stale))

    assert reader.summary(1).columns == list(DumpReader.SUMMARY_COLUMNS)
    assert DumpReader.SUMMARY_VERSION > 4  # regressing the bump re-serves those caches
