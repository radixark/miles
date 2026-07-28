import json
import os
import shutil
import statistics
import time

import polars as pl
import pytest
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader, _weight_version_summary
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

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
        row = joined.train_rows[sample.index]
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
    row = joined.train_rows[sample.index]
    assert payload["lp_diff"][0] == pytest.approx(float(row.log_probs[0] - row.rollout_log_probs[0]))


def test_tokens_null_the_stats_where_the_loss_is_masked(reader):
    """mask=0 positions hold placeholders the engine never scored: they serialize
    as null, never as numbers a chart would mistake for data."""
    masked = reader.tokens(0, REMOVED[0])
    assert masked["loss_mask"] and set(masked["loss_mask"]) == {0}
    for key in ("train_log_probs", "rollout_log_probs", "lp_diff", "imp_ratio", "ref_log_probs", "advantages"):
        assert masked[key] == [None] * len(masked["loss_mask"]), key

    row = reader.load_joined(0).train_rows[0]
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


def test_turns_counts_calls_not_spans():
    """A single generation call that spanned a weight update is still one turn."""
    from miles.dashboard.dump_reader import _weight_version_summary
    from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

    sample = Sample(group_index=0, index=0, prompt="p", tokens=[1, 2, 3], response="r", response_length=3, label="l")
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan("4", 0, 2), WeightVersionSpan("5", 2, 3)])
    ]

    assert _weight_version_summary(sample) == (["4", "5"], 1)


def test_turns_counts_a_call_that_carried_no_version():
    """An unstamped call still happened, so it must not vanish from the turn count."""
    from miles.dashboard.dump_reader import _weight_version_summary
    from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

    sample = Sample(group_index=0, index=0, prompt="p", tokens=[1, 2], response="r", response_length=2, label="l")
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan("4", 0, 1)]),
        WeightVersionsPerCall(spans=[]),
    ]

    assert _weight_version_summary(sample) == (["4"], 2)


class TestLegacyWeightVersionSummary:
    def test_reads_versions_from_a_pre_span_dump(self):
        """A dump that predates spans still reports its versions and turn count."""
        sample = Sample.from_dict({"status": "completed", "weight_versions": ["3", "4"], "tokens": [1, 2]})

        versions, turns = _weight_version_summary(sample)

        assert versions == ["3", "4"]
        assert turns == 2

    def test_prefers_spans_when_the_dump_has_them(self):
        """A current dump is read from its spans, in generation order."""
        sample = Sample(
            weight_versions=[
                WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
                WeightVersionsPerCall(spans=[WeightVersionSpan("4", 1, 2), WeightVersionSpan("5", 2, 3)]),
            ]
        )

        versions, turns = _weight_version_summary(sample)

        assert versions == ["3", "4", "5"]
        assert turns == 2

    def test_reports_nothing_when_the_sample_was_never_stamped(self):
        """A sample with no versions at all yields no series rather than a zero."""
        assert _weight_version_summary(Sample()) == ([], None)

    def test_repeated_legacy_versions_are_two_turns_but_not_mixed(self):
        """Two calls that happened to see the same weights are still two turns."""
        sample = Sample.from_dict({"status": "completed", "weight_versions": ["3", "3"], "tokens": [1, 2]})

        assert _weight_version_summary(sample) == (["3", "3"], 2)


class TestCurrentFormatWeightVersionSummary:
    def test_turns_counts_calls_when_every_call_is_unstamped(self):
        """Two calls that the engine never stamped report no versions but still count as two turns."""
        sample = Sample(weight_versions=[WeightVersionsPerCall(spans=[]), WeightVersionsPerCall(spans=[])])

        assert _weight_version_summary(sample) == ([], 2)

    def test_summary_does_not_mark_repeated_current_version_as_mixed(self, tmp_path):
        """Two calls that saw the same weights give two turns without flagging a mixed version."""
        sample = Sample(
            group_index=0,
            index=0,
            tokens=[1, 2],
            response_length=2,
            weight_versions=[
                WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
                WeightVersionsPerCall(spans=[WeightVersionSpan("3", 1, 2)]),
            ],
        )

        row = DumpReader(tmp_path)._summary_row(sample, None, rollout_id=0)

        assert row["turns"] == 2
        assert row["mixed_version"] is False
        assert row["weight_version"] == "3"


class TestSummaryCacheVersioning:
    def test_summary_invalidates_v2_cache_after_weight_version_schema_change(self, reader):
        """A summary cache stamped with the previous schema version is rebuilt and restamped at version 5."""
        import polars as pl

        expected = reader.summary(0)
        cache_path = reader.cache_dir / "rollout_0.parquet"
        sources_path = reader.cache_dir / "rollout_0.sources.json"
        pl.DataFrame({"sample_index": [-1]}).write_parquet(cache_path)
        sources_path.write_text(json.dumps(json.loads(sources_path.read_text()) | {"_summary_version": 2}))

        rebuilt = reader.summary(0)

        assert rebuilt.equals(expected)
        assert pl.read_parquet(cache_path).equals(expected)
        assert json.loads(sources_path.read_text())["_summary_version"] == 5


def test_pre_span_dump_survives_the_full_reader_pipeline(tmp_path):
    """A real dump downgraded to the pre-span format still rebuilds its parquet and summarises."""
    import polars as pl
    import torch

    dump_dummy_run(tmp_path, steps=1, dp_size=2, tp_dup=2, with_eval=False)
    shutil.rmtree(tmp_path / "dashboard_columns")
    for path in (tmp_path / "rollout_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        for data in pack["samples"]:
            data["weight_versions"] = [span["version"] for call in data["weight_versions"] for span in call]
        torch.save(pack, path)

    reader = DumpReader(tmp_path)
    df = reader.summary(0)

    assert df.height > 0
    agentic = df.filter(pl.col("sample_index") % 3 == 0)
    assert agentic["turns"].min() == 2
    assert agentic["mixed_version"].all()
    assert reader.tokens(0, int(agentic["sample_index"][0]))
