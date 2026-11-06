import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from miles.rollout import data_source
from miles.rollout.data_source import MixedRolloutDataSource, split_by_weight
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import RewardSpec


class _WordTokenizer:
    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [prompt.split() for prompt in prompts]}


@pytest.fixture(autouse=True)
def _word_tokenizer(monkeypatch):
    monkeypatch.setattr(data_source, "_load_prompt_tokenizer_and_processor", lambda args: (_WordTokenizer(), None))


def _write_jsonl(path: Path, prompts: list[str], **row_fields) -> Path:
    path.write_text("".join(json.dumps({"input": p, "label": p, **row_fields}) + "\n" for p in prompts))
    return path


def _make_source(tmp_path: Path, datasets: dict, **overrides) -> MixedRolloutDataSource:
    config = tmp_path / "mix.yaml"
    config.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False))
    defaults = dict(
        rollout_data_yaml_config=str(config),
        rollout_max_prompt_len=None,
        rollout_seed=42,
        rollout_shuffle=True,
        n_samples_per_prompt=2,
        input_key="input",
        label_key="label",
        metadata_key="metadata",
        tool_key="tools",
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
    )
    return MixedRolloutDataSource(SimpleNamespace(**{**defaults, **overrides}))


def _two_datasets(tmp_path: Path, *, math_size: int = 7, code_size: int = 50) -> dict:
    return {
        "math": {
            "path": str(_write_jsonl(tmp_path / "math.jsonl", [f"m{i}" for i in range(math_size)])),
            "samples_per_step": 3,
        },
        "code": {
            "path": str(_write_jsonl(tmp_path / "code.jsonl", [f"c{i}" for i in range(code_size)])),
            "samples_per_step": 5,
        },
    }


def _names(groups) -> list[str]:
    return [group[0].metadata["dataset_name"] for group in groups]


class TestSplitByWeight:
    @pytest.mark.parametrize("request_size", [1, 3, 8, 13])
    def test_the_long_run_split_matches_the_weights_exactly(self, request_size):
        """Exact after every multiple of the total weight (14) prompts, whatever the request size."""
        credits, totals = [0, 0, 0], Counter()
        for _ in range(14 * 3):
            counts, credits = split_by_weight(request_size, [3, 5, 6], credits)
            assert sum(counts) == request_size
            totals.update(dict(enumerate(counts)))
        total = 14 * 3 * request_size
        assert [totals[i] * 14 for i in range(3)] == [w * total for w in (3, 5, 6)]

    def test_one_prompt_at_a_time_rotates_by_weight(self):
        credits, picks = [0, 0], []
        for _ in range(8):
            counts, credits = split_by_weight(1, [3, 5], credits)
            picks.append(counts.index(1))
        assert Counter(picks) == {0: 3, 1: 5}
        assert credits == [0, 0]


class TestMixedRolloutDataSource:
    def test_groups_hold_one_prompt_and_ids_are_unique(self, tmp_path):
        source = _make_source(tmp_path, _two_datasets(tmp_path))
        groups = source.get_samples(8)

        assert len(groups) == 8
        for group in groups:
            assert len(group) == 2
            assert len({sample.prompt for sample in group}) == 1
            assert len({sample.group_index for sample in group}) == 1
        indices = [sample.index for group in groups for sample in group]
        assert indices == sorted(set(indices))
        assert Counter(_names(groups)) == {"math": 3, "code": 5}

    def test_a_small_dataset_wraps_on_its_own(self, tmp_path):
        source = _make_source(tmp_path, _two_datasets(tmp_path, math_size=7, code_size=50))
        drawn = [
            g[0].prompt for _ in range(10) for g in source.get_samples(8) if g[0].metadata["dataset_name"] == "math"
        ]

        assert len(drawn) == 30
        assert source.epoch_ids == {"math": 4, "code": 1}
        # 30 draws from 7 prompts: four full passes plus two prompts of the fifth
        assert sorted(Counter(drawn).values()) == [4, 4, 4, 4, 4, 5, 5]

    def test_resume_continues_with_exactly_the_same_prompts(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        source = _make_source(tmp_path, datasets)
        for size in (8, 1, 5):
            source.get_samples(size)
        source.save(tmp_path / "ckpt")
        expected = [[(s.prompt, s.index, s.metadata["dataset_name"]) for s in g] for g in source.get_samples(13)]

        resumed = _make_source(tmp_path, datasets)
        resumed.load(tmp_path / "ckpt")

        assert [
            [(s.prompt, s.index, s.metadata["dataset_name"]) for s in g] for g in resumed.get_samples(13)
        ] == expected

    def test_a_dataset_generate_function_reaches_its_samples_only(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["code"]["custom_generate_function_path"] = "json.dumps"
        source = _make_source(tmp_path, datasets)

        by_name = {group[0].metadata["dataset_name"]: group for group in source.get_samples(8)}
        assert {sample.generate_function_path for sample in by_name["code"]} == {"json.dumps"}
        assert {sample.generate_function_path for sample in by_name["math"]} == {None}

    def test_a_dataset_agent_block_reaches_its_samples_only(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["code"]["agent"] = {"max_turns": 3}
        source = _make_source(tmp_path, datasets)

        by_name = {group[0].metadata["dataset_name"]: group for group in source.get_samples(8)}
        assert [sample.metadata["agent_config"] for sample in by_name["code"]] == [{"max_turns": 3}] * 2
        # each sample gets its own copy, so an agent editing its config cannot leak into a sibling
        assert by_name["code"][0].metadata["agent_config"] is not by_name["code"][1].metadata["agent_config"]
        assert all("agent_config" not in sample.metadata for sample in by_name["math"])

    def test_a_dataset_agent_function_and_max_seq_len_reach_its_samples_only(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["code"] |= {"custom_agent_function_path": "json.loads", "max_seq_len": 4096}
        source = _make_source(tmp_path, datasets)

        by_name = {group[0].metadata["dataset_name"]: group for group in source.get_samples(8)}
        code, math = by_name["code"][0], by_name["math"][0]
        assert (code.agent_function_path, code.max_seq_len) == ("json.loads", 4096)
        assert (math.agent_function_path, math.max_seq_len) == (None, None)

    def test_a_dataset_response_budget_reaches_its_samples_only(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["code"]["max_response_len"] = 64
        source = _make_source(tmp_path, datasets)

        by_name = {group[0].metadata["dataset_name"]: group for group in source.get_samples(8)}
        assert {sample.max_response_len for sample in by_name["code"]} == {64}
        assert {sample.max_response_len for sample in by_name["math"]} == {None}

    def test_max_prompt_len_filters_only_its_dataset(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["math"]["path"] = str(_write_jsonl(tmp_path / "math.jsonl", ["short", "a much longer prompt"]))
        datasets["math"]["max_prompt_len"] = 1
        source = _make_source(tmp_path, datasets)

        assert [s.prompt for s in source.datasets["math"].samples] == ["short"]
        assert len(source.datasets["code"]) == 50

    def test_each_dataset_sets_its_own_reward_spec(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["math"]["rm_type"] = "math"
        datasets["code"]["custom_rm_path"] = "my_pkg.code_reward"
        source = _make_source(tmp_path, datasets)

        specs = {group[0].metadata["dataset_name"]: group[0].reward_spec for group in source.get_samples(8)}

        assert specs == {"math": RewardSpec(rm_type="math"), "code": RewardSpec(custom_rm_path="my_pkg.code_reward")}

    def test_a_dataset_without_reward_fields_keeps_the_run_wide_reward(self, tmp_path):
        source = _make_source(tmp_path, _two_datasets(tmp_path))
        assert all(group[0].reward_spec is None for group in source.get_samples(8))

    def test_non_mapping_metadata_fails_at_startup(self, tmp_path):
        """A string metadata column would crash the first request when the mixer tags the sample."""
        datasets = _two_datasets(tmp_path)
        datasets["math"]["path"] = str(_write_jsonl(tmp_path / "math.jsonl", ["p0", "p1"], metadata='{"a": 1}'))
        with pytest.raises(ValueError, match="'math': row 0 has non-mapping metadata"):
            _make_source(tmp_path, datasets)

    def test_a_prompt_row_cannot_choose_the_agent(self, tmp_path):
        """The agent travels on a Sample field only the mixer sets; a row's metadata key of that name is just data."""
        datasets = _two_datasets(tmp_path)
        datasets["math"]["path"] = str(
            _write_jsonl(tmp_path / "math.jsonl", ["p0"], metadata={"agent_function_path": "evil.run"})
        )
        group = next(
            g for g in _make_source(tmp_path, datasets).get_samples(8) if g[0].metadata["dataset_name"] == "math"
        )
        assert group[0].agent_function_path is None

    def test_a_dataset_empty_after_filtering_fails_at_startup(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["math"]["max_prompt_len"] = 1
        datasets["math"]["path"] = str(_write_jsonl(tmp_path / "math.jsonl", ["two words"]))
        with pytest.raises(ValueError, match="'math'.*no prompts"):
            _make_source(tmp_path, datasets)

    def test_resume_refuses_a_position_past_a_dataset_filtered_shorter(self, tmp_path):
        """A saved offset beyond the end used to make every later request hang on empty slices."""
        datasets = _two_datasets(tmp_path)
        prompts = ["s0", "s1", "s2", "s3", "long prompt a", "long prompt b", "long prompt c"]
        datasets["math"]["path"] = str(_write_jsonl(tmp_path / "math.jsonl", prompts))
        source = _make_source(tmp_path, datasets)
        source.get_samples(8)
        source.get_samples(8)
        assert source.offsets["math"] == 6
        source.save(tmp_path / "ckpt")

        # the run-wide limit is not a dataset setting, so only the position check catches it
        with pytest.raises(ValueError, match="past the end"):
            _make_source(tmp_path, datasets, rollout_max_prompt_len=1).load(tmp_path / "ckpt")

    def test_resume_with_new_weights_keeps_each_position_and_applies_the_new_ratio(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        source = _make_source(tmp_path, datasets)
        source.get_samples(8)
        source.save(tmp_path / "ckpt")

        datasets["math"]["samples_per_step"] = 1
        datasets["code"]["samples_per_step"] = 1
        resumed = _make_source(tmp_path, datasets)
        resumed.load(tmp_path / "ckpt")

        assert (resumed.offsets, resumed.epoch_ids) == (source.offsets, source.epoch_ids)
        assert Counter(_names(resumed.get_samples(100))) == {"math": 50, "code": 50}

    @pytest.mark.parametrize(
        "change",
        [
            {"custom_generate_function_path": "json.dumps"},
            {"rm_type": "math"},
            {"max_prompt_len": 100},
            {"max_response_len": 100},
        ],
    )
    def test_resume_refuses_a_changed_dataset_setting(self, tmp_path, change):
        """Groups drawn before the save still carry the old setting, so resuming under a new one would mix both."""
        datasets = _two_datasets(tmp_path)
        _make_source(tmp_path, datasets).save(tmp_path / "ckpt")

        datasets["code"] |= change
        with pytest.raises(ValueError, match=f"'code' changed {next(iter(change))}.*--rollout-data-reset-on-load"):
            _make_source(tmp_path, datasets).load(tmp_path / "ckpt")

    def test_a_reset_keeps_counting_sample_ids_but_starts_every_dataset_over(self, tmp_path):
        """Reusing trained sample ids after a reset would collide with the restored ownership records."""
        datasets = _two_datasets(tmp_path)
        source = _make_source(tmp_path, datasets)
        source.get_samples(8)
        source.save(tmp_path / "ckpt")

        del datasets["code"]
        reset = _make_source(tmp_path, datasets)
        reset.load_sample_ids(tmp_path / "ckpt")

        assert (reset.sample_index, reset.sample_group_index) == (source.sample_index, source.sample_group_index)
        assert reset.offsets == {"math": 0}

    def test_a_checkpoint_saved_before_settings_were_recorded_still_resumes(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        source = _make_source(tmp_path, datasets)
        source.get_samples(8)
        source.save(tmp_path / "ckpt")
        state = load_simple_checkpoint(directory=tmp_path / "ckpt")
        del state["settings"]
        save_simple_checkpoint(directory=tmp_path / "old", data=state)

        resumed = _make_source(tmp_path, datasets)
        resumed.load(tmp_path / "old")
        assert resumed.offsets == source.offsets

    def test_a_single_dataset_checkpoint_is_a_mismatch_that_a_reset_can_continue_from(self, tmp_path):
        """Moving a --prompt-data run onto a mix must reach the reset path, not crash on a missing key."""
        save_simple_checkpoint(
            directory=tmp_path / "ckpt",
            data={"sample_offset": 4, "epoch_id": 0, "sample_group_index": 5, "sample_index": 10, "metadata": {}},
        )
        source = _make_source(tmp_path, _two_datasets(tmp_path))
        with pytest.raises(data_source.RolloutDataResumeMismatch, match="single-dataset"):
            source.load(tmp_path / "ckpt")

        source.load_sample_ids(tmp_path / "ckpt")
        assert (source.sample_index, source.sample_group_index) == (10, 5)

    def test_a_checkpoint_without_data_source_state_is_a_mismatch(self, tmp_path):
        """A --disable-rollout-global-dataset run saves none, and the flag must be able to move it onto a mix."""
        source = _make_source(tmp_path, _two_datasets(tmp_path))
        (tmp_path / "ckpt").mkdir()

        with pytest.raises(data_source.RolloutDataResumeMismatch, match="no data source state") as err:
            source.load(tmp_path / "ckpt")
        assert err.value.needs_flag
        source.load_sample_ids(tmp_path / "ckpt")
        assert source.sample_index == 0

    def test_resume_refuses_a_different_dataset_list(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        _make_source(tmp_path, datasets).save(tmp_path / "ckpt")

        del datasets["code"]
        with pytest.raises(ValueError, match="now lists \\['math'\\]"):
            _make_source(tmp_path, datasets).load(tmp_path / "ckpt")


class TestSha256:
    def test_a_matching_hash_passes_and_a_wrong_one_stops_startup(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        actual = hashlib.sha256(Path(datasets["math"]["path"]).read_bytes()).hexdigest()

        datasets["math"]["sha256"] = actual[:8]
        _make_source(tmp_path, datasets)

        datasets["math"]["sha256"] = "0" * 8
        with pytest.raises(ValueError, match=f"expected {'0' * 8}, got {actual[:8]}"):
            _make_source(tmp_path, datasets)

    def test_a_file_rewritten_while_loading_is_refused(self, tmp_path, monkeypatch):
        """The file is hashed, then reopened to parse; a rewrite in between used to train on unpinned bytes."""
        datasets = _two_datasets(tmp_path)
        path = Path(datasets["math"]["path"])
        datasets["math"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:8]

        def rewrite_during_tokenizer_load(args):
            path.write_text(json.dumps({"input": "replacement data", "label": "replacement"}) + "\n")
            return _WordTokenizer(), None

        monkeypatch.setattr(data_source, "_load_prompt_tokenizer_and_processor", rewrite_during_tokenizer_load)
        with pytest.raises(ValueError, match="'math'.*changed while loading"):
            _make_source(tmp_path, datasets)

    def test_an_unpinned_file_is_hashed_once(self, tmp_path, monkeypatch):
        calls = []
        real = data_source._hash_file
        monkeypatch.setattr(data_source, "_hash_file", lambda path: calls.append(path) or real(path))
        _make_source(tmp_path, _two_datasets(tmp_path))
        assert len(calls) == 2

    def test_a_missing_hash_logs_the_value_to_pin(self, tmp_path, caplog):
        datasets = _two_datasets(tmp_path)
        actual = hashlib.sha256(Path(datasets["code"]["path"]).read_bytes()).hexdigest()

        with caplog.at_level("WARNING", logger="miles.rollout.data_source"):
            _make_source(tmp_path, datasets)
        # quoted, so an all-digit prefix stays a string when pasted into YAML
        assert f'sha256: "{actual[:8]}"' in caplog.text

    def test_resume_refuses_a_dataset_that_changed_since_the_checkpoint(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        source = _make_source(tmp_path, datasets)
        source.save(tmp_path / "ckpt")

        _write_jsonl(Path(datasets["code"]["path"]), ["edited"])
        with pytest.raises(ValueError, match="'code' changed since the checkpoint"):
            _make_source(tmp_path, datasets).load(tmp_path / "ckpt")
