import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from miles.rollout import data_source
from miles.rollout.data_source import MixedRolloutDataSource, split_by_weight


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

    def test_max_prompt_len_filters_only_its_dataset(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        datasets["math"]["path"] = str(_write_jsonl(tmp_path / "math.jsonl", ["short", "a much longer prompt"]))
        datasets["math"]["max_prompt_len"] = 1
        source = _make_source(tmp_path, datasets)

        assert [s.prompt for s in source.datasets["math"].samples] == ["short"]
        assert len(source.datasets["code"]) == 50

    def test_non_mapping_metadata_fails_at_startup(self, tmp_path):
        """A string metadata column would crash the first request when the mixer tags the sample."""
        datasets = _two_datasets(tmp_path)
        datasets["math"]["path"] = str(_write_jsonl(tmp_path / "math.jsonl", ["p0", "p1"], metadata='{"a": 1}'))
        with pytest.raises(ValueError, match="'math': row 0 has non-mapping metadata"):
            _make_source(tmp_path, datasets)

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

        datasets["math"]["max_prompt_len"] = 1
        with pytest.raises(ValueError, match="past the end"):
            _make_source(tmp_path, datasets).load(tmp_path / "ckpt")

    def test_resume_refuses_a_different_dataset_list(self, tmp_path):
        datasets = _two_datasets(tmp_path)
        _make_source(tmp_path, datasets).save(tmp_path / "ckpt")

        del datasets["code"]
        with pytest.raises(ValueError, match="now lists \\['math'\\]"):
            _make_source(tmp_path, datasets).load(tmp_path / "ckpt")
