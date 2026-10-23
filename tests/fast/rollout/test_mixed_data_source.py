import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from functools import reduce
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from miles.rollout.mixed_data_source import MixedRolloutDataSource
from miles.utils.rollout_data_config import build_rollout_dataset_configs


class _Tokenizer:
    def __call__(self, prompts, **kwargs):
        return {"input_ids": [list(range(len(prompt))) for prompt in prompts]}


@pytest.fixture
def make_source(tmp_path: Path, monkeypatch):
    processing = Mock(return_value=(_Tokenizer(), None))
    monkeypatch.setattr("miles.rollout.mixed_data_source.load_rollout_processing", processing)

    def make(*, weights=(3, 5), sizes=None, rows=None, shuffle=False, replicas=2, pin=False, dataset_overrides=None):
        sizes = sizes if sizes is not None else [7] * len(weights)
        datasets = {}
        for index, (weight, size) in enumerate(zip(weights, sizes, strict=True)):
            name = f"dataset_{index}"
            path = tmp_path / f"{name}.jsonl"
            content = (
                rows[index]
                if rows is not None
                else [
                    {"text": f"{name}/{row}", "label": str(row), "metadata": {"nested": {"row": row}}}
                    for row in range(size)
                ]
            )
            path.write_text("".join(json.dumps(row) + "\n" for row in content))
            datasets[name] = {"path": str(path), "samples_per_step": weight}
            if pin:
                datasets[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if dataset_overrides:
                datasets[name].update(dataset_overrides.get(index, {}))
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False))
        args = SimpleNamespace(
            rollout_data_config=str(config_path),
            rollout_max_prompt_len=None,
            input_key="text",
            label_key="label",
            tool_key=None,
            metadata_key="metadata",
            hf_checkpoint="not-loaded-in-this-test",
            chat_template_path=None,
            dump_details=None,
            multimodal_keys=None,
            apply_chat_template=False,
            apply_chat_template_kwargs={},
            rollout_seed=19,
            rollout_shuffle=shuffle,
            n_samples_per_prompt=replicas,
        )
        args.rollout_datasets = build_rollout_dataset_configs(args)
        return MixedRolloutDataSource(args), args, processing

    return make


def _counts(groups):
    return Counter(group[0].metadata["dataset_name"] for group in groups)


def _snapshot(groups):
    return [[sample.to_dict() for sample in group] for group in groups]


def test_mixes_prompts_before_replicating_into_independent_groups(make_source):
    source, _, processing = make_source(replicas=3)

    first = source.get_samples(8)
    second = source.get_samples(3)

    assert _counts(first) == {"dataset_0": 3, "dataset_1": 5}
    groups = first + second
    assert [group[0].group_index for group in groups] == list(range(11))
    assert [sample.index for group in groups for sample in group] == list(range(33))
    for group in groups:
        assert len({sample.prompt for sample in group}) == 1
        assert len({sample.group_index for sample in group}) == 1
        assert len({id(sample) for sample in group}) == 3
    assert processing.call_count == 1


@pytest.mark.parametrize("weights", [(3, 5), (1, 100), (2, 3, 7), (1, 1, 1, 1), (1000000, 1000000)])
def test_allocation_is_exact_over_full_weight_cycles_with_uneven_requests(make_source, weights):
    # Scale out a common factor: large integer weights must not require a huge request.
    divisor = reduce(gcd, weights)
    expected = {f"dataset_{i}": weight // divisor * 3 for i, weight in enumerate(weights)}
    source, _, _ = make_source(weights=weights)
    remaining = sum(expected.values())
    rng = random.Random(7)
    actual = Counter()
    while remaining:
        count = min(remaining, rng.randint(1, 13))
        actual.update(_counts(source.get_samples(count)))
        remaining -= count

    assert actual == expected


def test_request_partitioning_does_not_change_dataset_draws(make_source):
    source, args, _ = make_source(weights=(2, 3, 7), sizes=(2, 3, 5), shuffle=True)
    other = MixedRolloutDataSource(args)

    once = source.get_samples(57)
    partitioned = [group for count in (1, 2, 17, 1, 36) for group in other.get_samples(count)]

    def prompt_counts(groups):
        return Counter((group[0].metadata["dataset_name"], group[0].prompt) for group in groups)

    assert prompt_counts(once) == prompt_counts(partitioned)


def test_small_datasets_wrap_independently_several_times_per_request(make_source):
    source, _, _ = make_source(weights=(1, 1), sizes=(2, 3))

    groups = source.get_samples(20)

    assert len(groups) == 20
    assert _counts(groups) == {"dataset_0": 10, "dataset_1": 10}
    prompts = Counter(group[0].prompt for group in groups)
    assert prompts["dataset_0/0"] == prompts["dataset_0/1"] == 5
    assert [prompts[f"dataset_1/{i}"] for i in range(3)] == [4, 3, 3]


def test_dataset_tag_wins_and_returned_metadata_does_not_mutate_later_draws(make_source):
    rows = [[{"text": "only", "label": "answer", "metadata": {"dataset_name": "wrong", "nested": []}}]]
    source, _, _ = make_source(weights=(1,), sizes=(1,), rows=rows)

    (group,) = source.get_samples(1)
    group[0].metadata["nested"].append("changed")
    (later,) = source.get_samples(1)

    assert group[0].metadata["dataset_name"] == "dataset_0"
    assert group[1].metadata["nested"] == later[0].metadata["nested"] == []
    assert later[0].label == "answer"


def test_each_dataset_applies_its_own_prompt_filter(make_source):
    rows = [
        [{"text": "a", "label": "short"}, {"text": "long", "label": "long"}],
        [{"text": "bb", "label": "kept"}, {"text": "long", "label": "long"}],
    ]
    source, _, _ = make_source(
        weights=(1, 1), rows=rows, dataset_overrides={0: {"max_prompt_len": 1}, 1: {"max_prompt_len": 2}}
    )

    groups = source.get_samples(8)

    assert Counter(group[0].prompt for group in groups) == {"a": 4, "bb": 4}


@pytest.mark.parametrize("empty_after_filter", [False, True])
def test_empty_datasets_fail_with_the_dataset_name(make_source, empty_after_filter):
    rows = [[{"text": "long", "label": "x"}]] if empty_after_filter else [[]]
    with pytest.raises(ValueError, match="dataset_0.*no prompts"):
        make_source(weights=(1,), rows=rows, dataset_overrides={0: {"max_prompt_len": 1}})


@pytest.mark.parametrize("limit", [None, 100])
def test_raw_conversations_cannot_silently_bypass_prompt_limits(make_source, limit):
    prompt = [{"role": "user", "content": "a conversation"}]
    rows = [[{"text": prompt, "label": "x"}]]
    if limit is not None:
        with pytest.raises(ValueError, match="dataset_0.*requires --apply-chat-template"):
            make_source(weights=(1,), rows=rows, dataset_overrides={0: {"max_prompt_len": limit}})
    else:
        source, _, _ = make_source(weights=(1,), rows=rows)
        assert source.get_samples(1)[0][0].prompt == prompt


def test_invalid_row_metadata_fails_with_the_dataset_name(make_source):
    rows = [[{"text": "a prompt", "label": "x", "metadata": "not a mapping"}]]
    with pytest.raises(ValueError, match="dataset_0.*metadata must be a mapping"):
        make_source(weights=(1,), rows=rows)


def test_missing_pin_warns_with_a_digest_that_can_be_pasted_into_the_config(make_source, caplog):
    _, args, _ = make_source(weights=(1,))
    expected = hashlib.sha256(Path(args.rollout_datasets[0].path).read_bytes()).hexdigest()
    assert "no sha256 pin" in caplog.text
    assert expected in caplog.text


def test_bad_pin_reports_expected_and_actual_digests(make_source):
    with pytest.raises(ValueError, match="dataset_0.*expected.*got"):
        make_source(weights=(1,), dataset_overrides={0: {"sha256": "0" * 64}})


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("pin", [False, True])
def test_checkpoint_resume_restores_the_exact_next_groups_and_identities(make_source, tmp_path, shuffle, pin):
    source, args, _ = make_source(sizes=(2, 5), shuffle=shuffle, pin=pin, replicas=3)
    source.get_samples(3)  # Leaves both a fractional allocation and different dataset positions.
    source.save(tmp_path / "checkpoint")
    expected = [_snapshot(source.get_samples(count)) for count in (1, 17, 0, 8, 3)]

    restored = MixedRolloutDataSource(args)
    restored.load(tmp_path / "checkpoint")

    assert [_snapshot(restored.get_samples(count)) for count in (1, 17, 0, 8, 3)] == expected


def test_zero_sized_requests_do_not_advance_checkpointed_sampling_state(make_source, tmp_path):
    source, args, _ = make_source(shuffle=True)
    source.get_samples(3)
    source.save(tmp_path / "checkpoint")
    restored = MixedRolloutDataSource(args)
    restored.load(tmp_path / "checkpoint")

    assert source.get_samples(0) == []
    assert _snapshot(source.get_samples(9)) == _snapshot(restored.get_samples(9))


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_request_sizes_are_rejected(make_source, count):
    source, _, _ = make_source()
    with pytest.raises(ValueError, match="non-negative integer"):
        source.get_samples(count)


@pytest.mark.parametrize("change_after_construction", [False, True])
def test_resume_detects_modified_unpinned_data(make_source, tmp_path, change_after_construction):
    source, args, _ = make_source()
    source.get_samples(3)
    source.save(tmp_path / "checkpoint")
    if change_after_construction:
        restored = MixedRolloutDataSource(args)
    path = Path(args.rollout_datasets[0].path)
    path.write_text(path.read_text().replace("dataset_0/0", "changed/0"))
    if not change_after_construction:
        restored = MixedRolloutDataSource(args)

    with pytest.raises(ValueError, match="dataset_0.*changed on resume.*expected.*got"):
        restored.load(tmp_path / "checkpoint")


@pytest.mark.parametrize("change", ["weights", "order", "prompt_limit", "seed", "shuffle", "replicas", "metadata_key"])
def test_resume_rejects_changes_that_would_change_the_sample_sequence(make_source, tmp_path, change):
    source, args, _ = make_source()
    source.save(tmp_path / "checkpoint")
    args = copy.copy(args)
    configs = args.rollout_datasets
    if change == "weights":
        args.rollout_datasets = (replace(configs[0], samples_per_step=9), configs[1])
    elif change == "order":
        args.rollout_datasets = tuple(reversed(configs))
    elif change == "prompt_limit":
        args.rollout_datasets = (replace(configs[0], max_prompt_len=100), configs[1])
    elif change == "metadata_key":
        args.rollout_datasets = (replace(configs[0], metadata_key="other"), configs[1])
    elif change == "seed":
        args.rollout_seed += 1
    elif change == "shuffle":
        args.rollout_shuffle = True
    else:
        args.n_samples_per_prompt += 1
    restored = MixedRolloutDataSource(args)

    with pytest.raises(ValueError, match="configuration changed"):
        restored.load(tmp_path / "checkpoint")


def test_adding_a_matching_checksum_pin_does_not_invalidate_a_checkpoint(make_source, tmp_path):
    source, args, _ = make_source()
    source.get_samples(3)
    source.save(tmp_path / "checkpoint")
    expected = _snapshot(source.get_samples(9))
    args = copy.copy(args)
    args.rollout_datasets = tuple(
        replace(config, sha256=hashlib.sha256(Path(config.path).read_bytes()).hexdigest())
        for config in args.rollout_datasets
    )
    restored = MixedRolloutDataSource(args)
    restored.load(tmp_path / "checkpoint")

    assert _snapshot(restored.get_samples(9)) == expected


def test_shuffle_does_not_reseed_the_global_random_generator(make_source):
    state = random.getstate()
    source, _, _ = make_source(sizes=(2, 3), shuffle=True)
    source.get_samples(100)
    assert random.getstate() == state


@pytest.mark.parametrize("field,value", [("format_version", 999), ("credits", [1, 0]), ("sample_index", -1)])
def test_invalid_checkpoint_is_rejected_without_advancing_the_source(make_source, tmp_path, field, value):
    source, args, _ = make_source(shuffle=True)
    source.save(tmp_path / "checkpoint")
    expected = _snapshot(source.get_samples(8))
    state_path = tmp_path / "checkpoint" / "state.pt"
    state = torch.load(state_path, weights_only=False)
    state[field] = value
    torch.save(state, state_path)
    restored = MixedRolloutDataSource(args)

    with pytest.raises(ValueError):
        restored.load(tmp_path / "checkpoint")
    assert _snapshot(restored.get_samples(8)) == expected


def test_sliced_dataset_paths_hash_the_underlying_file_and_keep_the_slice(make_source, tmp_path):
    source, args, _ = make_source(weights=(1,), sizes=(5,))
    (config,) = args.rollout_datasets
    digest = hashlib.sha256(Path(config.path).read_bytes()).hexdigest()
    args.rollout_datasets = (replace(config, path=config.path + "@[1:3]", sha256=digest),)
    sliced = MixedRolloutDataSource(args)

    groups = sliced.get_samples(6)
    sliced.save(tmp_path / "sliced-checkpoint")
    restored = MixedRolloutDataSource(args)
    restored.load(tmp_path / "sliced-checkpoint")

    assert Counter(group[0].prompt for group in groups) == {"dataset_0/1": 3, "dataset_0/2": 3}
    assert _snapshot(restored.get_samples(3)) == _snapshot(sliced.get_samples(3))


def test_parquet_dataset_is_loaded_and_hashed_through_the_same_mixer(make_source, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, args, _ = make_source(weights=(1,))
    path = tmp_path / "prompts.parquet"
    pq.write_table(pa.table({"text": ["parquet prompt"], "label": ["answer"]}), path)
    (config,) = args.rollout_datasets
    args.rollout_datasets = (replace(config, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),)
    source = MixedRolloutDataSource(args)

    (group,) = source.get_samples(1)

    assert (group[0].prompt, group[0].label, group[0].metadata["dataset_name"]) == (
        "parquet prompt",
        "answer",
        "dataset_0",
    )
