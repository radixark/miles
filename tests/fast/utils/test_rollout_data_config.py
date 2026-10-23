import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from miles.utils.arguments import _resolve_data_source_path, get_miles_extra_args_provider
from miles.utils.rollout_data_config import (
    DEFAULT_DATA_SOURCE_PATH,
    MIXED_DATA_SOURCE_PATH,
    build_rollout_dataset_configs,
)


def _args(**overrides):
    config = {"datasets": {"math": {"path": "math.jsonl", "samples_per_step": 3}}}
    encoded = base64.b64encode(yaml.safe_dump(config).encode()).decode()
    return SimpleNamespace(
        **{
            "rollout_data_config": f"base64:{encoded}",
            "prompt_data": None,
            "data_source_path": DEFAULT_DATA_SOURCE_PATH,
            "rollout_global_dataset": True,
            "num_rollout": 10,
            "partial_rollout": False,
            "input_key": "prompt",
            "label_key": "answer",
            "metadata_key": "metadata",
            "tool_key": None,
            "rollout_max_prompt_len": 100,
            **overrides,
        }
    )


def _config_arg(raw: dict) -> str:
    return "base64:" + base64.b64encode(yaml.safe_dump(raw).encode()).decode()


def test_training_defaults_do_not_read_eval_overrides():
    args = _args(eval_input_key="eval_prompt", eval_label_key="eval_answer")

    (config,) = build_rollout_dataset_configs(args)

    assert (config.name, config.path, config.samples_per_step) == ("math", "math.jsonl", 3)
    assert (config.input_key, config.label_key, config.max_prompt_len) == ("prompt", "answer", 100)


def test_mapping_list_file_and_inline_forms_resolve_identically(tmp_path: Path):
    entry = {"path": "math.jsonl", "samples_per_step": 3, "sha256": "AB" * 32}
    path = tmp_path / "mix.yaml"
    path.write_text(yaml.safe_dump({"datasets": {"math": entry}}))
    (mapping,) = build_rollout_dataset_configs(_args(rollout_data_config=str(path)))
    (listed,) = build_rollout_dataset_configs(
        _args(rollout_data_config=_config_arg({"datasets": [{"name": "math", **entry}]}))
    )

    assert mapping == listed
    assert mapping.sha256 == "ab" * 32


def test_dataset_fields_override_defaults_and_launch_settings():
    raw = {
        "defaults": {"input_key": "question", "label_key": "solution", "max_prompt_len": 80},
        "datasets": {
            "math": {"path": "math.jsonl", "samples_per_step": 3, "label_key": None, "max_prompt_len": 40},
            "science": {"path": "science.jsonl", "samples_per_step": 5},
        },
    }

    math, science = build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))

    assert (math.input_key, math.label_key, math.max_prompt_len) == ("question", None, 40)
    assert (science.input_key, science.label_key, science.max_prompt_len) == ("question", "solution", 80)


@pytest.mark.parametrize("weight", [0, -1, 1.5, True, "3", None])
def test_invalid_weights_fail_before_loading_data(weight):
    raw = {"datasets": {"math": {"path": "math.jsonl", "samples_per_step": weight}}}
    with pytest.raises(ValueError, match="positive integer weight"):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))


@pytest.mark.parametrize("limit", [101, 0, -1, 2.5, True, "40"])
def test_prompt_limits_are_positive_integers_within_the_launch_bound(limit):
    raw = {"datasets": {"math": {"path": "math.jsonl", "samples_per_step": 1, "max_prompt_len": limit}}}
    with pytest.raises(ValueError, match="max_prompt_len"):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))


def test_null_prompt_limit_cannot_disable_the_launch_bound():
    raw = {"datasets": {"math": {"path": "math.jsonl", "samples_per_step": 1, "max_prompt_len": None}}}
    (config,) = build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))
    assert config.max_prompt_len == 100


@pytest.mark.parametrize("digest", ["", "bad", "x" * 64, 123, True])
def test_invalid_checksums_are_rejected(digest):
    raw = {"datasets": {"math": {"path": "math.jsonl", "samples_per_step": 1, "sha256": digest}}}
    with pytest.raises(ValueError, match="sha256"):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))


@pytest.mark.parametrize(
    "field", ["rm_type", "custom_rm_path", "max_response_len", "generate_function", "agent", "typo"]
)
def test_unimplemented_and_unknown_fields_do_not_silently_parse(field):
    raw = {"datasets": {"math": {"path": "math.jsonl", "samples_per_step": 1, field: "unused"}}}
    with pytest.raises(ValueError, match="Unsupported rollout dataset fields"):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))


@pytest.mark.parametrize("raw", [{}, {"datasets": {}}, {"datasets": []}, {"unrecognized": {}}])
def test_a_config_needs_at_least_one_dataset(raw):
    with pytest.raises(ValueError):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg(raw)))


def test_duplicate_dataset_names_are_rejected():
    entry = {"name": "same", "path": "math.jsonl", "samples_per_step": 1}
    with pytest.raises(ValueError, match="Duplicate rollout dataset name"):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg({"datasets": [entry, entry]})))


@pytest.mark.parametrize("defaults", [[], "bad", {"samples_per_step": 3}, {"sha256": "a" * 64}])
def test_defaults_only_accept_loading_fields(defaults):
    with pytest.raises(ValueError, match="defaults"):
        build_rollout_dataset_configs(_args(rollout_data_config=_config_arg({"datasets": {}, "defaults": defaults})))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"prompt_data": "single.jsonl"}, "--prompt-data"),
        ({"rollout_global_dataset": False}, "global rollout dataset"),
        ({"num_rollout": None}, "explicit --num-rollout"),
        ({"partial_rollout": True}, "--partial-rollout"),
        ({"data_source_path": "custom.Source"}, "custom --data-source-path"),
    ],
)
def test_mixer_argument_conflicts_fail_at_startup(overrides, message):
    with pytest.raises(ValueError, match=message):
        _resolve_data_source_path(_args(**overrides))


def test_config_selects_the_mixer_and_resolves_its_dataset_settings():
    args = _args()
    _resolve_data_source_path(args)
    assert args.data_source_path == MIXED_DATA_SOURCE_PATH
    assert args.rollout_datasets[0].name == "math"


def test_explicit_mixer_path_still_requires_a_config():
    with pytest.raises(ValueError, match="requires --rollout-data-config"):
        _resolve_data_source_path(_args(rollout_data_config=None, data_source_path=MIXED_DATA_SOURCE_PATH))


def test_existing_single_dataset_and_partial_rollout_selection_are_preserved():
    args = _args(rollout_data_config=None)
    _resolve_data_source_path(args)
    assert args.data_source_path == DEFAULT_DATA_SOURCE_PATH
    args.partial_rollout = True
    _resolve_data_source_path(args)
    assert args.data_source_path == "miles.rollout.data_source.LegacyRolloutDataSourceWithBuffer"


def test_mixer_flag_is_registered_by_the_real_argument_provider():
    import argparse
    from unittest.mock import patch

    with patch("sys.argv", ["test"]):
        parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-data-config",
            _args().rollout_data_config,
            "--rollout-batch-size",
            "8",
            "--num-rollout",
            "10",
        ]
    )

    _resolve_data_source_path(args)

    assert args.data_source_path == MIXED_DATA_SOURCE_PATH
    assert args.rollout_datasets[0].samples_per_step == 3
