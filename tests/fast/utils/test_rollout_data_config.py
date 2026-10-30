import base64
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import patch

import pytest
import yaml

from miles.ray.rollout.rollout_executor import _compute_rollout_function_config
from miles.utils.args.schema import Arg, BaseConfig
from miles.utils.arguments import parse_args
from miles.utils.function_registry import function_registry
from miles.utils.rollout_data_config import (
    RolloutDatasetConfig,
    check_dataset_length_limits,
    load_rollout_data_config,
    parse_rollout_datasets,
)

_SHA = "9f2c0e41"

_DOCUMENT = {
    "datasets": {
        "math": {"path": "/data/math.jsonl", "samples_per_step": 3, "sha256": _SHA},
        "agentic": {
            "path": "/data/agentic.jsonl",
            "samples_per_step": 5,
            "max_prompt_len": 8192,
            "max_response_len": 32768,
            "generate_function": "miles.rollout.generate_hub.single_turn.generate",
            "agent": {"max_turns": 50, "sandbox": {"cpu": 2}},
        },
    }
}


def _inline(document: object) -> str:
    return "base64:" + base64.b64encode(yaml.safe_dump(document, sort_keys=False).encode()).decode()


def _with_dataset(**fields: object) -> dict:
    return {"datasets": {"math": {"path": "/data/math.jsonl", "samples_per_step": 1, **fields}}}


class TestLoad:
    def test_file_and_inline_payload_give_the_same_datasets(self, tmp_path):
        config_file = tmp_path / "mix.yaml"
        config_file.write_text(yaml.safe_dump(_DOCUMENT, sort_keys=False))

        from_file = load_rollout_data_config(str(config_file))

        assert from_file == load_rollout_data_config(_inline(_DOCUMENT))
        assert from_file == [
            RolloutDatasetConfig(name="math", path="/data/math.jsonl", samples_per_step=3, sha256=_SHA),
            RolloutDatasetConfig(
                name="agentic",
                path="/data/agentic.jsonl",
                samples_per_step=5,
                max_prompt_len=8192,
                max_response_len=32768,
                generate_function="miles.rollout.generate_hub.single_turn.generate",
                agent={"max_turns": 50, "sandbox": {"cpu": 2}},
            ),
        ]


class TestRejects:
    @pytest.mark.parametrize(
        "document",
        [
            [],
            {},
            {"datasets": {}},
            {"datasets": [{"name": "math", "path": "/data/math.jsonl", "samples_per_step": 1}]},
            {**_with_dataset(), "defaults": {}},
            {"datasets": {"math": "/data/math.jsonl"}},
        ],
    )
    def test_a_malformed_document(self, document):
        with pytest.raises(ValueError, match="--rollout-data-yaml-config"):
            parse_rollout_datasets(document)

    @pytest.mark.parametrize(
        "fields",
        [
            {"sample_per_step": 2},
            {"rm_type": "math"},
            {"name": "other"},
            {"samples_per_step": 0},
            {"samples_per_step": True},
            {"samples_per_step": "3"},
            {"samples_per_step": 3.0},
            {"path": ""},
            {"max_prompt_len": 0},
            {"max_response_len": -1},
            {"generate_function": ""},
            {"agent": ["max_turns", 50]},
            {"sha256": _SHA.upper()},
            {"sha256": _SHA[:-1]},
            {"sha256": _SHA + "0" * 56},
        ],
    )
    def test_a_bad_dataset_field(self, fields):
        with pytest.raises(ValueError, match="dataset 'math'"):
            parse_rollout_datasets(_with_dataset(**fields))

    @pytest.mark.parametrize("missing", ["path", "samples_per_step"])
    def test_a_missing_required_field(self, missing):
        document = _with_dataset()
        del document["datasets"]["math"][missing]
        with pytest.raises(ValueError, match="dataset 'math'"):
            parse_rollout_datasets(document)


class TestLengthLimits:
    @pytest.mark.parametrize(
        ("fields", "prompt_bound", "response_bound"),
        [
            ({"max_prompt_len": 8192, "max_response_len": 32768}, 8192, 32768),
            ({"max_prompt_len": 10**6, "max_response_len": 10**6}, None, None),
            ({}, 1, 1),
        ],
    )
    def test_a_limit_up_to_the_run_value_is_accepted(self, fields, prompt_bound, response_bound):
        check_dataset_length_limits(
            parse_rollout_datasets(_with_dataset(**fields)),
            rollout_max_prompt_len=prompt_bound,
            rollout_max_response_len=response_bound,
        )

    @pytest.mark.parametrize(
        ("fields", "flag"),
        [
            ({"max_prompt_len": 8193}, "--rollout-max-prompt-len 8192"),
            ({"max_response_len": 32769}, "--rollout-max-response-len 32768"),
        ],
    )
    def test_a_limit_above_the_run_value_is_rejected(self, fields, flag):
        with pytest.raises(ValueError, match=flag):
            check_dataset_length_limits(
                parse_rollout_datasets(_with_dataset(**fields)),
                rollout_max_prompt_len=8192,
                rollout_max_response_len=32768,
            )


_BASE_ARGV = [
    "pytest",
    "--train-backend",
    "fsdp",
    "--ci-test",
    "--no-enable-sample-ownership-checker",
    "--rollout-batch-size",
    "2",
    "--n-samples-per-prompt",
    "1",
    "--num-rollout",
    "1",
    "--rollout-num-gpus",
    "4",
    "--rollout-num-gpus-per-engine",
    "2",
    "--hf-checkpoint",
    "Qwen/Qwen3-0.6B",
    "--rm-type",
    "math",
    "--rollout-max-response-len",
    "32768",
]


def _parse(*extra_argv: str):
    with patch("sys.argv", _BASE_ARGV + list(extra_argv)):
        return parse_args()


class TestStartupValidation:
    def test_a_valid_config_selects_the_mixed_data_source(self):
        args = _parse("--rollout-data-yaml-config", _inline(_with_dataset(max_prompt_len=8192, sha256=_SHA)))
        assert args.data_source_path.path == "miles.rollout.data_source.MixedRolloutDataSource"

    def test_a_per_dataset_response_budget_fails_at_startup_until_samples_carry_it(self):
        with pytest.raises(NotImplementedError, match="max_response_len not supported yet"):
            _parse("--rollout-data-yaml-config", _inline(_with_dataset(max_response_len=64)))

    def test_it_needs_num_rollout(self):
        i = _BASE_ARGV.index("--num-rollout")
        argv = _BASE_ARGV[:i] + _BASE_ARGV[i + 2 :] + ["--num-epoch", "1"]
        with patch("sys.argv", argv + ["--rollout-data-yaml-config", _inline(_with_dataset())]):
            with pytest.raises(ValueError, match="needs --num-rollout"):
                parse_args()

    def test_it_cannot_be_combined_with_partial_rollout(self):
        with pytest.raises(ValueError, match="--partial-rollout"):
            _parse("--rollout-data-yaml-config", _inline(_with_dataset()), "--partial-rollout")

    def test_it_cannot_be_combined_with_a_custom_data_source(self):
        with pytest.raises(ValueError, match="drop --data-source-path"):
            _parse("--rollout-data-yaml-config", _inline(_with_dataset()), "--data-source-path", "my_pkg.MySource")

    def test_it_cannot_be_combined_with_prompt_data(self):
        with pytest.raises(ValueError, match="replaces --prompt-data"):
            _parse("--rollout-data-yaml-config", _inline(_DOCUMENT), "--prompt-data", "/dev/null")

    def test_it_cannot_be_combined_with_disabling_the_global_dataset(self):
        with pytest.raises(ValueError, match="--disable-rollout-global-dataset"):
            _parse("--rollout-data-yaml-config", _inline(_DOCUMENT), "--disable-rollout-global-dataset")

    def test_a_dataset_limit_above_the_run_value_fails_at_startup(self):
        too_long = _with_dataset(max_response_len=32769)
        with pytest.raises(ValueError, match="exceeds --rollout-max-response-len 32768"):
            _parse("--rollout-data-yaml-config", _inline(too_long))

    def test_the_prompt_limit_derived_from_the_context_length_bounds_a_dataset(self):
        """`--rollout-max-prompt-len` defaults from `--rollout-max-context-len`; the check must see that value."""
        too_long = _with_dataset(max_prompt_len=8192)
        with pytest.raises(ValueError, match="exceeds --rollout-max-prompt-len 8191"):
            _parse("--rollout-data-yaml-config", _inline(too_long), "--rollout-max-context-len", "8192")

    def test_unset_leaves_startup_unchanged(self):
        assert _parse("--prompt-data", "/dev/null").rollout_data_yaml_config is None


class _TypedGenerateConfig(BaseConfig):
    typed_marker: Annotated[int, Arg()] = 41


class TestDatasetHookSettings:
    def test_a_dataset_generate_function_gets_its_own_flags(self):
        """A hook named only in the YAML must have its flags registered, resolved and merged into the rollout view."""
        document = _with_dataset(generate_function="test3947.typed_generate")
        with function_registry.temporary(
            "test3947.typed_generate", SimpleNamespace(config_class=_TypedGenerateConfig)
        ):
            args = _parse("--rollout-data-yaml-config", _inline(document), "--typed-marker", "7")
            view = _compute_rollout_function_config(args, args.rollout_function_path)

        assert args.rollout_dataset_functions["test3947.typed_generate"].config.typed_marker == 7
        assert view.typed_marker == 7

    def test_a_dataset_max_seq_len_above_the_agentic_limit_fails_at_startup(self):
        """--max-seq-len is registered by the dataset's agentic generate function itself, then bounds the dataset."""
        document = _with_dataset(
            generate_function="miles.rollout.generate_hub.agentic_tool_call.generate", max_seq_len=70000
        )
        with pytest.raises(ValueError, match="exceeds --max-seq-len 65536"):
            _parse("--rollout-data-yaml-config", _inline(document), "--max-seq-len", "65536")

    def test_a_run_without_yaml_hooks_has_none(self):
        assert _parse("--prompt-data", "/dev/null").rollout_dataset_functions == {}
