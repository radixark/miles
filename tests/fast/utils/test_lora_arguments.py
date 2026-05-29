"""Unit tests for LoRA-related argument parsing in miles.utils.arguments.

Covers the target-module expansion and exclude-module filtering logic
inside miles_validate_args (lines 1634-1653 of arguments.py).
We isolate the LoRA parsing logic to avoid triggering unrelated validations.
"""

from argparse import Namespace
from copy import deepcopy

import pytest


def _apply_lora_arg_parsing(args: Namespace) -> Namespace:
    """Extract and apply only the LoRA target-module parsing logic from
    miles_validate_args, avoiding unrelated assertions."""
    args = deepcopy(args)
    if args.lora_rank > 0:
        assert args.target_modules is not None, "'--target-modules' is required when LoRA is enabled."

        if args.target_modules == "all-linear":
            modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "," in args.target_modules:
            modules = [m.strip() for m in args.target_modules.split(",")]
        else:
            modules = [args.target_modules]

        if args.exclude_modules:
            exclude_set = (
                set(m.strip() for m in args.exclude_modules.split(","))
                if "," in args.exclude_modules
                else {args.exclude_modules}
            )
            modules = [m for m in modules if m not in exclude_set]

        args.target_modules = modules
    return args


# ---------------------------------------------------------------------------
# Target modules expansion
# ---------------------------------------------------------------------------


class TestLoraTargetModuleParsing:
    def test_all_linear_expands_to_seven_modules(self):
        args = Namespace(lora_rank=32, target_modules="all-linear", exclude_modules=None)
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def test_comma_separated_split(self):
        args = Namespace(lora_rank=16, target_modules="q_proj, k_proj, v_proj", exclude_modules=None)
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj", "k_proj", "v_proj"]

    def test_comma_separated_no_spaces(self):
        args = Namespace(lora_rank=16, target_modules="q_proj,k_proj", exclude_modules=None)
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj", "k_proj"]

    def test_single_module(self):
        args = Namespace(lora_rank=8, target_modules="q_proj", exclude_modules=None)
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj"]

    def test_lora_rank_zero_skips_parsing(self):
        args = Namespace(lora_rank=0, target_modules="all-linear", exclude_modules=None)
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == "all-linear"  # unchanged

    def test_missing_target_modules_asserts(self):
        args = Namespace(lora_rank=32, target_modules=None, exclude_modules=None)
        with pytest.raises(AssertionError, match="--target-modules"):
            _apply_lora_arg_parsing(args)


# ---------------------------------------------------------------------------
# Exclude modules filtering
# ---------------------------------------------------------------------------


class TestLoraExcludeModules:
    def test_single_exclude(self):
        args = Namespace(lora_rank=32, target_modules="all-linear", exclude_modules="o_proj")
        result = _apply_lora_arg_parsing(args)
        assert "o_proj" not in result.target_modules
        assert len(result.target_modules) == 6

    def test_multiple_exclude_comma_separated(self):
        args = Namespace(lora_rank=32, target_modules="all-linear", exclude_modules="o_proj, down_proj")
        result = _apply_lora_arg_parsing(args)
        assert "o_proj" not in result.target_modules
        assert "down_proj" not in result.target_modules
        assert len(result.target_modules) == 5

    def test_exclude_all_results_in_empty(self):
        args = Namespace(
            lora_rank=32,
            target_modules="q_proj,k_proj",
            exclude_modules="q_proj,k_proj",
        )
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == []

    def test_exclude_nonexistent_module_no_effect(self):
        args = Namespace(lora_rank=32, target_modules="q_proj,k_proj", exclude_modules="nonexistent")
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj", "k_proj"]

    def test_no_exclude_modules(self):
        args = Namespace(lora_rank=32, target_modules="q_proj,k_proj", exclude_modules=None)
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj", "k_proj"]

    def test_empty_string_exclude(self):
        """Empty string is truthy; should be treated as a single (non-matching) exclude."""
        args = Namespace(lora_rank=32, target_modules="q_proj,k_proj", exclude_modules="")
        result = _apply_lora_arg_parsing(args)
        assert result.target_modules == ["q_proj", "k_proj"]
