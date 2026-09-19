"""Tests for the baseline LoRA target-module argument parsing contract."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils.arguments import parse_lora_target_modules

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _parse(**overrides) -> Namespace:
    values = dict(
        lora_rank=32,
        target_modules="all-linear",
        exclude_modules=None,
        hf_checkpoint=None,
        megatron_to_hf_mode="raw",
        lora_provider_path=None,
    )
    values.update(overrides)
    args = Namespace(**values)
    parse_lora_target_modules(args)
    return args


@pytest.fixture(autouse=True)
def _dense_hf_config(monkeypatch):
    monkeypatch.setattr("miles.utils.arguments.load_hf_config", lambda _checkpoint: SimpleNamespace())


def test_all_linear_expands_to_dense_projection_set():
    args = _parse()
    assert args.target_modules == ALL_LINEAR
    assert args._target_modules_expanded_from_all_linear


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ("q_proj, k_proj, v_proj", ["q_proj", "k_proj", "v_proj"]),
        ("q_proj,k_proj", ["q_proj", "k_proj"]),
        ("q_proj", ["q_proj"]),
    ],
)
def test_explicit_targets_are_split(targets, expected):
    assert _parse(target_modules=targets).target_modules == expected


def test_zero_rank_skips_parsing():
    assert _parse(lora_rank=0).target_modules == "all-linear"


def test_enabled_lora_requires_targets():
    with pytest.raises(AssertionError, match="--target-modules"):
        _parse(target_modules=None)


@pytest.mark.parametrize(
    ("targets", "excludes", "expected"),
    [
        ("all-linear", "o_proj", [name for name in ALL_LINEAR if name != "o_proj"]),
        ("all-linear", "o_proj, down_proj", [name for name in ALL_LINEAR if name not in {"o_proj", "down_proj"}]),
        ("q_proj,k_proj", "q_proj,k_proj", []),
        ("q_proj,k_proj", "nonexistent", ["q_proj", "k_proj"]),
        ("q_proj,k_proj", None, ["q_proj", "k_proj"]),
        ("q_proj,k_proj", "", ["q_proj", "k_proj"]),
    ],
)
def test_excludes_are_applied_exactly(targets, excludes, expected):
    assert _parse(target_modules=targets, exclude_modules=excludes).target_modules == expected
