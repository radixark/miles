"""What the rollout engines get from a LoRA run: the served adapter, its merge into the base, or the frozen base."""

from argparse import Namespace

import pytest

from miles.utils.lora.arguments import validate_lora_args
from miles.utils.lora.utils import lora_merge_for_rollout, lora_rollout_enabled


@pytest.mark.parametrize(
    "flags, served, merged",
    [
        (dict(lora_rank=0), False, False),
        (dict(lora_rank=0, lora_merge_for_rollout=True), False, False),
        (dict(lora_rank=8), True, False),
        (dict(lora_rank=8, lora_train_only=True), False, False),
        (dict(lora_rank=8, lora_merge_for_rollout=True), False, True),
    ],
    ids=["full-model", "full-model-ignores-merge", "served", "train-only", "merged"],
)
def test_rollout_modes(flags, served, merged):
    args = Namespace(**{"lora_adapter_path": None, "lora_train_only": False, "lora_merge_for_rollout": False, **flags})
    assert lora_rollout_enabled(args) is served
    assert lora_merge_for_rollout(args) is merged


def _merge_args(**overrides):
    defaults = dict(
        multi_lora_n_adapters=0,
        train_backend="torchtitan",
        lora_rank=8,
        lora_adapter_path=None,
        lora_merge_for_rollout=True,
        lora_train_only=False,
        check_lora_weight_equal=False,
    )
    return Namespace(**(defaults | overrides))


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(train_backend="megatron"), "--train-backend torchtitan"),
        (dict(lora_train_only=True), "--lora-train-only"),
        (dict(check_lora_weight_equal=True), "--check-weight-update-equal"),
    ],
)
def test_merge_for_rollout_rejects_conflicting_modes(override, message):
    with pytest.raises(AssertionError, match=message):
        validate_lora_args(_merge_args(**override))
