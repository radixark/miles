"""What the rollout engines get from a LoRA run: the served adapter, its merge into the base, or the frozen base."""

from argparse import Namespace

import pytest

from miles.utils.lora.arguments import validate_lora_args
from miles.utils.lora.utils import lora_merge_for_rollout, lora_rollout_enabled, lora_sync_reads_adapter


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
    args = Namespace(**flags)
    assert lora_rollout_enabled(args) == served
    assert lora_merge_for_rollout(args) == merged
    # the weight sync reads the adapter unless the engines keep the frozen base
    assert lora_sync_reads_adapter(args) == (served or merged)


def _merge_args(**overrides):
    defaults = dict(
        multi_lora_n_adapters=0,
        train_backend="megatron",
        lora_rank=8,
        lora_adapter_path=None,
        lora_merge_for_rollout=True,
        megatron_to_hf_mode="raw",
        lora_train_only=False,
        check_lora_weight_equal=False,
    )
    return Namespace(**(defaults | overrides))


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(megatron_to_hf_mode="bridge"), "--megatron-to-hf-mode raw"),
        (
            dict(
                multi_lora_n_adapters=2,
                colocate=False,
                ft_components=[],
                offload_train=False,
                calculate_per_token_loss=False,
            ),
            "no multi-LoRA",
        ),
        (dict(lora_train_only=True), "--lora-train-only"),
        (dict(check_lora_weight_equal=True), "--check-weight-update-equal"),
    ],
)
def test_merge_for_rollout_rejects_conflicting_modes(override, message):
    with pytest.raises(AssertionError, match=message):
        validate_lora_args(_merge_args(**override))
