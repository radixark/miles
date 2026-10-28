from types import SimpleNamespace

import pytest

from miles.utils.lora.utils import lora_colocated_engine_keeps_base, lora_sync_sends_base


def _args(**overrides):
    args = dict(
        lora_rank=8,
        colocate=True,
        offload_rollout=True,
        lora_base_cpu_backup=False,
        check_weight_update_equal=False,
        lora_train_only=False,
    )
    return SimpleNamespace(**(args | overrides))


@pytest.mark.parametrize(
    "overrides,keeps_base,sends_base",
    [
        ({}, False, True),
        ({"lora_base_cpu_backup": True}, True, False),
        ({"offload_rollout": False}, True, False),
        ({"lora_base_cpu_backup": True, "check_weight_update_equal": True}, True, True),
        ({"lora_base_cpu_backup": True, "lora_train_only": True}, True, True),
    ],
    ids=["released-without-backup", "host-backup", "never-released", "weight-checker", "train-only"],
)
def test_the_sync_sends_the_base_exactly_when_the_engines_lose_it(overrides, keeps_base, sends_base):
    """The trainer's host backup and the weight sync both read this decision; they must agree."""
    args = _args(**overrides)
    assert lora_colocated_engine_keeps_base(args) is keeps_base
    assert lora_sync_sends_base(args) is sends_base
