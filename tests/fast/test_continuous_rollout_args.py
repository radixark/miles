from types import SimpleNamespace

import pytest

from miles.utils.arguments import validate_continuous_rollout


def _args(**changes):
    return SimpleNamespace(**(dict(
        continuous_rollout=True, train_backend="megatron", fully_async=False,
        num_rollout=1, lr_decay_style="constant", weight_decay_incr_style="constant",
        lr_warmup_fraction=None, debug_exit_after_rollout=None,
        debug_train_only=False, debug_rollout_only=False,
    ) | changes))


def test_disabled_continuous_is_compatible_with_old_namespace():
    validate_continuous_rollout(SimpleNamespace())


def test_constant_schedule_horizon_is_valid():
    validate_continuous_rollout(_args())


@pytest.mark.parametrize("changes", [
    {"train_backend": "fsdp"}, {"fully_async": True}, {"num_rollout": None},
    {"num_rollout": 0}, {"num_rollout": -1}, {"lr_decay_style": "cosine"},
    {"weight_decay_incr_style": "linear"}, {"lr_warmup_fraction": 0.1},
    {"debug_exit_after_rollout": 1}, {"debug_train_only": True}, {"debug_rollout_only": True},
])
def test_incompatible_continuous_configuration_fails(changes):
    with pytest.raises(ValueError, match="continuous-rollout"):
        validate_continuous_rollout(_args(**changes))
