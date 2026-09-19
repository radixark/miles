"""CPU-only tests: no tokenizer, inference server, reward backend, or GPU required."""

import pytest
import torch

from miles.backends.megatron_utils.model import _ZeroGradientWatchdog


@pytest.fixture
def watchdog():
    # A private instance, never the module singleton: the singleton outlives a
    # single train() call and a leaked streak would hide a missing reset.
    return _ZeroGradientWatchdog()


def observe(watchdog, limit, norm, role="actor", rollout_id=4, step_id=0):
    watchdog.observe(limit, role, norm, rollout_id, step_id)


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_disables_watchdog_and_keeps_no_state(watchdog, limit):
    # The flag is validated upstream; a non-positive value reaching here must not
    # be read as "trip immediately".
    for step_id in range(8):
        observe(watchdog, limit, 0.0, step_id=step_id)
    assert watchdog.streaks == {}


@pytest.mark.parametrize("limit", [1, 2, 3, 5])
def test_only_the_limit_th_consecutive_zero_raises(watchdog, limit):
    for step_id in range(limit - 1):
        observe(watchdog, limit, 0.0, step_id=step_id)
    with pytest.raises(RuntimeError) as exc:
        observe(watchdog, limit, 0.0, step_id=limit - 1)
    assert f"{limit} consecutive optimizer steps with a zero gradient norm" in str(exc.value)


def test_error_message_names_the_flag_the_limit_and_the_failing_step(watchdog):
    with pytest.raises(RuntimeError) as exc:
        observe(watchdog, 1, 0.0, role="critic", rollout_id=9, step_id=3)
    message = str(exc.value)
    assert message.startswith("critic: 1 consecutive optimizer steps with a zero gradient norm")
    assert "--max-consecutive-zero-grad-steps=1" in message
    assert "last was rollout 9 step 3" in message


def test_nonzero_norm_in_the_middle_of_a_streak_restarts_it(watchdog):
    observe(watchdog, 3, 0.0)
    observe(watchdog, 3, 0.0)
    observe(watchdog, 3, 0.25)  # one healthy backward is enough to clear the streak
    observe(watchdog, 3, 0.0)
    observe(watchdog, 3, 0.0)
    with pytest.raises(RuntimeError, match="3 consecutive"):
        observe(watchdog, 3, 0.0)


@pytest.mark.parametrize("norm", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_norm_neither_increments_nor_resets_the_streak(watchdog, norm):
    observe(watchdog, 2, 0.0)
    observe(watchdog, 2, norm)  # inf/nan is a different failure, handled on the optimizer path
    with pytest.raises(RuntimeError, match="2 consecutive"):
        observe(watchdog, 2, 0.0)


def test_unmeasured_norm_is_ignored_rather_than_counted_as_zero(watchdog):
    # Megatron returns None instead of a norm when clip_grad <= 0. Treating that as
    # a zero gradient would kill a healthy run; --clip-grad > 0 is enforced upstream,
    # but the watchdog must not crash or miscount if a None ever reaches it.
    observe(watchdog, 2, 0.0)
    observe(watchdog, 2, None)
    with pytest.raises(RuntimeError, match="2 consecutive"):
        observe(watchdog, 2, 0.0)


@pytest.mark.parametrize("wrap", [float, torch.tensor], ids=["float", "tensor"])
def test_tensor_and_float_norms_behave_identically(watchdog, wrap):
    observe(watchdog, 2, wrap(1.5))
    observe(watchdog, 2, wrap(0.0))
    observe(watchdog, 2, wrap(2.0))
    observe(watchdog, 2, wrap(float("nan")))
    observe(watchdog, 2, wrap(0.0))
    with pytest.raises(RuntimeError, match="2 consecutive"):
        observe(watchdog, 2, wrap(0.0))


def test_roles_keep_independent_streaks(watchdog):
    # A dead actor backward must not be attributed to a healthy critic and vice versa.
    observe(watchdog, 2, 0.0, role="actor")
    observe(watchdog, 2, 0.0, role="critic")
    observe(watchdog, 2, 1.0, role="critic")
    with pytest.raises(RuntimeError, match="^actor: 2 consecutive"):
        observe(watchdog, 2, 0.0, role="actor")


def test_streak_is_cleared_after_raising(watchdog):
    # The singleton survives the exception; a caller that catches it must not be
    # re-killed by the very next zero step.
    observe(watchdog, 2, 0.0)
    with pytest.raises(RuntimeError):
        observe(watchdog, 2, 0.0)
    observe(watchdog, 2, 0.0)
    with pytest.raises(RuntimeError, match="2 consecutive"):
        observe(watchdog, 2, 0.0)


def test_every_zero_step_warns_with_the_running_count(watchdog, caplog):
    with caplog.at_level("WARNING"):
        observe(watchdog, 3, 0.0, rollout_id=6, step_id=1)
        observe(watchdog, 3, 0.0, rollout_id=6, step_id=2)
    warnings = [record.message for record in caplog.records if "zero gradient norm" in record.message]
    assert warnings == [
        "actor rollout 6 step 1 produced a zero gradient norm (1 consecutive, limit 3)",
        "actor rollout 6 step 2 produced a zero gradient norm (2 consecutive, limit 3)",
    ]
