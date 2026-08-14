import pytest
from tests.e2e.ft import test_random_crash_fully_async__kill_train_rollout__dp2_cp2 as fully_async_random_entry
from tests.e2e.ft.conftest_ft.execution import get_fully_async_args, get_train_script
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_mode_supports_fully_async


def test_the_fully_async_soak_launches_the_async_trainer() -> None:
    """A fully-async run is train_async.py plus the flag; either half alone is a sync run."""
    assert get_train_script(fully_async=True) == "train_async.py"
    assert "--fully-async " in get_fully_async_args(fully_async=True)


def test_a_sync_soak_is_left_exactly_as_it_was() -> None:
    """The existing soaks must keep launching train.py with no extra generation flags."""
    assert get_train_script(fully_async=False) == "train.py"
    assert get_fully_async_args(fully_async=False) == ""


def test_the_fully_async_soak_pauses_generation_in_place() -> None:
    """The default retract mode can deadlock flush_cache under load, which reads as the hang the soak hunts."""
    assert "--pause-generation-mode in_place " in get_fully_async_args(fully_async=True)


def test_the_fully_async_random_entry_generates_with_real_engines() -> None:
    """Regression guard: a mode change must not turn this entry into a debug-rollout-data run."""
    assert MODES[fully_async_random_entry._MODE].has_real_rollout


def test_the_fully_async_random_entry_crashes_engines_as_well_as_trainers() -> None:
    """Killing an engine that keeps generating across weight updates is the fault only fully-async can produce."""
    assert MODES[fully_async_random_entry._MODE].ft_components == ("train", "rollout")


def test_a_debug_rollout_mode_cannot_be_run_fully_async() -> None:
    """Replaying recorded rollouts would prove nothing about generating while training."""
    with pytest.raises(AssertionError, match="no rollout engines"):
        assert_mode_supports_fully_async(
            MODES["kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer"],
            mode="kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer",
        )


def test_a_colocated_mode_cannot_be_run_fully_async() -> None:
    """train_async.py rejects colocation, so the soak must fail before it burns a cluster."""
    with pytest.raises(AssertionError, match="colocated"):
        assert_mode_supports_fully_async(
            MODES["kill_rollout__dp2_cp2__colocate"], mode="kill_rollout__dp2_cp2__colocate"
        )
