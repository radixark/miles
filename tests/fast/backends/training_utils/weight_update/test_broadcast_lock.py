"""Mock-based tests for the rollout-engine lock lifecycle in
training_utils/weight_update/protocols/broadcast.py.

Lock.acquire (miles/ray/utils.py) is polled until it returns True, so a
broadcast failure that skips the release makes every later weight sync spin
forever. These tests pin lock contention, argument forwarding, and release on
the success path and both failure paths (broadcast setup and engine-side
completion).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed

_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"


class _LockState:
    """In-process stand-in for the miles.ray.utils.Lock actor."""

    def __init__(self):
        self.locked = False
        self.release_calls = 0

    def acquire(self):
        if self.locked:
            return False
        self.locked = True
        return True

    def release(self):
        # Mirrors the real actor: releasing an unheld lock is a bug.
        assert self.locked, "Lock is not acquired, cannot release."
        self.release_calls += 1
        self.locked = False


def _make_updater(lock_state: _LockState) -> UpdateWeightFromDistributed:
    updater = object.__new__(UpdateWeightFromDistributed)
    lock_handle = MagicMock()
    lock_handle.acquire.remote.side_effect = lock_state.acquire
    lock_handle.release.remote.side_effect = lock_state.release
    updater.rollout_engine_lock = lock_handle
    updater.group_name = "miles-pp_0"
    updater._model_update_groups = MagicMock()
    updater.rollout_engines = [MagicMock()]
    updater._selector = "all"
    return updater


def _passthrough_ray_get(mock_ray, fail_on=None):
    """ray.get returns its argument, or raises when handed the sentinel."""

    def _get(ref):
        if fail_on is not None and ref is fail_on:
            raise RuntimeError("engine died during broadcast")
        return ref

    mock_ray.get.side_effect = _get


def _named_tensors() -> list[tuple[str, torch.Tensor]]:
    return [("layer.weight", torch.zeros(2, 2))]


@patch(f"{_MODULE}.update_weights_from_distributed")
@patch(f"{_MODULE}.ray")
def test_success_path_releases_lock_and_clears_tensors(mock_ray, mock_update):
    lock_state = _LockState()
    _passthrough_ray_get(mock_ray)
    mock_update.return_value = [MagicMock()]
    updater = _make_updater(lock_state)
    tensors = _named_tensors()

    updater.send_bucket(tensors)

    assert lock_state.locked is False
    assert lock_state.release_calls == 1
    assert tensors == []
    call_args = mock_update.call_args.args
    assert call_args[:3] == (updater.group_name, updater._model_update_groups, updater.rollout_engines)
    assert call_args[3] is tensors
    assert mock_update.call_args.kwargs == {"selector": "all"}


@patch(f"{_MODULE}.time.sleep")
@patch(f"{_MODULE}.update_weights_from_distributed")
@patch(f"{_MODULE}.ray")
def test_lock_contention_is_polled_until_acquired(mock_ray, mock_update, mock_sleep):
    lock_state = _LockState()
    lock_state.locked = True
    _passthrough_ray_get(mock_ray)
    mock_update.return_value = [MagicMock()]
    updater = _make_updater(lock_state)

    mock_sleep.side_effect = lambda _seconds: setattr(lock_state, "locked", False)

    updater.send_bucket(_named_tensors())

    mock_sleep.assert_called_once_with(0.1)
    assert lock_state.locked is False
    assert lock_state.release_calls == 1


@patch(f"{_MODULE}.update_weights_from_distributed")
@patch(f"{_MODULE}.ray")
def test_broadcast_failure_releases_lock_and_propagates(mock_ray, mock_update):
    lock_state = _LockState()
    _passthrough_ray_get(mock_ray)
    mock_update.side_effect = RuntimeError("NCCL broadcast failed")
    updater = _make_updater(lock_state)
    tensors = _named_tensors()

    with pytest.raises(RuntimeError, match="NCCL broadcast failed"):
        updater.send_bucket(tensors)

    assert lock_state.locked is False
    assert lock_state.release_calls == 1
    assert len(tensors) == 1  # nothing was cleared on the failure path


@patch(f"{_MODULE}.update_weights_from_distributed")
@patch(f"{_MODULE}.ray")
def test_engine_failure_on_refs_releases_lock_and_propagates(mock_ray, mock_update):
    lock_state = _LockState()
    failing_refs = [MagicMock()]
    _passthrough_ray_get(mock_ray, fail_on=failing_refs)
    mock_update.return_value = failing_refs
    updater = _make_updater(lock_state)

    with pytest.raises(RuntimeError, match="engine died during broadcast"):
        updater.send_bucket(_named_tensors())

    assert lock_state.locked is False
    assert lock_state.release_calls == 1


@patch(f"{_MODULE}.update_weights_from_distributed")
@patch(f"{_MODULE}.ray")
def test_weight_sync_succeeds_after_a_failed_one(mock_ray, mock_update):
    lock_state = _LockState()
    _passthrough_ray_get(mock_ray)
    mock_update.side_effect = RuntimeError("NCCL broadcast failed")
    updater = _make_updater(lock_state)

    with pytest.raises(RuntimeError):
        updater.send_bucket(_named_tensors())

    # Guard before retrying: with a leaked lock the retry below would poll
    # acquire() forever instead of failing, so assert the release explicitly.
    assert lock_state.locked is False

    mock_update.side_effect = None
    mock_update.return_value = [MagicMock()]
    tensors = _named_tensors()
    updater.send_bucket(tensors)

    assert tensors == []
    assert lock_state.locked is False
    assert lock_state.release_calls == 2
