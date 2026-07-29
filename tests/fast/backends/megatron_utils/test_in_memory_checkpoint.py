from unittest.mock import patch

import pytest

from miles.backends.megatron_utils.ft.in_memory_checkpoint import InMemoryCheckpointManager


@pytest.fixture()
def manager():
    with patch("miles.backends.megatron_utils.ft.in_memory_checkpoint.get_args") as mock_get_args:
        mock_args = mock_get_args.return_value
        mock_args.non_persistent_ckpt_type = "local"
        mock_args.non_persistent_local_ckpt_algo = "fully_parallel"
        yield InMemoryCheckpointManager()


class TestInMemoryCheckpointManager:
    def test_find_latest_returns_minus_one_initially(self, manager: InMemoryCheckpointManager):
        assert manager.find_latest() == -1

    def test_load_before_save_raises(self, manager: InMemoryCheckpointManager):
        with pytest.raises(AssertionError, match="No in-memory checkpoint"):
            manager.load()

    def test_save_then_load_returns_same_object(self, manager: InMemoryCheckpointManager):
        sentinel = object()
        manager.save(state_dict=sentinel, iteration=5)

        assert manager.find_latest() == 5

        result, name = manager.load()
        assert result is sentinel
        assert "5" in name

    def test_load_is_idempotent_returns_same_state(self, manager: InMemoryCheckpointManager):
        sentinel = object()
        manager.save(state_dict=sentinel, iteration=1)

        first, _ = manager.load()
        second, _ = manager.load()
        assert first is sentinel
        assert second is sentinel

    def test_save_twice_without_reset_raises(self, manager: InMemoryCheckpointManager):
        manager.save(state_dict=object(), iteration=1)

        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=2)

    def test_save_after_load_still_raises_load_does_not_reset(self, manager: InMemoryCheckpointManager):
        """`load()` is idempotent and does NOT clear state, so a second `save()`
        on the same manager — even after `load()` — must still raise."""
        manager.save(state_dict=object(), iteration=1)
        manager.load()

        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=2)

    def test_async_save_raises(self, manager: InMemoryCheckpointManager):
        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=1, is_async=True)
