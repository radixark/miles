import pytest

from miles.backends.training_utils.weight_update.protocols.utils import rank_errors
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank


class _FakeDist:
    """This rank's view of a world whose other ranks report `others`."""

    def __init__(self, others: list[str | None]) -> None:
        self._others = others

    def get_world_size(self) -> int:
        return 1 + len(self._others)

    def all_gather_object(self, output: list, value, group=None) -> None:
        output[:] = [value, *self._others]


def _world(monkeypatch: pytest.MonkeyPatch, others: list[str | None]) -> None:
    monkeypatch.setattr(rank_errors, "dist", _FakeDist(others))
    monkeypatch.setattr(rank_errors, "get_gloo_group", lambda: None)


def test_a_rank_whose_own_step_passed_still_raises_when_another_failed(monkeypatch):
    """Otherwise it walks into the next collective, which the failed rank never joins."""
    _world(monkeypatch, [None, "OSError: No space left on device"])
    with pytest.raises(RuntimeError, match="shard write failed on rank 2: OSError: No space left on device"):
        with fail_on_every_rank("shard write"):
            pass


def test_the_failed_rank_raises_with_its_own_error_as_the_cause(monkeypatch):
    _world(monkeypatch, [None])
    with pytest.raises(RuntimeError, match="failed on rank 0: ValueError: bad shard") as raised:
        with fail_on_every_rank("shard write"):
            raise ValueError("bad shard")
    assert isinstance(raised.value.__cause__, ValueError)


def test_nothing_is_raised_when_every_rank_passed(monkeypatch):
    _world(monkeypatch, [None, None])
    with fail_on_every_rank("shard write"):
        pass
