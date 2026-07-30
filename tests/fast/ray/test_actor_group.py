import asyncio
from types import SimpleNamespace

import pytest

from miles.ray import actor_group


class _FakeTrainGroup(actor_group.RayTrainGroup):
    def __init__(self, results: list[bool]) -> None:
        self.args = SimpleNamespace(async_save=True)
        self.results = results
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def _broadcast(self, method_name: str, *args: object, **kwargs: object) -> list[bool]:
        self.calls.append((method_name, args, kwargs))
        return self.results


def test_finalize_async_save_fans_out_to_every_actor() -> None:
    group = _FakeTrainGroup([True, True, True])

    assert asyncio.run(group.finalize_async_save(blocking=False)) is True
    assert group.calls == [("finalize_async_save", (), {"blocking": False})]


def test_finalize_async_save_rejects_rank_disagreement() -> None:
    group = _FakeTrainGroup([True, False])

    with pytest.raises(RuntimeError, match="completion disagreed across ranks"):
        asyncio.run(group.finalize_async_save(blocking=True))


def test_finalize_async_save_skips_non_async_backend() -> None:
    group = _FakeTrainGroup([True])
    group.args.async_save = False

    assert asyncio.run(group.finalize_async_save(blocking=True)) is True
    assert group.calls == []
