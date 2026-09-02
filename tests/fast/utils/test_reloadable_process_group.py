from types import SimpleNamespace
from unittest.mock import sentinel

import pytest

from miles.utils import reloadable_process_group
from miles.utils.reloadable_process_group import ReloadableProcessGroup


class TestReloadProcessGroups:
    def test_reload_synchronizes_all_ranks_before_returning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fast rank must not enter a collective while peers are still recreating process groups."""
        pid = 42
        events: list[tuple[str, object | None]] = []
        first_group = SimpleNamespace(
            group=None,
            inner_args=([0, 1],),
            inner_kwargs={"backend": "nccl"},
        )
        second_group = SimpleNamespace(
            group=None,
            inner_args=([0, 1],),
            inner_kwargs={"backend": "nccl"},
        )
        created_groups = iter((sentinel.first_group, sentinel.second_group))

        def create_group(*args, **kwargs):
            events.append(("new_group", (args, kwargs)))
            return next(created_groups)

        monkeypatch.setattr(reloadable_process_group.os, "getpid", lambda: pid)
        monkeypatch.setitem(ReloadableProcessGroup.GROUPS, pid, [first_group, second_group])
        monkeypatch.setitem(reloadable_process_group.old_new_group_dict, pid, create_group)
        monkeypatch.setattr(reloadable_process_group.torch.cuda, "synchronize", lambda: events.append(("cuda", None)))
        monkeypatch.setattr(
            reloadable_process_group,
            "get_gloo_group",
            lambda: sentinel.gloo_group,
            raising=False,
        )
        monkeypatch.setattr(
            reloadable_process_group.dist,
            "barrier",
            lambda *, group: events.append(("barrier", group)),
        )

        ReloadableProcessGroup.reload_process_groups()

        assert first_group.group is sentinel.first_group
        assert second_group.group is sentinel.second_group
        assert [event[0] for event in events] == ["new_group", "new_group", "cuda", "barrier"]
        assert events[-1] == ("barrier", sentinel.gloo_group)
