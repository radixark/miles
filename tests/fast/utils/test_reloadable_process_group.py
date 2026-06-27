from datetime import timedelta
from types import SimpleNamespace

from miles.utils import reloadable_process_group as rpg


class _FakeP2POp:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass


class _FakeGroup:
    pass


def _noop(*args, **kwargs):
    return None


def _fake_dist(new_group, destroyed):
    def get_rank(group=None):
        return 0

    def get_world_size(group=None):
        return 2

    return SimpleNamespace(
        new_group=new_group,
        get_rank=get_rank,
        get_world_size=get_world_size,
        get_backend=_noop,
        get_global_rank=_noop,
        get_group_rank=_noop,
        get_process_group_ranks=_noop,
        all_reduce=_noop,
        all_gather=_noop,
        all_gather_object=_noop,
        all_gather_into_tensor=_noop,
        all_to_all=_noop,
        all_to_all_single=_noop,
        broadcast=_noop,
        broadcast_object_list=_noop,
        reduce=_noop,
        reduce_scatter=_noop,
        reduce_scatter_tensor=_noop,
        scatter=_noop,
        gather=_noop,
        barrier=_noop,
        send=_noop,
        recv=_noop,
        _coalescing_manager=_noop,
        isend=_noop,
        irecv=_noop,
        P2POp=_FakeP2POp,
        destroy_process_group=lambda group: destroyed.append(group),
    )


def test_reload_process_groups_preserves_new_group_options(monkeypatch):
    pid = 12345
    calls = []
    destroyed = []
    timeout = timedelta(minutes=120)
    pg_options = object()

    def fake_new_group(*args, **kwargs):
        group = _FakeGroup()
        calls.append((args, dict(kwargs), group))
        return group

    monkeypatch.setattr(rpg, "dist", _fake_dist(fake_new_group, destroyed))
    monkeypatch.setattr(rpg, "old_new_group_dict", {})
    monkeypatch.setattr(rpg.ReloadableProcessGroup, "GROUPS", {})
    monkeypatch.setattr(rpg.os, "getpid", lambda: pid)

    rpg.monkey_patch_torch_dist()

    group = rpg.dist.new_group([0, 1], backend="nccl", timeout=timeout, pg_options=pg_options)
    first_inner_group = calls[-1][2]
    assert group.group is first_inner_group

    rpg.ReloadableProcessGroup.destroy_process_groups()
    assert group.group is None
    assert destroyed == [first_inner_group]

    rpg.ReloadableProcessGroup.reload_process_groups()
    assert group.group is calls[-1][2]
    assert calls[-1][0] == ([0, 1],)
    assert calls[-1][1]["backend"] == "nccl"
    assert calls[-1][1]["timeout"] is timeout
    assert calls[-1][1]["pg_options"] is pg_options
