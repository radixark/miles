from datetime import timedelta

from tests.ci.ci_register import register_cpu_ci

from miles.utils import reloadable_process_group as rpg

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


def test_reload_replays_original_new_group_timeout_and_options(monkeypatch):
    """Reload must replay timeout/pg_options, not ``{ranks, backend=nccl}``.

    A reloaded group can still complete a short collective after falling back
    to PyTorch's default timeout. That hides the offload/heal watchdog bug.
    Spy on the constructor instead.
    """
    pid = 4242
    calls: list[tuple[tuple, dict, object]] = []
    timeout = timedelta(minutes=120)
    pg_options = object()

    class FakeGroup:
        pass

    def fake_new_group(*args, **kwargs):
        group = FakeGroup()
        calls.append((args, dict(kwargs), group))
        return group

    monkeypatch.setattr(rpg, "old_new_group_dict", {})
    monkeypatch.setattr(rpg.ReloadableProcessGroup, "GROUPS", {})
    monkeypatch.setattr(rpg.os, "getpid", lambda: pid)
    monkeypatch.setattr(rpg.dist, "new_group", fake_new_group)
    monkeypatch.setattr(rpg.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(rpg.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(rpg.dist, "destroy_process_group", lambda group: None)

    rpg.monkey_patch_torch_dist()
    group = rpg.dist.new_group([0, 1], backend="nccl", timeout=timeout, pg_options=pg_options)
    assert isinstance(group, rpg.ReloadableProcessGroup)
    first_inner = calls[-1][2]
    assert group.group is first_inner

    rpg.ReloadableProcessGroup.destroy_process_groups()
    assert group.group is None

    rpg.ReloadableProcessGroup.reload_process_groups()
    reload_args, reload_kwargs, reloaded_inner = calls[-1]
    assert group.group is reloaded_inner
    assert reloaded_inner is not first_inner
    assert reload_args == ([0, 1],)
    assert reload_kwargs["backend"] == "nccl"
    assert reload_kwargs["timeout"] is timeout
    assert reload_kwargs["pg_options"] is pg_options
    assert set(reload_kwargs) == {"backend", "timeout", "pg_options"}
