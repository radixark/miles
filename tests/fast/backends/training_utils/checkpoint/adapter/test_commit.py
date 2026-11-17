"""Commits are atomic link swaps, ownership is enforced, and recovery removes only unreferenced entries."""

import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint.adapter import commit as commit_module
from miles.backends.training_utils.checkpoint.adapter.commit import (
    TENANT_FILENAME,
    OwnershipError,
    commit_adapter_dir,
    delete_adapter_dir,
    delete_retired_adapter_dirs,
    delete_unreferenced_adapter_dirs,
    resolve_adapter_dir,
    tenant_digest,
    write_adapter_dir,
)


def _save(root, name, payload, *, tenant="alice", overwrite=True):
    with write_adapter_dir(root, name, tenant_digest=None if tenant is None else tenant_digest(tenant)) as written:
        (written.path / "payload").write_text(payload)
    return commit_adapter_dir(root, name, written, overwrite=overwrite)


def test_overwrite_points_the_name_at_the_new_commit_and_removes_the_old(tmp_path):
    first = _save(tmp_path, "w", "1")
    second = _save(tmp_path, "w", "2")
    resolved = resolve_adapter_dir(tmp_path, "w", tenant="alice")
    assert resolved == second and (resolved.path / "payload").read_text() == "2"
    assert not first.path.exists() and first.path.with_name(first.path.name + ".deleting").is_dir()
    delete_retired_adapter_dirs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["w", second.path.name])
    assert not os.path.isabs(os.readlink(tmp_path / "w")), "links must be relative so copies keep working"


def test_a_commit_leaves_retired_dirs_to_the_cleanup_task(tmp_path):
    first = _save(tmp_path, "w", "1")
    _save(tmp_path, "w", "2")
    _save(tmp_path, "w", "3")
    assert first.path.with_name(first.path.name + ".deleting").is_dir(), "a cleanup task may be removing it"
    delete_unreferenced_adapter_dirs(tmp_path)
    assert not any(p.name.endswith(".deleting") for p in tmp_path.iterdir())


def test_refuses_to_overwrite_when_asked_not_to(tmp_path):
    first = _save(tmp_path, "w", "1")
    with pytest.raises(FileExistsError):
        _save(tmp_path, "w", "2", overwrite=False)
    assert (resolve_adapter_dir(tmp_path, "w", tenant=None).path / "payload").read_text() == "1"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["w", first.path.name]), "the refused write is removed"


@pytest.mark.parametrize("overwrite", [False, True])
def test_refuses_to_overwrite_a_regular_file(tmp_path, overwrite):
    (tmp_path / "w").write_text("user file")
    with pytest.raises(FileExistsError):
        _save(tmp_path, "w", "2", overwrite=overwrite)
    assert (tmp_path / "w").read_text() == "user file"


def test_another_tenant_cannot_resolve_or_delete(tmp_path):
    _save(tmp_path, "w", "1")
    with pytest.raises(OwnershipError):
        resolve_adapter_dir(tmp_path, "w", tenant="mallory")
    with pytest.raises(OwnershipError):
        delete_adapter_dir(tmp_path, "w", tenant="mallory")
    delete_adapter_dir(tmp_path, "w", tenant="alice")
    assert resolve_adapter_dir(tmp_path, "w", tenant=None) is None
    _save(tmp_path, "untenanted", "1", tenant=None)
    with pytest.raises(OwnershipError):
        resolve_adapter_dir(tmp_path, "untenanted", tenant="alice")


def test_a_name_that_is_not_one_visible_path_component_is_refused(tmp_path):
    for name, refusal in ((".w", "must not start with"), ("x/../y", "one path component")):
        with pytest.raises(AssertionError, match=refusal):
            with write_adapter_dir(tmp_path, name):
                pass
    with pytest.raises(AssertionError, match="one path component"):
        resolve_adapter_dir(tmp_path, "x/../w", tenant=None)


def test_a_failed_write_is_never_visible_and_leaves_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="adapter checkpoint failed"):
        with write_adapter_dir(tmp_path, "w") as written:
            (written.path / "partial").write_text("x")
            raise OSError("disk full")
    assert resolve_adapter_dir(tmp_path, "w", tenant=None) is None
    assert list(tmp_path.iterdir()) == [], "a retry under a new name must not leave the failed write behind"


def test_a_commit_failing_before_the_link_swap_removes_what_it_wrote(tmp_path, monkeypatch):
    first = _save(tmp_path, "w", "1")

    def fail(src, dst):
        raise OSError("link swap failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="link swap failed"):
        _save(tmp_path, "w", "2")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["w", first.path.name]), "no orphan dir or link"


def test_a_failed_prepare_leaves_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="adapter checkpoint failed"):
        # a digest json cannot encode fails the tenant.json write after the mkdir
        with write_adapter_dir(tmp_path, "w", tenant_digest=object()):
            pass
    assert list(tmp_path.iterdir()) == []


def test_recovery_of_one_name_does_not_touch_a_dotted_neighbour(tmp_path):
    kept = _save(tmp_path, "foo.bar", "neighbour")
    with write_adapter_dir(tmp_path, "foo") as orphan:
        pass
    (tmp_path / "dangling").symlink_to(".dangling.0011223344556677")
    delete_unreferenced_adapter_dirs(tmp_path, "foo")
    assert not orphan.path.exists()
    assert kept.path.exists() and resolve_adapter_dir(tmp_path, "foo.bar", tenant=None) == kept
    assert (tmp_path / "dangling").is_symlink(), "recovery scoped to 'foo' must leave other names alone"
    delete_unreferenced_adapter_dirs(tmp_path)
    assert not (tmp_path / "dangling").is_symlink()


def test_recovery_keeps_a_dangling_foreign_link(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "y").symlink_to(tmp_path / "unmounted" / "adapter")
    (root / "z").symlink_to(".other.0011223344556677")
    delete_unreferenced_adapter_dirs(root)
    assert sorted(p.name for p in root.iterdir()) == ["y", "z"]
    with pytest.raises(FileExistsError):
        _save(root, "y", "new", overwrite=False)
    assert os.readlink(root / "y") == str(tmp_path / "unmounted" / "adapter")


def test_recovery_removes_a_crashed_commits_staging_link_and_leaves_foreign_hidden_dirs(tmp_path):
    with write_adapter_dir(tmp_path, "w") as crashed:
        pass
    # crash between creating the staging link and renaming it onto the name
    (tmp_path / f"{crashed.path.name}.link").symlink_to(crashed.path.name)
    (tmp_path / ".git").mkdir()
    delete_unreferenced_adapter_dirs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".git"]


def test_an_overwrite_replaces_a_checkpoint_saved_before_commits(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "payload").write_text("legacy")
    with pytest.raises(FileExistsError):
        _save(tmp_path, "w", "new", overwrite=False)
    assert (tmp_path / "w" / "payload").read_text() == "legacy", "a refused save leaves the old checkpoint"
    _save(tmp_path, "w", "new", overwrite=True)
    assert (resolve_adapter_dir(tmp_path, "w", tenant="alice").path / "payload").read_text() == "new"
    delete_unreferenced_adapter_dirs(tmp_path)
    assert len(list(tmp_path.iterdir())) == 2, "the name and its one commit; the adopted old dir is gone"


def test_a_failed_adoption_keeps_the_checkpoint_saved_before_commits(tmp_path, monkeypatch):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "payload").write_text("legacy")
    real_replace = os.replace

    def fail_once(src, dst):
        monkeypatch.setattr(os, "replace", real_replace)
        raise OSError("link swap failed")

    monkeypatch.setattr(os, "replace", fail_once)
    with pytest.raises(OSError, match="link swap failed"):
        _save(tmp_path, "w", "new")
    delete_unreferenced_adapter_dirs(tmp_path, "w")
    assert (resolve_adapter_dir(tmp_path, "w", tenant=None).path / "payload").read_text() == "legacy"


@pytest.mark.parametrize("overwrite", [True, False])
def test_a_retried_commit_of_the_same_write_keeps_it(tmp_path, monkeypatch, overwrite):
    first = _save(tmp_path, "w", "1")
    with write_adapter_dir(tmp_path, "w") as written:
        (written.path / "payload").write_text("2")
    real_fsync = commit_module._fsync

    def fail_once(path):
        monkeypatch.setattr(commit_module, "_fsync", real_fsync)
        raise OSError("fsync failed after the link swap")

    monkeypatch.setattr(commit_module, "_fsync", fail_once)
    with pytest.raises(OSError, match="after the link swap"):
        commit_adapter_dir(tmp_path, "w", written, overwrite=True)
    assert commit_adapter_dir(tmp_path, "w", written, overwrite=overwrite).path == written.path
    assert (resolve_adapter_dir(tmp_path, "w", tenant=None).path / "payload").read_text() == "2"
    assert not first.path.exists()


def test_a_link_to_a_foreign_dir_is_never_replaced_or_deleted(tmp_path):
    shared = tmp_path / "shared" / "pretrained_adapter"
    shared.mkdir(parents=True)
    (shared / "payload").write_text("shared")
    root = tmp_path / "root"
    root.mkdir()
    (root / "w").symlink_to(shared)
    with pytest.raises(ValueError, match="did not write"):
        _save(root, "w", "new", overwrite=True)
    with pytest.raises(ValueError, match="did not write"):
        delete_adapter_dir(root, "w", tenant=None)
    assert (shared / "payload").read_text() == "shared"
    assert os.readlink(root / "w") == str(shared)
    assert [p.name for p in root.iterdir()] == ["w"], "the refused write is removed"


@pytest.mark.parametrize("target", ["pretrained_adapter", "my.adapter.v1"])
def test_resolve_refuses_a_link_to_a_foreign_dir(tmp_path, target):
    (tmp_path / target).mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "w").symlink_to(tmp_path / target)
    with pytest.raises(ValueError, match="did not write"):
        resolve_adapter_dir(root, "w", tenant=None)


def test_the_written_files_and_dir_are_synced_before_the_link_swap(tmp_path, monkeypatch):
    events, path_by_fd = [], {}
    real_open, real_fsync, real_replace = os.open, os.fsync, os.replace

    def recording_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        path_by_fd[fd] = Path(path)
        return fd

    def recording_fsync(fd):
        events.append(("fsync", path_by_fd[fd]))
        real_fsync(fd)

    def recording_replace(src, dst):
        events.append(("replace", Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    metadata = (TENANT_FILENAME, "adapter_parametrization.json", "training_state.json")
    with write_adapter_dir(tmp_path, "w", tenant_digest=tenant_digest("alice")) as written:
        for filename in metadata[1:]:
            (written.path / filename).write_text("{}")
    commit_adapter_dir(tmp_path, "w", written, overwrite=False)
    synced = {path for op, path in events[: events.index(("replace", tmp_path / "w"))] if op == "fsync"}
    assert {written.path, *(written.path / f for f in metadata)} <= synced


def test_one_committer_per_root_a_commit_removes_another_in_flight_write_of_the_name(tmp_path):
    """Only the root's single committer writes and commits; a second writer of the same name in flight is
    unreferenced to recovery, so the commit removes it and that write fails instead of committing."""
    with pytest.raises(RuntimeError, match="No such file or directory"):
        with write_adapter_dir(tmp_path, "w") as in_flight:
            (in_flight.path / "payload").write_text("in flight")
            committed = _save(tmp_path, "w", "committed")
            assert not in_flight.path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["w", committed.path.name])
    assert (resolve_adapter_dir(tmp_path, "w", tenant="alice").path / "payload").read_text() == "committed"
