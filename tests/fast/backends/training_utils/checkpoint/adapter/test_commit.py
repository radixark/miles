"""Commits are atomic link swaps, ownership is enforced, and recovery removes only unreferenced entries."""

import os

import pytest

from miles.backends.training_utils.checkpoint.adapter.commit import (
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
    _save(tmp_path, "w", "1")
    with pytest.raises(FileExistsError):
        _save(tmp_path, "w", "2", overwrite=False)
    assert (resolve_adapter_dir(tmp_path, "w", tenant=None).path / "payload").read_text() == "1"


def test_another_tenant_cannot_resolve_or_delete(tmp_path):
    _save(tmp_path, "w", "1")
    with pytest.raises(OwnershipError):
        resolve_adapter_dir(tmp_path, "w", tenant="mallory")
    with pytest.raises(OwnershipError):
        delete_adapter_dir(tmp_path, "w", tenant="mallory")
    delete_adapter_dir(tmp_path, "w", tenant="alice")
    assert resolve_adapter_dir(tmp_path, "w", tenant=None) is None


def test_a_failed_write_is_never_visible_and_leaves_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="adapter checkpoint failed"):
        with write_adapter_dir(tmp_path, "w") as written:
            (written.path / "partial").write_text("x")
            raise OSError("disk full")
    assert resolve_adapter_dir(tmp_path, "w", tenant=None) is None
    assert list(tmp_path.iterdir()) == [], "a retry under a new name must not leave the failed write behind"


def test_recovery_of_one_name_does_not_touch_a_dotted_neighbour(tmp_path):
    kept = _save(tmp_path, "foo.bar", "neighbour")
    with write_adapter_dir(tmp_path, "foo") as orphan:
        pass
    (tmp_path / "dangling").symlink_to(".gone.0011223344556677")
    delete_unreferenced_adapter_dirs(tmp_path, "foo")
    assert not orphan.path.exists()
    assert kept.path.exists() and resolve_adapter_dir(tmp_path, "foo.bar", tenant=None) == kept
    assert (tmp_path / "dangling").is_symlink(), "recovery scoped to 'foo' must leave other names alone"
    delete_unreferenced_adapter_dirs(tmp_path)
    assert not (tmp_path / "dangling").is_symlink()


def test_user_names_cannot_start_with_a_dot(tmp_path):
    with pytest.raises(AssertionError, match="must not start with"):
        with write_adapter_dir(tmp_path, ".w"):
            pass


def test_recovery_removes_a_crashed_commits_staging_link_and_leaves_foreign_hidden_dirs(tmp_path):
    with write_adapter_dir(tmp_path, "w") as crashed:
        pass
    # crash between creating the staging link and renaming it onto the name
    (tmp_path / f"{crashed.path.name}.link").symlink_to(crashed.path.name)
    (tmp_path / ".git").mkdir()
    delete_unreferenced_adapter_dirs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".git"]


def test_user_names_must_be_one_path_component(tmp_path):
    with pytest.raises(AssertionError, match="one path component"):
        with write_adapter_dir(tmp_path, "x/../y"):
            pass


def test_an_adapter_without_a_tenant_belongs_to_no_tenant(tmp_path):
    _save(tmp_path, "w", "1", tenant=None)
    with pytest.raises(OwnershipError):
        resolve_adapter_dir(tmp_path, "w", tenant="alice")


def test_resolve_refuses_a_path_as_a_name(tmp_path):
    with pytest.raises(AssertionError, match="one path component"):
        resolve_adapter_dir(tmp_path, "x/../w", tenant=None)


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
