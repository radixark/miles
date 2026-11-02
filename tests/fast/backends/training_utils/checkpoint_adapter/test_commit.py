"""Commits are atomic link swaps, ownership is enforced, and recovery removes only unreferenced entries."""

import os

import pytest

from miles.backends.training_utils.checkpoint.adapter.commit import (
    OwnershipError,
    commit_adapter_dir,
    delete_adapter_dir,
    delete_unreferenced_adapter_dirs,
    resolve_adapter_dir,
    write_adapter_dir,
)


def _save(root, name, payload, *, tenant="alice", overwrite=True):
    with write_adapter_dir(root, name, tenant=tenant) as written:
        (written.path / "payload").write_text(payload)
    return commit_adapter_dir(root, name, written, overwrite=overwrite)


def test_overwrite_points_the_name_at_the_new_commit_and_removes_the_old(tmp_path):
    first = _save(tmp_path, "w", "1")
    second = _save(tmp_path, "w", "2")
    resolved = resolve_adapter_dir(tmp_path, "w", tenant="alice")
    assert resolved == second and (resolved.path / "payload").read_text() == "2"
    assert not first.path.exists()
    assert not os.path.isabs(os.readlink(tmp_path / "w")), "links must be relative so copies keep working"


def test_refuses_to_overwrite_when_asked_not_to(tmp_path):
    _save(tmp_path, "w", "1")
    with pytest.raises(FileExistsError):
        _save(tmp_path, "w", "2", overwrite=False)
    assert (resolve_adapter_dir(tmp_path, "w").path / "payload").read_text() == "1"


def test_another_tenant_cannot_resolve_or_delete(tmp_path):
    _save(tmp_path, "w", "1")
    with pytest.raises(OwnershipError):
        resolve_adapter_dir(tmp_path, "w", tenant="mallory")
    with pytest.raises(OwnershipError):
        delete_adapter_dir(tmp_path, "w", tenant="mallory")
    delete_adapter_dir(tmp_path, "w", tenant="alice")
    assert resolve_adapter_dir(tmp_path, "w") is None


def test_a_failed_write_is_never_visible(tmp_path):
    with pytest.raises(RuntimeError, match="adapter write failed"):
        with write_adapter_dir(tmp_path, "w") as written:
            (written.path / "partial").write_text("x")
            raise OSError("disk full")
    assert resolve_adapter_dir(tmp_path, "w") is None
    delete_unreferenced_adapter_dirs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_recovery_of_one_name_does_not_touch_a_dotted_neighbour(tmp_path):
    kept = _save(tmp_path, "foo.bar", "neighbour")
    with write_adapter_dir(tmp_path, "foo") as orphan:
        pass
    (tmp_path / "dangling").symlink_to(".gone.0011223344556677")
    delete_unreferenced_adapter_dirs(tmp_path, "foo")
    assert not orphan.path.exists()
    assert kept.path.exists() and resolve_adapter_dir(tmp_path, "foo.bar") == kept
    assert (tmp_path / "dangling").is_symlink(), "recovery scoped to 'foo' must leave other names alone"
    delete_unreferenced_adapter_dirs(tmp_path)
    assert not (tmp_path / "dangling").is_symlink()


def test_user_names_cannot_start_with_a_dot(tmp_path):
    with pytest.raises(AssertionError, match="must not start with"):
        with write_adapter_dir(tmp_path, ".w"):
            pass
