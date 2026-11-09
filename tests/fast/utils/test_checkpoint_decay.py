import pytest

from miles.utils import checkpoint_decay


def _make_steps(root, steps):
    for step in steps:
        (root / f"iter_{step:07d}").mkdir()
        (root / "rollout" / str(step)).mkdir(parents=True)


def _list_steps(root):
    iters = sorted(int(p.name[len("iter_") :]) for p in root.glob("iter_*"))
    assert iters == sorted(int(p.name) for p in (root / "rollout").iterdir())
    return iters


def test_keeps_newest_and_every_nth_step_with_its_rollout_state(tmp_path):
    _make_steps(tmp_path, range(3, 40))

    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=39, keep_recent=5, keep_every=10)

    assert _list_steps(tmp_path) == [9, 19, 29, 35, 36, 37, 38, 39]


def test_leaves_steps_after_a_rolled_back_save_and_keeps_the_saved_one(tmp_path):
    _make_steps(tmp_path, range(90, 131))

    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=101, keep_recent=3, keep_every=10)

    assert _list_steps(tmp_path) == [99, 100, 101, *range(102, 131)]


def test_deletes_only_iter_and_rollout_step_dirs_under_save(tmp_path):
    _make_steps(tmp_path, range(1, 30))
    for name in ("iter_12", "iter_0000001.bak", "latest_checkpointed_iteration.txt", "logs"):
        (tmp_path / name).mkdir()
    (tmp_path / "rollout" / "notes").mkdir()

    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=29, keep_recent=3, keep_every=10)

    for name in ("iter_12", "iter_0000001.bak", "latest_checkpointed_iteration.txt", "logs", "rollout/notes"):
        assert (tmp_path / name).is_dir()


def test_pinned_step_survives_while_neighbours_are_pruned(tmp_path):
    _make_steps(tmp_path, range(1, 10))

    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=4, keep_recent=2, keep_every=100, pin_saved=True)
    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=9, keep_recent=2, keep_every=100)

    assert (tmp_path / "iter_0000004.keep").is_file()
    assert sorted(p.name for p in tmp_path.glob("iter_???????")) == ["iter_0000004", "iter_0000008", "iter_0000009"]
    assert sorted(p.name for p in (tmp_path / "rollout").iterdir()) == ["4", "8", "9"]


def test_removing_the_marker_lets_the_step_be_pruned(tmp_path):
    _make_steps(tmp_path, range(1, 10))
    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=4, keep_recent=2, keep_every=100, pin_saved=True)
    (tmp_path / "iter_0000004.keep").unlink()

    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=9, keep_recent=2, keep_every=100)

    assert sorted(p.name for p in tmp_path.glob("iter_???????")) == ["iter_0000008", "iter_0000009"]


def test_a_symlinked_step_is_refused_and_its_target_survives(tmp_path):
    outside = tmp_path / "outside"
    (outside / "keep").mkdir(parents=True)
    save = tmp_path / "save"
    save.mkdir()
    (save / "iter_0000001").symlink_to(outside)
    _make_steps(save, range(2, 10))

    with pytest.raises(OSError):
        checkpoint_decay.prune_checkpoints(str(save), saved_step=9, keep_recent=2, keep_every=100)

    assert (outside / "keep").is_dir()


def test_with_a_save_interval_it_keeps_every_nth_saved_step(tmp_path):
    _make_steps(tmp_path, range(4, 100, 5))

    checkpoint_decay.prune_checkpoints(str(tmp_path), saved_step=99, keep_recent=2, keep_every=20)

    assert _list_steps(tmp_path) == [19, 39, 59, 79, 94, 99]
