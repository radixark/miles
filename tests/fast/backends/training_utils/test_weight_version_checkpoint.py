from pathlib import Path

from miles.backends.training_utils.weight_version_checkpoint import read_weight_version, write_weight_version


class TestWeightVersionCheckpoint:
    def test_versions_are_isolated_by_trainer_and_loaded_iteration(self, tmp_path: Path) -> None:
        """Loading an older iteration or another policy must not read the latest policy counter."""
        first = tmp_path / "trainers" / "solver-actor"
        second = tmp_path / "trainers" / "verifier-actor"
        write_weight_version(checkpoint_dir=first, iteration=3, weight_version=9)
        write_weight_version(checkpoint_dir=first, iteration=4, weight_version=12)
        write_weight_version(checkpoint_dir=second, iteration=3, weight_version=6)
        (first / "latest_checkpointed_iteration.txt").write_text("4")

        assert read_weight_version(checkpoint_dir=first, iteration=3) == 9
        assert read_weight_version(checkpoint_dir=first, iteration=4) == 12
        assert read_weight_version(checkpoint_dir=second, iteration=3) == 6
        assert (first / "iter_0000003" / "weight_version.txt").read_text() == "9"
        assert list(tmp_path.rglob("*.tmp")) == []

    def test_legacy_checkpoints_start_the_counter_at_zero(self, tmp_path: Path) -> None:
        """Checkpoints saved without a weight version remain loadable."""
        (tmp_path / "iter_0000003").mkdir()

        assert read_weight_version(checkpoint_dir=tmp_path, iteration=3) == 0

    def test_overwriting_an_iteration_replaces_its_counter(self, tmp_path: Path) -> None:
        """Saving again after a rollback must replace the abandoned counter."""
        write_weight_version(checkpoint_dir=tmp_path, iteration=3, weight_version=12)
        write_weight_version(checkpoint_dir=tmp_path, iteration=3, weight_version=9)

        assert read_weight_version(checkpoint_dir=tmp_path, iteration=3) == 9
