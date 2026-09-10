from pathlib import Path

from miles.backends.training_utils.weight_version_checkpoint import read_weight_version, write_weight_version


class TestWeightVersionCheckpoint:
    def test_versions_are_selected_by_checkpoint_iteration(self, tmp_path: Path) -> None:
        """Restoring an older checkpoint reads its own absolute version."""
        write_weight_version(checkpoint_dir=tmp_path, iteration=3, weight_version=9)
        write_weight_version(checkpoint_dir=tmp_path, iteration=4, weight_version=12)

        assert read_weight_version(checkpoint_dir=tmp_path, iteration=3) == 9
        assert read_weight_version(checkpoint_dir=tmp_path, iteration=4) == 12

    def test_legacy_checkpoint_starts_at_zero(self, tmp_path: Path) -> None:
        """A checkpoint predating the counter remains loadable."""
        (tmp_path / "iter_0000003").mkdir()

        assert read_weight_version(checkpoint_dir=tmp_path, iteration=3) == 0

    def test_rewriting_rolled_back_iteration_replaces_counter(self, tmp_path: Path) -> None:
        """Saving after rollback replaces the abandoned version at that iteration."""
        write_weight_version(checkpoint_dir=tmp_path, iteration=3, weight_version=12)
        write_weight_version(checkpoint_dir=tmp_path, iteration=3, weight_version=9)

        assert read_weight_version(checkpoint_dir=tmp_path, iteration=3) == 9
