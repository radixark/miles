import subprocess
from pathlib import Path

import pytest

from miles.utils.test_utils.comparisons import dumps
from miles.utils.test_utils.comparisons.dumps import _find_leaf_dump_dirs, compare_dumps


class TestFindLeafDumpDirs:
    def test_two_pt_files_in_one_leaf_yield_single_entry(self, tmp_path: Path) -> None:
        """Multiple .pt files sharing one leaf dir dedup to a single relative entry."""
        leaf = tmp_path / "fwd_bwd" / "rollout_0"
        leaf.mkdir(parents=True)
        (leaf / "step_0.pt").touch()
        (leaf / "step_1.pt").touch()

        assert _find_leaf_dump_dirs(tmp_path) == ["fwd_bwd/rollout_0"]

    def test_two_leaves_returned_sorted(self, tmp_path: Path) -> None:
        """Distinct leaf dirs are returned sorted by their relative path string."""
        leaf_b = tmp_path / "leaf_b"
        leaf_a = tmp_path / "leaf_a"
        leaf_b.mkdir()
        leaf_a.mkdir()
        (leaf_b / "x.pt").touch()
        (leaf_a / "y.pt").touch()

        assert _find_leaf_dump_dirs(tmp_path) == ["leaf_a", "leaf_b"]

    def test_pt_file_directly_in_root_yields_dot(self, tmp_path: Path) -> None:
        """A .pt file directly under root has parent equal to root, reported as '.'."""
        (tmp_path / "step_0.pt").touch()

        assert _find_leaf_dump_dirs(tmp_path) == ["."]

    def test_no_pt_files_yields_empty_list(self, tmp_path: Path) -> None:
        """A tree with no .pt files produces an empty list."""
        (tmp_path / "sub").mkdir()

        assert _find_leaf_dump_dirs(tmp_path) == []


def test_excluded_tensors_are_replaced_by_a_separate_semantic_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excluded tensor files must not reach the generic comparator while its report remains auditable."""
    baseline_leaf = tmp_path / "baseline" / "dumps" / "rollout_2"
    target_leaf = tmp_path / "target" / "dumps" / "rollout_2"
    baseline_leaf.mkdir(parents=True)
    target_leaf.mkdir(parents=True)
    for leaf in (baseline_leaf, target_leaf):
        (leaf / "step=0___name=grad__model.weight.pt").touch()
        (leaf / "step=0___name=grad__local_head_witness.weight.pt").touch()

    def fake_run_comparator(**kwargs: object) -> subprocess.CompletedProcess[str]:
        baseline_path = kwargs["baseline_path"]
        target_path = kwargs["target_path"]
        assert isinstance(baseline_path, Path)
        assert isinstance(target_path, Path)
        baseline_names = {path.name for path in baseline_path.glob("*.pt")}
        target_names = {path.name for path in target_path.glob("*.pt")}
        assert baseline_names == {"step=0___name=grad__model.weight.pt"}
        assert target_names == baseline_names
        (target_path / "comparator_report.jsonl").write_text("report\n")
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr(dumps, "run_comparator", fake_run_comparator)

    compare_dumps(
        baseline_dir=str(tmp_path / "baseline"),
        target_dir=str(tmp_path / "target"),
        diff_thresholds=[(".*", "rel <= 0")],
        allow_skipped_pattern="^$",
        allow_failed_pattern="^$",
        excluded_tensor_pattern=r".*witness.*",
    )

    assert (target_leaf / "comparator_report.jsonl").read_text() == "report\n"


class TestIgnoredFiles:
    def test_non_pt_files_are_ignored(self, tmp_path: Path) -> None:
        """Files not matching *.pt (including *.pth) are ignored by the glob."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "weights.pth").touch()
        (tmp_path / "data.json").touch()

        assert _find_leaf_dump_dirs(tmp_path) == []
