from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_process


def test_npu_command_patch_applies_to_the_current_package() -> None:
    """The NPU launcher patch must target a file that git can apply to this checkout."""
    root = Path(__file__).resolve().parents[3]
    run_process(
        [
            "git",
            "-C",
            str(root),
            "apply",
            "--check",
            "--include=miles/utils/external_utils/command_utils/npu.py",
            str(root / "docker/npu_patch/miles.patch"),
        ],
        capture_output=True,
        check=True,
    )
    patch = (root / "docker/npu_patch/miles.patch").read_text()
    assert "+++ b/miles/utils/external_utils/command_utils/npu.py" in patch
    assert "+++ b/miles/utils/external_utils/command_utils.py" not in patch
