from pathlib import Path

from miles.utils.file_utils import atomic_write_text


def write_weight_version(checkpoint_dir: Path, *, iteration: int, weight_version: int) -> None:
    directory = checkpoint_dir / f"iter_{iteration:07d}"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path=directory / "weight_version.txt", text=str(weight_version))


def read_weight_version(checkpoint_dir: Path, *, iteration: int) -> int:
    path = checkpoint_dir / f"iter_{iteration:07d}" / "weight_version.txt"
    if not path.is_file():
        return 0
    return int(path.read_text().strip())
