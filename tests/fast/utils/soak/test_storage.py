import subprocess
from pathlib import Path

import pytest
from tests.utils.soak import storage


@pytest.mark.parametrize(
    ("filesystem", "mount", "size", "valid"),
    [
        ("nfs4", "/data", 2 * 10**12, True),
        ("overlay", "/data", 2 * 10**12, False),
        ("ext4", "/", 2 * 10**12, False),
        ("ext4", "/data", 10**11, False),
    ],
)
def test_dump_storage_rejects_overlay_root_and_small_disks(
    filesystem: str, mount: str, size: int, valid: bool
) -> None:
    """A large reported capacity cannot make overlay or root storage a valid dump destination."""
    facts = storage.DumpStorageFacts(
        requested_path=Path("/data/dumps"),
        checked_path=Path("/data"),
        mount_point=Path(mount),
        filesystem_type=filesystem,
        total_bytes=size,
        available_bytes=10**10,
        df_output="raw df",
    )
    if valid:
        storage._assert_large_mounted_storage(facts)
    else:
        with pytest.raises(AssertionError):
            storage._assert_large_mounted_storage(facts)


async def test_missing_dump_directory_is_checked_on_its_existing_parent_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight inspects the filesystem that would receive a new directory before writing it."""
    requested = tmp_path / "missing" / "run"
    commands: list[list[str]] = []

    async def command(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        output = "Filesystem Type 1024-blocks Used Available Capacity Mounted on\nserver:/data nfs4 2000000000 100 1999999900 1% /data\n"
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(storage, "run_command", command)
    facts = await storage._validate_dump_storage(requested)
    assert all(command[-1] == str(tmp_path.resolve()) for command in commands)
    assert facts.total_bytes == 2000000000 * 1024
    assert facts.available_bytes == 1999999900 * 1024
    assert facts.requested_path == requested and not requested.exists()


def test_actual_dump_and_checkpoint_arguments_are_checked_even_outside_the_default_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit write paths cannot bypass preflight by leaving the configured dumps root mounted correctly."""
    checked: list[Path] = []
    monkeypatch.setattr(storage, "validate_dump_storage", checked.append)
    storage.validate_training_storage(
        "--dumper-dir '/other/dumps with spaces' --save /other/checkpoints --load /readonly/model"
    )
    assert set(checked) == {Path("/other/dumps with spaces"), Path("/other/checkpoints")}
