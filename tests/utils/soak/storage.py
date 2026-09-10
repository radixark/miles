import asyncio
import logging
import shlex
from pathlib import Path

from tests.utils.soak.action import run_command

from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class DumpStorageFacts(FrozenStrictBaseModel):
    requested_path: Path
    checked_path: Path
    mount_point: Path
    filesystem_type: str
    total_bytes: int
    available_bytes: int
    df_output: str


def validate_training_storage(train_args: str) -> None:
    argv = shlex.split(train_args)
    paths = {
        Path(value)
        for flag in ("--dumper-dir", "--save-debug-event-data", "--save-debug-rollout-data", "--save")
        if (value := ArgvManipulator.get_effective(argv, flag)) is not None
    }
    for path in sorted(paths):
        validate_dump_storage(path)


def validate_dump_storage(path: Path) -> DumpStorageFacts:
    return asyncio.run(_validate_dump_storage(path))


async def _validate_dump_storage(path: Path) -> DumpStorageFacts:
    checked = path.resolve()
    while not checked.exists():
        checked = checked.parent
    human = await run_command(["df", "-hT", str(checked)], timeout_seconds=30)
    logger.info("Dump storage for %s:\n%s", path, human.stdout)
    numeric = await run_command(["df", "-PkT", str(checked)], timeout_seconds=30)
    facts = _parse_storage_facts(
        requested_path=path, checked_path=checked, numeric_output=numeric.stdout, df_output=human.stdout
    )
    _assert_large_mounted_storage(facts)
    return facts


def _parse_storage_facts(
    *, requested_path: Path, checked_path: Path, numeric_output: str, df_output: str
) -> DumpStorageFacts:
    lines = numeric_output.strip().splitlines()
    assert len(lines) == 2, f"Expected one filesystem record from df: {numeric_output}"
    columns = lines[1].split(maxsplit=6)
    assert len(columns) == 7, f"Malformed filesystem record: {lines[1]}"
    return DumpStorageFacts(
        requested_path=requested_path,
        checked_path=checked_path,
        filesystem_type=columns[1],
        total_bytes=int(columns[2]) * 1024,
        available_bytes=int(columns[4]) * 1024,
        mount_point=Path(columns[6]),
        df_output=df_output,
    )


def _assert_large_mounted_storage(facts: DumpStorageFacts) -> None:
    assert facts.filesystem_type not in {
        "overlay",
        "tmpfs",
        "ramfs",
    }, f"Dump path {facts.requested_path} uses {facts.filesystem_type}, not persistent bulk storage"
    assert facts.mount_point != Path("/"), f"Dump path {facts.requested_path} falls on the root filesystem"
    assert (
        facts.total_bytes >= 10**12
    ), f"Dump path {facts.requested_path} is not on TB-scale storage: {facts.total_bytes} bytes"
    assert facts.available_bytes > 0, f"Dump storage is full: {facts.requested_path}"
