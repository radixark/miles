import asyncio
import shutil
from pathlib import Path

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakCollectionClosedEvent,
    SoakEvidenceArchivedEvent,
    SoakRunContextEvent,
    file_sha256,
    read_events,
)

_COLLECTION_TIMEOUT_SECONDS = 180.0


async def finish(event_log: EventLog) -> None:
    try:
        async with asyncio.timeout(_COLLECTION_TIMEOUT_SECONDS):
            await asyncio.to_thread(_collect, event_log)
    except TimeoutError as error:
        raise TimeoutError(f"Soak evidence collection exceeded {_COLLECTION_TIMEOUT_SECONDS}s") from error


def _collect(event_log: EventLog) -> None:
    if (path := event_log.path) is not None:
        if contexts := [event for event in event_log.events if isinstance(event, SoakRunContextEvent)]:
            event_log.append(_archive_sources(sources=contexts[-1].sources, destination=path.parent / "sources"))
        assert (
            read_events(path, require_closed=False) == event_log.events
        ), f"Persisted soak evidence differs from memory: {path}"
    event_log.append(SoakCollectionClosedEvent())


def _archive_sources(*, sources: dict[str, Path], destination: Path) -> SoakEvidenceArchivedEvent:
    archived: dict[str, Path] = {}
    missing: list[str] = []
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        assert name and Path(name).name == name and name not in (".", ".."), f"Invalid evidence source name: {name}"
        if not source.is_dir():
            missing.append(name)
            continue
        root = destination / name
        target = root / source.name
        shutil.copytree(source, target)
        for discarded in sorted(source.parent.glob(".trash_*")):
            if discarded.is_dir():
                shutil.copytree(discarded, root / discarded.name)
        archived[name] = target
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(destination.parent))] = file_sha256(path)
    return SoakEvidenceArchivedEvent(sources=archived, missing_sources=missing, sha256_of_file=hashes)
