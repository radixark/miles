from pathlib import Path

CHECKPOINT_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def read_checkpoint_tracker_iteration(checkpoint_root: Path) -> int | None:
    tracker = checkpoint_root / CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)
