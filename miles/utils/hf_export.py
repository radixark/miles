"""Shared constants and helpers for HF checkpoint export and consumption.

Kept dependency-free: imported by the megatron export path, the rollout manager's
eval controller, and the standalone checkpoint eval service.
"""

from pathlib import Path

HF_EXPORT_COMPLETE_MARKER = ".complete"


def is_complete_hf_export(path: str | Path) -> bool:
    """Whether ``path`` holds a finished HF export (the completeness marker exists)."""
    return (Path(path) / HF_EXPORT_COMPLETE_MARKER).exists()


def looks_like_hf_checkpoint(path: str | Path) -> bool:
    """Marker-less fallback heuristic for checkpoints written before the marker existed."""
    path = Path(path)
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
