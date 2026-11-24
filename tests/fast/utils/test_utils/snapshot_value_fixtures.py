from dataclasses import dataclass
from pathlib import Path

from pydantic import field_serializer

from miles.utils.pydantic_utils import StrictBaseModel


class _SnapshotModel(StrictBaseModel):
    location: Path

    @field_serializer("location")
    def _serialize_location(self, value: Path) -> str:
        return "serializer-mask"


@dataclass(frozen=True)
class _SnapshotDataclass:
    location: Path
