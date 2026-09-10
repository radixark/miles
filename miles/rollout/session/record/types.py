from dataclasses import dataclass
from typing import Protocol

from miles.rollout.session.types import SessionRecord

SessionRecordKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class RecordRef:
    key: SessionRecordKey


class RecordStorageError(RuntimeError):
    pass


class RecordReadError(RecordStorageError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class RecordBackend(Protocol):
    async def put(self, key: SessionRecordKey, record: SessionRecord) -> None: ...

    async def get(self, key: SessionRecordKey) -> SessionRecord: ...

    async def delete(self, key: SessionRecordKey) -> None: ...

    async def close(self) -> None: ...
