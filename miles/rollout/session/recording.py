from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

from miles.rollout.session.record.store import RecordStore
from miles.rollout.session.record.types import RecordRef
from miles.rollout.session.types import SessionRecord


@dataclass(frozen=True)
class RecordCheckpoint:
    ref: RecordRef
    tools: list[dict] | None = None


@contextmanager
def commit_record(store: RecordStore, session_id: str, record: SessionRecord):
    """Publish under the session gate without awaiting; discard if publication fails."""
    tools = deepcopy(record.request.get("tools"))
    ref = store.put(session_id, record)
    try:
        yield RecordCheckpoint(ref, tools)
    except BaseException:
        store.delete(ref)
        raise
