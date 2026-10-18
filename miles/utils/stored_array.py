"""Replay arrays of finished samples, held in the object store instead of the rollout executor's heap.

With ``--rollout-replay-via-object-store``, a finished training sample's routed experts and indexer
top-k are put into the object store as soon as the sample finishes, and the sample keeps a
``StoredArray`` in their place. The executor owns the object until it hands the batch to the trainer:
dropping the sample removes the object, and the hand-off moves it into the ``RolloutDataPack``, which
the driver removes after the train step. The trainer reads the array back unchanged.
"""

import uuid
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from miles.utils import object_store
from miles.utils.object_store import StoreObjectRef, ValueSpec
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.types import Sample

REPLAY_ARRAY_FIELDS = ("rollout_routed_experts", "rollout_indexer_topk")

_ROWS_KEY = "rows"
_ROWS_SPEC = {_ROWS_KEY: ValueSpec(codec="typed_ragged")}

_checkpoint_write_dir: ContextVar[Path | None] = ContextVar("stored_array_checkpoint_write_dir", default=None)
_checkpoint_read_dir: ContextVar[Path | None] = ContextVar("stored_array_checkpoint_read_dir", default=None)


def stored_field_name(field: str) -> str:
    """The train data column carrying the handles of ``field``'s stored arrays."""
    return f"{field}_stored"


class StoredArrayHandle(FrozenStrictBaseModel):
    """What the train data carries for a stored array: the object and the array it holds."""

    ref: StoreObjectRef
    dtype: str
    shape: tuple[int, ...]

    def read(self) -> np.ndarray:
        with object_store.get_instance().get(self.ref) as value:
            # copy out: the value may be backed by a buffer the release returns
            array = np.array(value[_ROWS_KEY][0], dtype=self.dtype)
        assert array.shape == self.shape, f"stored array has shape {array.shape}, its handle says {self.shape}"
        return array


class StoredArray:
    """A sample field whose bytes live in the object store; removed there when this object dies un-handed-off."""

    __slots__ = ("handle", "_finalizer", "__weakref__")

    def __init__(self, handle: StoredArrayHandle):
        self.handle = handle
        self._finalizer = weakref.finalize(self, object_store.get_instance().remove, handle.ref)
        self._finalizer.atexit = False

    @classmethod
    def put(cls, array: np.ndarray) -> "StoredArray":
        ref = object_store.get_instance().put(value={_ROWS_KEY: [array]}, value_spec=_ROWS_SPEC)
        return cls(StoredArrayHandle(ref=ref, dtype=array.dtype.name, shape=array.shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.handle.shape

    def __len__(self) -> int:
        return self.handle.shape[0]

    def hand_off(self) -> StoreObjectRef:
        """Move ownership of the object to the caller; the executor no longer removes it."""
        assert self._finalizer.detach() is not None, "a stored array is handed off at most once"
        return self.handle.ref

    # Copies share one owner: a second one would remove the object under the first.
    def __copy__(self) -> "StoredArray":
        return self

    def __deepcopy__(self, memo) -> "StoredArray":
        return self

    def __array__(self, *args, **kwargs):
        raise TypeError("a StoredArray holds no bytes in this process; read its handle instead")

    def __reduce__(self):
        directory = _checkpoint_write_dir.get()
        assert directory is not None, "a StoredArray pickles only inside stored_array_checkpoint_writes"
        name = f"{uuid.uuid4().hex}.npy"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / name, self.handle.read())
        return _load_stored_array, (name,)


def store_replay_arrays(samples: list[Sample]) -> None:
    """Put the replay arrays of finished samples into the object store (blocking)."""
    for sample in samples:
        for field in REPLAY_ARRAY_FIELDS:
            # a sample already finished in an earlier attempt holds a StoredArray
            if isinstance(value := getattr(sample, field), np.ndarray):
                setattr(sample, field, StoredArray.put(value))


def _load_stored_array(name: str) -> StoredArray:
    directory = _checkpoint_read_dir.get()
    assert directory is not None, "a StoredArray unpickles only inside stored_array_checkpoint_reads"
    return StoredArray.put(np.load(directory / name))


@contextmanager
def stored_array_checkpoint_writes(directory: Path) -> Iterator[None]:
    """Stored arrays pickled in this scope write their bytes to ``directory``, one file each."""
    token = _checkpoint_write_dir.set(directory)
    try:
        yield
    finally:
        _checkpoint_write_dir.reset(token)


@contextmanager
def stored_array_checkpoint_reads(directory: Path) -> Iterator[None]:
    """Stored arrays unpickled in this scope are read from ``directory`` and put back into the store."""
    token = _checkpoint_read_dir.set(directory)
    try:
        yield
    finally:
        _checkpoint_read_dir.reset(token)
