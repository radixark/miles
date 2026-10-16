"""A fake Mooncake object store that serves SGLang output-store bundles."""

from typing import Any

from miles.utils import object_store
from miles.utils.object_store import MooncakeObjectStore


class FakeMooncakeStore(MooncakeObjectStore):
    """The object-store boundary: serves one bundle as Mooncake's dict get returns it (Python lists)."""

    def __init__(self, bundle: dict[str, Any], *, remove_error: Exception | None = None):
        self._bundle = bundle
        self._remove_error = remove_error
        self.removed: list[Any] = []
        self.released = False

    def get(self, ref):
        return object_store.ObjectStoreGetResult(value=self._bundle, release_fn=self._release)

    def remove(self, ref):
        self.removed.append(ref.payload)
        if self._remove_error is not None:
            raise self._remove_error

    def _release(self, value):
        self.released = True


def output_store_ref(handle: dict[str, Any], **fields: tuple[str, list[int]]) -> dict[str, Any]:
    """``meta_info["output_store_ref"]`` for a bundle with the given ``name=(dtype, shape)`` fields."""
    return {
        "handle": handle,
        "fields": {name: {"dtype": dtype, "shape": shape} for name, (dtype, shape) in fields.items()},
    }
