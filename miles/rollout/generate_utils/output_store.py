"""Read SGLang output-store references back into replay arrays.

With ``--sglang-output-store-backend mooncake``, a training request sets
``return_outputs_via_store`` and SGLang returns its routed experts, indexer top-k
and sampling mask as one Mooncake bundle instead of inline base64 / JSON lists:
``meta_info["output_store_ref"] = {"handle": ..., "fields": {name: {"dtype", "shape"}}}``.
The resolver reads the bundle through Miles' object store, copies every field into
an owned array, and removes the bundle, so each object lives for one read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from miles.utils import object_store
from miles.utils.object_store import BaseObjectStore, MooncakeObjectStore, StoreObjectRef

logger = logging.getLogger(__name__)

OUTPUT_STORE_REF_KEY = "output_store_ref"

# SGLang writes only the fields a request asked for, each as one row of this layout.
_FIELD_LAYOUTS: dict[str, tuple[np.dtype, int]] = {
    "routed_experts": (np.dtype(np.int32), 3),
    "indexer_topk": (np.dtype(np.int32), 3),
    "output_token_sampling_mask_lengths": (np.dtype(np.int32), 1),
    "output_token_sampling_mask_token_ids": (np.dtype(np.int32), 1),
    "output_token_sampling_logprobs": (np.dtype(np.float32), 1),
}
_SAMPLING_MASK_FIELDS = (
    "output_token_sampling_mask_lengths",
    "output_token_sampling_mask_token_ids",
    "output_token_sampling_logprobs",
)


@dataclass(frozen=True)
class ReplayOutputs:
    """Replay arrays of one response read back from the output store; ``None`` when not returned."""

    routed_experts: np.ndarray | None = None
    indexer_topk: np.ndarray | None = None
    sampling_mask_lengths: np.ndarray | None = None
    sampling_mask_token_ids: np.ndarray | None = None
    sampling_logprobs: np.ndarray | None = None

    @property
    def has_sampling_mask(self) -> bool:
        return self.sampling_mask_lengths is not None


def output_store_enabled(args: Any) -> bool:
    # The flag exists only when the installed SGLang has --output-store-backend.
    return getattr(args, "sglang_output_store_backend", "none") == "mooncake"


async def resolve_replay_outputs(
    meta_info: Mapping[str, Any], *, store: BaseObjectStore | None = None
) -> ReplayOutputs | None:
    """Read and remove the bundle ``meta_info`` references, off the event loop.

    Returns ``None`` for a response without a ref. ``store`` defaults to the
    process's object store, looked up only when a ref is present.
    """
    output_store_ref = meta_info.get(OUTPUT_STORE_REF_KEY)
    if output_store_ref is None:
        return None
    store = store if store is not None else object_store.get_instance()
    return await asyncio.to_thread(read_replay_outputs, output_store_ref, store=store)


def read_replay_outputs(output_store_ref: Any, *, store: BaseObjectStore) -> ReplayOutputs:
    """Validate the ref, copy its fields out of the store, then remove the bundle (blocking).

    The bundle is removed even when reading or validating it fails; a failed
    removal is logged with the handle and does not fail the read.
    """
    handle, layouts = _parse_ref(output_store_ref)
    if not isinstance(store, MooncakeObjectStore):
        raise ValueError("an output_store_ref can only be read with --object-store-backend mooncake")
    ref = object_store.mooncake_ref_from_handle(handle)
    try:
        with store.get(ref) as value:
            arrays = _copy_fields(value, layouts)
    finally:
        _remove(store, ref, handle)
    return ReplayOutputs(
        routed_experts=arrays.get("routed_experts"),
        indexer_topk=arrays.get("indexer_topk"),
        sampling_mask_lengths=arrays.get("output_token_sampling_mask_lengths"),
        sampling_mask_token_ids=arrays.get("output_token_sampling_mask_token_ids"),
        sampling_logprobs=arrays.get("output_token_sampling_logprobs"),
    )


def _parse_ref(output_store_ref: Any) -> tuple[dict[str, Any], dict[str, tuple[np.dtype, tuple[int, ...]]]]:
    if not isinstance(output_store_ref, Mapping) or set(output_store_ref) != {"handle", "fields"}:
        raise ValueError(f"output_store_ref must be an object with 'handle' and 'fields', got {output_store_ref!r}")
    handle, fields = output_store_ref["handle"], output_store_ref["fields"]
    if not isinstance(handle, Mapping):
        raise ValueError(f"output_store_ref.handle must be an object, got {handle!r}")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError(f"output_store_ref.fields must be a non-empty object, got {fields!r}")

    layouts = {}
    for name, field in fields.items():
        if name not in _FIELD_LAYOUTS:
            raise ValueError(f"output_store_ref has unknown field {name!r}")
        dtype, ndim = _FIELD_LAYOUTS[name]
        shape = field.get("shape") if isinstance(field, Mapping) else None
        if (
            not isinstance(field, Mapping)
            or field.get("dtype") != dtype.name
            or not isinstance(shape, list)
            or len(shape) != ndim
            or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
        ):
            raise ValueError(f"output_store_ref field {name!r} must be {dtype.name} with {ndim} dims, got {field!r}")
        layouts[name] = (dtype, tuple(shape))

    num_sampling_fields = sum(name in layouts for name in _SAMPLING_MASK_FIELDS)
    if num_sampling_fields not in (0, len(_SAMPLING_MASK_FIELDS)):
        raise ValueError(f"output_store_ref must carry all or none of {_SAMPLING_MASK_FIELDS}, got {sorted(layouts)}")
    return dict(handle), layouts


def _copy_fields(
    value: Mapping[str, Any], layouts: dict[str, tuple[np.dtype, tuple[int, ...]]]
) -> dict[str, np.ndarray]:
    if set(value) != set(layouts):
        raise ValueError(f"output store bundle holds fields {sorted(value)}, but its ref lists {sorted(layouts)}")
    arrays = {}
    for name, (dtype, shape) in layouts.items():
        rows = value[name]
        if len(rows) != 1:
            raise ValueError(f"output store field {name!r} has {len(rows)} rows, expected 1")
        # Rows arrive as Python lists or pool-backed arrays; either way copy into an owned array.
        array = np.array(rows[0], dtype=dtype)
        if array.size == 0 and math.prod(shape) == 0:
            # An empty nested list keeps none of the trailing dims.
            array = array.reshape(shape)
        if array.shape != shape:
            raise ValueError(f"output store field {name!r} has shape {array.shape}, but its ref says {shape}")
        arrays[name] = array
    return arrays


def _remove(store: BaseObjectStore, ref: StoreObjectRef, handle: dict[str, Any]) -> None:
    try:
        store.remove(ref)
    except Exception:
        # Nothing else references the object; the logged handle is the only record of it.
        logger.error("Failed to remove an SGLang output-store object; handle=%s", json.dumps(handle), exc_info=True)
