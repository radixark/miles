"""Output store refs of samples dropped before training, waiting for the rollout executor.

Every drop site runs in the rollout executor's process but sees neither the executor nor the
rollout id, so it records the dropped samples' refs here. The executor's next ``get`` files them
with that rollout's trained samples and deletes them once that rollout has trained: a dropped
session leaf shares refs with its trained siblings, so deleting at the drop could be too early.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from miles.utils.types import Sample

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef

_dropped: dict[str, StoreRef] = {}


def store_refs_of(samples: Iterable[Any]) -> dict[str, StoreRef]:
    """Every output store ref the samples own, by key; groups may nest samples in lists."""
    refs = {}
    for item in samples:
        if isinstance(item, Sample):
            if (plan := item.rollout_store_plan) is not None:
                refs.update((ref.key, ref) for ref in plan.owned_refs)
        else:
            refs.update(store_refs_of(item))
    return refs


def record_dropped_samples(samples: Iterable[Any]) -> None:
    """Hand the refs of samples that will never train to the rollout executor."""
    _dropped.update(store_refs_of(samples))


def take_dropped_refs() -> dict[str, StoreRef]:
    """The refs recorded since the last call; the caller now owns them."""
    refs = dict(_dropped)
    _dropped.clear()
    return refs
