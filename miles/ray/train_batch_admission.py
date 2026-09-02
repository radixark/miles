"""Immutable protocol values for admitting manager-published train batches.

The manager owns publication settlement.  Trainer adapters only prove that each
rank can read the exact published object-store references it consumes.  They
return an immutable cohort pin for the later training lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from miles.utils.ray_utils import Box


class TrainerAdmissionStatus(Enum):
    """Recorded source-settlement state for one trainer admission."""

    PENDING = "pending"
    COMMITTED = "committed"
    COMMIT_FAILED = "commit_failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


class TrainerCohortChangedError(RuntimeError):
    """The trainer cohort no longer matches its side-effect-free admission."""


@dataclass(frozen=True)
class TrainBatchPublication:
    """Exact immutable identity of a manager-published leased train batch."""

    manager_incarnation: str
    admission_id: int
    rollout_id: int
    data_ref_ids: tuple[str, ...]
    required_roles: frozenset[str]


@dataclass(frozen=True)
class TrainerRankReceipt:
    """Side-effect-free proof from one trainer rank.

    ``data_parallel`` is the rank and size of the data-parallel group whose shard this
    rank read. It is ``None`` when the rank read every published reference.
    """

    publication: TrainBatchPublication
    rank: int
    data_parallel: tuple[int, int] | None = None


@dataclass(frozen=True)
class TrainerCellCohort:
    """The exact live ranks admitted in one trainer cell."""

    cell_index: int
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class TrainerCohort:
    """Immutable quorum/cell/rank snapshot returned by a trainer group."""

    quorum_id: int | None
    cells: tuple[TrainerCellCohort, ...]


@dataclass(frozen=True)
class TrainerAdmissionReceipt:
    """A role-level acknowledgement for one exact publication."""

    publication: TrainBatchPublication
    role: str
    cohort: TrainerCohort


def required_trainer_roles(args: Namespace, rollout_id: int) -> frozenset[str]:
    """Return the roles that must acknowledge ``rollout_id``."""
    if not getattr(args, "use_critic", False):
        return frozenset({"actor"})
    if rollout_id < getattr(args, "num_critic_only_steps", 0):
        return frozenset({"critic"})
    return frozenset({"actor", "critic"})


def data_ref_ids(data_ref: Box | list[Box]) -> tuple[str, ...]:
    """Return stable digests for an ordered nonempty Box publication."""
    refs = data_ref if isinstance(data_ref, list) else [data_ref]
    if not refs:
        raise ValueError("A train-data publication must contain at least one Box reference.")
    if any(not isinstance(ref, Box) for ref in refs):
        raise TypeError("Train-data publication references must be Box values.")
    return tuple(_data_ref_id(ref) for ref in refs)


def _data_ref_id(ref: Box) -> str:
    inner = ref.inner
    if callable(to_hex := getattr(inner, "hex", None)):
        payload = to_hex()
    elif isinstance(inner, bytes):
        payload = inner.hex()
    else:
        try:
            payload = json.dumps(inner, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Unsupported exported object-store reference {type(inner).__name__}.") from error
    type_name = f"{type(inner).__module__}.{type(inner).__qualname__}"
    return hashlib.sha256(f"{type_name}:{payload}".encode()).hexdigest()


def validate_data_parallel_coverage(
    receipts: Sequence[TrainerRankReceipt],
    publication: TrainBatchPublication,
) -> None:
    """Check that rank receipts together cover every published train batch shard.

    Receipts that report no data-parallel layout read every published reference, so
    one of them already covers the whole batch.
    """
    layouts = tuple(receipt.data_parallel for receipt in receipts if receipt.data_parallel is not None)
    if not layouts:
        return
    sizes = sorted({size for _, size in layouts})
    if len(sizes) > 1:
        raise RuntimeError(
            f"Trainer admission {publication.admission_id} received data-parallel sizes {sizes} from one cohort."
        )
    size = sizes[0]
    if len(publication.data_ref_ids) != size:
        raise RuntimeError(
            f"Trainer admission {publication.admission_id} received receipts for a data-parallel size of "
            f"{size} against {len(publication.data_ref_ids)} published shards."
        )
    # A receipt without a layout read every published shard, so none can be left unread.
    if len(layouts) != len(receipts):
        return
    missing = sorted(set(range(size)) - {rank for rank, _ in layouts})
    if missing:
        raise RuntimeError(f"Trainer admission {publication.admission_id} left train batch shards {missing} unread.")


def validate_publication_data_ref(publication: TrainBatchPublication, data_ref: Any) -> None:
    """Reject a trainer acknowledgement for any publication other than its token."""
    if data_ref_ids(data_ref) != publication.data_ref_ids:
        raise ValueError(f"Admission {publication.admission_id} does not match the published data reference.")


class RayTrainerAdmissionAdapter:
    """Call manager settlement APIs and reconcile lost Ray responses."""

    def __init__(self, rollout_manager: object) -> None:
        self._rollout_manager = rollout_manager

    async def commit(
        self,
        publication: TrainBatchPublication,
        receipts: tuple[TrainerAdmissionReceipt, ...],
    ) -> TrainerAdmissionStatus:
        try:
            return await self._rollout_manager.commit_trainer_admission.remote(publication, receipts)
        except asyncio.CancelledError:
            raise
        except Exception as commit_error:
            try:
                status = await self.status(publication)
            except Exception as status_error:
                raise commit_error from status_error
            if status is TrainerAdmissionStatus.COMMITTED:
                return status
            raise

    async def rollback(self, publication: TrainBatchPublication) -> TrainerAdmissionStatus:
        try:
            return await self._rollout_manager.rollback_trainer_admission.remote(publication)
        except asyncio.CancelledError:
            raise
        except Exception as rollback_error:
            try:
                status = await self.status(publication)
            except Exception as status_error:
                raise rollback_error from status_error
            if status is TrainerAdmissionStatus.ROLLED_BACK:
                return status
            raise

    async def status(self, publication: TrainBatchPublication) -> TrainerAdmissionStatus:
        return await self._rollout_manager.get_trainer_admission_status.remote(publication)
