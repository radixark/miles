"""Coordinate admission, settlement, and execution of one train batch."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from miles.ray.train_batch_admission import (
    RayTrainerAdmissionAdapter,
    TrainBatchPublication,
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
)
from miles.utils.data import remove_rollout_data_refs


class _PendingCommitError(RuntimeError):
    """A manager explicitly rejected commit while its source is still pending."""


class _CommitCancelled(RuntimeError):
    """A commit await was cancelled after the manager recorded a status."""

    def __init__(self, cancellation: asyncio.CancelledError, status: TrainerAdmissionStatus) -> None:
        super().__init__(f"Trainer admission commit was cancelled with status {status.value}.")
        self.cancellation = cancellation
        self.status = status


class TrainBatchCoordinator:
    """Provide one lifecycle seam for legacy and leased trainer batches."""

    def __init__(
        self,
        *,
        args: Any,
        actor_model: object,
        critic_model: object | None,
        rollout_manager: object | None,
        admission_adapter: object | None = None,
    ) -> None:
        self._args = args
        self._actor_model = actor_model
        self._critic_model = critic_model
        self._admission_adapter = admission_adapter
        if self._admission_adapter is None and rollout_manager is not None:
            self._admission_adapter = RayTrainerAdmissionAdapter(rollout_manager)

    async def train(self, rollout_id: int, rollout_data_pack: dict[str, Any]) -> None:
        """Train one batch, settling a leased publication when present."""
        if "trainer_admission" not in rollout_data_pack:
            await self._train_legacy(rollout_id, rollout_data_pack)
            remove_rollout_data_refs(self._args, rollout_data_pack)
            return

        publication = rollout_data_pack["trainer_admission"]
        if not isinstance(publication, TrainBatchPublication):
            raise ValueError(f"Rollout {rollout_id} has an invalid trainer admission publication.")

        receipts: list[TrainerAdmissionReceipt] = []
        try:
            await self._admit(rollout_id, rollout_data_pack, publication, receipts)
        except BaseException as admission_error:
            await self._settle_before_training(
                publication=publication,
                receipts=receipts,
                primary_error=admission_error,
                rollback=True,
            )
            raise

        try:
            committed = await self._commit_or_reconcile(publication, receipts)
        except _CommitCancelled as cancelled:
            if cancelled.status is TrainerAdmissionStatus.COMMITTED:
                await self._settle_after_training(
                    rollout_data_pack=rollout_data_pack,
                    receipts=receipts,
                    primary_error=cancelled.cancellation,
                )
            else:
                await self._settle_before_training(
                    publication=publication,
                    receipts=receipts,
                    primary_error=cancelled.cancellation,
                    rollback=cancelled.status is TrainerAdmissionStatus.PENDING,
                )
            raise cancelled.cancellation from None
        except BaseException as commit_error:
            await self._settle_before_training(
                publication=publication,
                receipts=receipts,
                primary_error=commit_error,
                rollback=isinstance(commit_error, _PendingCommitError),
            )
            raise

        if not committed:
            rejection = RuntimeError(f"Trainer admission {publication.admission_id} was not committed.")
            await self._settle_before_training(
                publication=publication,
                receipts=receipts,
                primary_error=rejection,
                rollback=True,
            )
            raise rejection

        try:
            await self._train_admitted(rollout_id, rollout_data_pack, receipts)
        except BaseException as training_error:
            await self._settle_after_training(
                rollout_data_pack=rollout_data_pack,
                receipts=receipts,
                primary_error=training_error,
            )
            raise
        await self._settle_after_training(
            rollout_data_pack=rollout_data_pack,
            receipts=receipts,
        )

    async def rollback_prefetched(self, rollout_data_pack: dict[str, Any]) -> bool:
        """Return an unadmitted leased batch to its source for replay.

        Legacy train-data packs have no manager-owned admission to settle and are
        left untouched. The return value says whether rollback consumed the pack.
        """
        if "trainer_admission" not in rollout_data_pack:
            return False
        publication = rollout_data_pack["trainer_admission"]
        if not isinstance(publication, TrainBatchPublication):
            raise ValueError("Prefetched train data has an invalid trainer admission publication.")
        if self._admission_adapter is None:
            raise RuntimeError("A rollout manager is required for leased trainer data.")
        status = await self._admission_adapter.rollback(publication)
        if status is not TrainerAdmissionStatus.ROLLED_BACK:
            raise RuntimeError(f"Trainer admission {publication.admission_id} rollback failed: {status!r}.")
        return True

    async def _admit(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
        publication: TrainBatchPublication,
        receipts: list[TrainerAdmissionReceipt],
    ) -> None:
        if publication.rollout_id != rollout_id:
            raise ValueError(f"Trainer admission {publication.admission_id} has the wrong rollout id.")
        if self._admission_adapter is None:
            raise RuntimeError("A rollout manager is required for leased trainer data.")

        for role in self._role_order(publication.required_roles):
            group = self._group_for_role(role)
            if group is None:
                raise RuntimeError(f"Trainer admission {publication.admission_id} has no {role} trainer group.")
            receipt = await group.admit_train_batch(rollout_id, rollout_data_pack)
            if not isinstance(receipt, TrainerAdmissionReceipt):
                raise RuntimeError(f"Trainer admission {publication.admission_id} received an invalid {role} receipt.")
            if receipt.publication != publication or receipt.role != role:
                raise RuntimeError(f"Trainer admission {publication.admission_id} received a stale {role} receipt.")
            receipts.append(receipt)

        expected_roles = set(publication.required_roles)
        if {receipt.role for receipt in receipts} != expected_roles:
            raise RuntimeError(
                f"Trainer admission {publication.admission_id} did not acknowledge every required role."
            )

    async def _commit_or_reconcile(
        self,
        publication: TrainBatchPublication,
        receipts: list[TrainerAdmissionReceipt],
    ) -> bool:
        assert self._admission_adapter is not None
        try:
            status = await self._admission_adapter.commit(publication, tuple(receipts))
        except asyncio.CancelledError as cancellation:
            try:
                status = await self._status(publication)
            except BaseException as status_error:
                raise cancellation from status_error
            raise _CommitCancelled(cancellation, status) from cancellation
        except BaseException as commit_error:
            try:
                status = await self._status(publication)
            except BaseException as status_error:
                raise commit_error from status_error
            if status is TrainerAdmissionStatus.COMMITTED:
                return True
            if status is TrainerAdmissionStatus.PENDING:
                raise _PendingCommitError(
                    f"Trainer admission {publication.admission_id} commit was rejected while still pending."
                ) from commit_error
            raise commit_error

        if status is TrainerAdmissionStatus.COMMITTED:
            return True
        if status is TrainerAdmissionStatus.PENDING:
            return False
        raise RuntimeError(f"Trainer admission {publication.admission_id} commit failed: {status!r}.")

    async def _status(self, publication: TrainBatchPublication) -> TrainerAdmissionStatus:
        assert self._admission_adapter is not None
        return await self._admission_adapter.status(publication)

    async def _settle_before_training(
        self,
        *,
        publication: TrainBatchPublication,
        receipts: list[TrainerAdmissionReceipt],
        primary_error: BaseException,
        rollback: bool,
    ) -> None:
        cleanup_error: BaseException | None = None
        if rollback:
            try:
                await self._rollback(publication)
            except BaseException as error:
                cleanup_error = error
        try:
            await self._discard_pins(receipts)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise primary_error from cleanup_error

    async def _rollback(self, publication: TrainBatchPublication) -> None:
        if self._admission_adapter is None:
            return
        status = await self._admission_adapter.rollback(publication)
        if status is not TrainerAdmissionStatus.ROLLED_BACK:
            raise RuntimeError(f"Trainer admission {publication.admission_id} rollback failed: {status!r}.")

    async def _discard_pins(self, receipts: list[TrainerAdmissionReceipt]) -> None:
        for receipt in receipts:
            group = self._group_for_role(receipt.role)
            if group is None:
                continue
            discard = getattr(group, "discard_train_batch_admission", None)
            if discard is None:
                continue
            result = discard(receipt)
            if inspect.isawaitable(result):
                await result

    async def _train_admitted(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
        receipts: list[TrainerAdmissionReceipt],
    ) -> None:
        by_role = {receipt.role: receipt for receipt in receipts}
        values = None
        if "critic" in by_role:
            critic = self._group_for_role("critic")
            assert critic is not None
            values = await critic.train(
                rollout_id,
                rollout_data_pack,
                admission_receipt=by_role["critic"],
            )
            await self._maybe_offload("critic")
        if "actor" in by_role:
            actor = self._group_for_role("actor")
            assert actor is not None
            if values is None:
                await actor.train(
                    rollout_id,
                    rollout_data_pack,
                    admission_receipt=by_role["actor"],
                )
            else:
                await actor.train(
                    rollout_id,
                    rollout_data_pack,
                    external_data=values,
                    admission_receipt=by_role["actor"],
                )
            await self._maybe_offload("actor")

    async def _train_legacy(self, rollout_id: int, rollout_data_pack: dict[str, Any]) -> None:
        values = None
        if getattr(self._args, "use_critic", False):
            critic = self._group_for_role("critic")
            if critic is None:
                raise RuntimeError("Critic training is enabled but no critic trainer group exists.")
            values = await critic.train(rollout_id, rollout_data_pack)
            await self._maybe_offload("critic")
        if not getattr(self._args, "use_critic", False) or rollout_id >= getattr(
            self._args, "num_critic_only_steps", 0
        ):
            actor = self._group_for_role("actor")
            if actor is None:
                raise RuntimeError("No actor trainer group exists.")
            if values is None:
                await actor.train(rollout_id, rollout_data_pack)
            else:
                await actor.train(rollout_id, rollout_data_pack, external_data=values)
            if getattr(self._args, "use_critic", False):
                await self._maybe_offload("actor")

    async def _maybe_offload(self, role: str) -> None:
        if not getattr(self._args, "use_critic", False) or not getattr(self._args, "offload_train", False):
            return
        group = self._group_for_role(role)
        if group is not None:
            await group.offload()

    async def _settle_after_training(
        self,
        *,
        rollout_data_pack: dict[str, Any],
        receipts: list[TrainerAdmissionReceipt],
        primary_error: BaseException | None = None,
    ) -> None:
        cleanup_error: BaseException | None = None
        try:
            await self._discard_pins(receipts)
        except BaseException as error:
            cleanup_error = error
        try:
            remove_rollout_data_refs(self._args, rollout_data_pack)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if primary_error is not None and cleanup_error is not None:
            raise primary_error from cleanup_error
        if cleanup_error is not None:
            raise cleanup_error

    def _group_for_role(self, role: str) -> object | None:
        return self._critic_model if role == "critic" else self._actor_model if role == "actor" else None

    @staticmethod
    def _role_order(required_roles: frozenset[str]) -> tuple[str, ...]:
        known = tuple(role for role in ("critic", "actor") if role in required_roles)
        unknown = tuple(sorted(required_roles.difference(known)))
        return known + unknown
