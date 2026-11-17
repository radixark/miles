import asyncio
import logging
import os
import shutil
import time
from collections import deque

logger = logging.getLogger(__name__)


class EvalDispatcher:
    """Dispatches eval points without holding up the training loop.

    Exports a snapshot per point, fires the eval, and reclaims the snapshot once the
    point settles. Failures degrade to a skipped point outside CI; under ``--ci-test``,
    every skipped point fails the run. Shared-engine eval takes the plain blocking call
    instead, unless engines are carved out of training for it (``--eval-pool-num-engines``):
    then each point pins them at the current weights and gives them back when it settles.
    """

    def __init__(self, args, actor_model, rollout_manager, inference_controller=None):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.inference_controller = inference_controller
        self.pending: deque[tuple[int, asyncio.Task, str | None]] = deque()
        # whether an eval point under --eval-pool-num-engines got room before its weight update
        self._room_of_rollout_id: dict[int, bool] = {}
        self._exported: list[str] = []

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        """A caller-supplied ``hf_dir`` is an existing checkpoint, not one of our exports."""
        if self.args.eval_pool_num_engines > 0:
            await self._dispatch_on_carved_pool(rollout_id, force=force)
            return
        if not self.args.eval_uses_snapshots:
            await self.rollout_manager.eval(rollout_id)
            return

        if not await self._make_room(rollout_id, force=force):
            return

        export_time = None
        exported_dir = None
        require_marker = hf_dir is None
        if hf_dir is None:
            if self.args.eval_hf_dir is None:
                hf_dir = self.args.save_hf.format(rollout_id=rollout_id)
            else:
                hf_dir = os.path.join(self.args.eval_hf_dir, f"step_{rollout_id}")
                try:
                    export_time = await self._export(rollout_id, hf_dir)
                except Exception as e:
                    logger.error(f"HF snapshot export for eval {rollout_id} failed: {e}")
                    await self.rollout_manager.report_eval_skip(rollout_id, "export_failed")
                    return
                exported_dir = hf_dir

        task = asyncio.ensure_future(
            self.rollout_manager.eval(
                rollout_id, hf_dir=hf_dir, export_time_seconds=export_time, require_marker=require_marker
            )
        )
        self.pending.append((rollout_id, task, exported_dir))

    async def _dispatch_on_carved_pool(self, rollout_id: int, *, force: bool) -> None:
        """The carved engines hold the weights of this point; the hf_dir of a caller-supplied checkpoint is moot."""
        assert self.inference_controller is not None, "carving engines out for eval needs the inference controller"
        # a point with no weight update before it (the one before training) makes its room here
        room = self._room_of_rollout_id.pop(rollout_id, None)
        if room is None:
            room = await self._make_room(rollout_id, force=force)
        if not room:
            return
        carve_out = await self.inference_controller.carve_out_eval_pool()
        if (skip_reason := carve_out.skip_reason) is not None:
            await self.rollout_manager.report_eval_skip(rollout_id, skip_reason)
            return
        task = asyncio.ensure_future(self._eval_on_carved_pool(rollout_id, carve_out.weight_version))
        self.pending.append((rollout_id, task, None))

    async def make_room_before_weight_update(self, rollout_id: int, *, force: bool) -> None:
        """Settle the eval holding the carved engines before this point's weight update.

        Given-back engines rejoin training only through a weight update; settled after it, they would still be
        pending weights when this point carves, and training could not spare them.
        """
        if self.args.eval_pool_num_engines == 0:
            return
        self._room_of_rollout_id[rollout_id] = await self._make_room(rollout_id, force=force)

    async def _eval_on_carved_pool(self, rollout_id: int, weight_version: str) -> None:
        # the engines go back to training the moment this point settles, whatever its outcome
        try:
            await self.rollout_manager.eval(rollout_id, weight_version=weight_version)
        finally:
            await self.inference_controller.give_back_eval_pool()

    async def _make_room(self, rollout_id: int, *, force: bool) -> bool:
        """Settle the oldest pending eval when the in-flight limit is reached, or skip this point under 'skip'."""
        await self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip" and not force:
                await self.rollout_manager.report_eval_skip(rollout_id, "busy")
                return False
            await self._settle(*self.pending.popleft())
        return True

    async def drain(self) -> None:
        while self.pending:
            await self._settle(*self.pending.popleft())

    async def _export(self, rollout_id: int, hf_dir: str) -> float:
        start = time.time()
        if self.args.offload_train:
            await self.actor_model.onload()
        try:
            await self.actor_model.export_hf(rollout_id, hf_dir)
        finally:
            if self.args.offload_train:
                await self.actor_model.offload()
        return time.time() - start

    async def _reap_finished(self) -> None:
        while self.pending:
            if not self.pending[0][1].done():
                break
            await self._settle(*self.pending.popleft())

    async def _settle(self, rollout_id: int, task: asyncio.Task, exported_dir: str | None) -> None:
        try:
            await task
        except Exception:
            logger.exception(f"Async eval for rollout {rollout_id} raised")
            await self.rollout_manager.report_eval_skip(rollout_id, "crashed")
        finally:
            self._retire(exported_dir)

    def _retire(self, exported_dir: str | None) -> None:
        """Every settled point retires exactly one snapshot, whatever its outcome."""
        if exported_dir is None:
            return
        self._exported.append(exported_dir)
        while len(self._exported) > self.args.eval_keep_snapshots:
            victim = self._exported.pop(0)
            shutil.rmtree(victim, ignore_errors=True)
            logger.info(f"GC'd consumed eval snapshot {victim}")
