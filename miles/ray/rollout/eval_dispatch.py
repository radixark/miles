import logging
import os
import time
from collections import deque

import ray

logger = logging.getLogger(__name__)


class EvalDispatcher:
    """Fire-and-forget evals against the dedicated eval fleet; blocking legacy call
    when ``--eval-num-gpus`` is 0. Failures degrade to a skipped point, never a crash."""

    def __init__(self, args, actor_model, rollout_manager):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, ray.ObjectRef]] = deque()

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        if self.args.eval_num_gpus <= 0:
            await self.rollout_manager.eval.remote(rollout_id)
            return

        self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip" and not force:
                await self.rollout_manager.report_eval_skip.remote(rollout_id, "busy")
                return
            oldest_id, oldest_ref = self.pending.popleft()
            await self._await_ref(oldest_id, oldest_ref)

        export_time = None
        if hf_dir is None:
            try:
                hf_dir, export_time = await self._ensure_snapshot(rollout_id)
            except Exception as e:
                logger.error(f"HF snapshot export for eval {rollout_id} failed: {e}")
                await self.rollout_manager.report_eval_skip.remote(rollout_id, "export_failed")
                return

        ref = self.rollout_manager.eval.remote(rollout_id, hf_dir=hf_dir, export_time_seconds=export_time)
        if self.args.eval_dispatch == "blocking":
            await self._await_ref(rollout_id, ref)
        else:
            self.pending.append((rollout_id, ref))

    async def drain(self) -> None:
        while self.pending:
            rollout_id, ref = self.pending.popleft()
            await self._await_ref(rollout_id, ref)

    async def _ensure_snapshot(self, rollout_id: int) -> tuple[str, float | None]:
        if self.args.eval_hf_dir is None:
            # Reuse mode: the --save-hf checkpoint for this step was just written.
            return self.args.save_hf.format(rollout_id=rollout_id), None
        hf_dir = os.path.join(self.args.eval_hf_dir, f"step_{rollout_id}")
        start = time.time()
        await self.actor_model.export_hf(rollout_id, hf_dir)
        return hf_dir, time.time() - start

    def _reap_finished(self) -> None:
        # The manager's eval lock serializes evals, so pending refs finish in order.
        while self.pending:
            done, _ = ray.wait([self.pending[0][1]], timeout=0)
            if not done:
                break
            rollout_id, ref = self.pending.popleft()
            try:
                ray.get(ref)
            except Exception:
                logger.exception(f"Async eval for rollout {rollout_id} raised")

    async def _await_ref(self, rollout_id: int, ref) -> None:
        try:
            await ref
        except Exception:
            logger.exception(f"Async eval for rollout {rollout_id} raised")
