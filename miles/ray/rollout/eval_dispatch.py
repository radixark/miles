import logging
import os
import shutil
import time
from collections import deque

import ray

logger = logging.getLogger(__name__)


class EvalDispatcher:
    """Dispatches eval points without holding up the training loop.

    Exports a snapshot per point, fires the eval, and reclaims the snapshot once the
    point settles. Failures degrade to a skipped point, never a crash. Shared-engine
    eval takes the plain blocking call instead.
    """

    def __init__(self, args, actor_model, rollout_manager):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, ray.ObjectRef, str | None]] = deque()
        self._exported: list[str] = []

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        """A caller-supplied ``hf_dir`` is an existing checkpoint, not one of our exports."""
        if not self.args.eval_uses_snapshots:
            await self.rollout_manager.eval.remote(rollout_id)
            return

        await self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip" and not force:
                await self.rollout_manager.report_eval_skip.remote(rollout_id, "busy")
                return
            await self._settle(*self.pending.popleft())

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
                    shutil.rmtree(hf_dir, ignore_errors=True)
                    await self.rollout_manager.report_eval_skip.remote(rollout_id, "export_failed")
                    return
                exported_dir = hf_dir

        ref = self.rollout_manager.eval.remote(
            rollout_id, hf_dir=hf_dir, export_time_seconds=export_time, require_marker=require_marker
        )
        self.pending.append((rollout_id, ref, exported_dir))

    async def drain(self) -> None:
        while self.pending:
            await self._settle(*self.pending.popleft())

    async def _export(self, rollout_id: int, hf_dir: str) -> float:
        start = time.time()
        await self.actor_model.export_hf(rollout_id, hf_dir)
        return time.time() - start

    async def _reap_finished(self) -> None:
        while self.pending:
            done, _ = ray.wait([self.pending[0][1]], timeout=0)
            if not done:
                break
            await self._settle(*self.pending.popleft())

    async def _settle(self, rollout_id: int, ref, exported_dir: str | None) -> None:
        try:
            await ref
        except Exception:
            logger.exception(f"Async eval for rollout {rollout_id} raised")
            await self.rollout_manager.report_eval_skip.remote(rollout_id, "crashed")
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
