import logging
import os
import time
from collections import deque

import ray

from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


def eval_fn_needs_snapshot(args) -> bool:
    """Whether the eval fn declares ``eval_needs_snapshot = True``: it delivers HF
    snapshots to its own backend, so dispatch must export one per eval point."""
    fn = load_function(args.eval_function_path)
    return bool(getattr(fn, "eval_needs_snapshot", False))


class EvalDispatcher:
    """Fire-and-forget evals pinned to HF snapshots — against the dedicated eval
    fleet (``--eval-num-gpus``) or an eval fn that owns weight delivery
    (``eval_needs_snapshot``). Blocking shared-engine call otherwise. Failures
    degrade to a skipped point, never a crash."""

    def __init__(self, args, actor_model, rollout_manager):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, ray.ObjectRef]] = deque()
        fn_owns_delivery = eval_fn_needs_snapshot(args)
        assert not (fn_owns_delivery and args.eval_num_gpus > 0), (
            "eval fn declares eval_needs_snapshot (it delivers weights to its own backend); "
            "--eval-num-gpus must be 0"
        )
        if fn_owns_delivery:
            # Mirrors the --eval-num-gpus > 0 validation in arguments.py, which cannot
            # see the fn attribute (eval_function_path is only resolved after it runs).
            assert (
                enable_experimental_rollout_refactor()
            ), "eval_needs_snapshot requires the class-based rollout API (MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1)."
            assert args.eval_hf_dir is not None or args.save_hf is not None, (
                "eval_needs_snapshot requires a snapshot source: set --eval-hf-dir (staging exports) "
                "or --save-hf (reuse periodic HF checkpoints)."
            )
            assert args.eval_keep_snapshots >= args.eval_max_in_flight, (
                f"--eval-keep-snapshots ({args.eval_keep_snapshots}) must be >= --eval-max-in-flight "
                f"({args.eval_max_in_flight}), otherwise a pending eval's snapshot could be GC'd."
            )
        self._snapshot_eval = fn_owns_delivery or args.eval_num_gpus > 0

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        if not self._snapshot_eval:
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
            return self.args.save_hf.format(rollout_id=rollout_id), None
        hf_dir = os.path.join(self.args.eval_hf_dir, f"step_{rollout_id}")
        start = time.time()
        await self.actor_model.export_hf(rollout_id, hf_dir)
        return hf_dir, time.time() - start

    def _reap_finished(self) -> None:
        while self.pending:
            done, _ = ray.wait([self.pending[0][1]], timeout=0)
            if not done:
                break
            rollout_id, ref = self.pending.popleft()
            try:
                ray.get(ref)
            except Exception:
                logger.exception(f"Async eval for rollout {rollout_id} raised")
                self.rollout_manager.report_eval_skip.remote(rollout_id, "crashed")

    async def _await_ref(self, rollout_id: int, ref) -> None:
        try:
            await ref
        except Exception:
            logger.exception(f"Async eval for rollout {rollout_id} raised")
            await self.rollout_manager.report_eval_skip.remote(rollout_id, "crashed")
