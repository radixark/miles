import asyncio
import logging
import os
import time
from collections import deque

import ray

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)


class EvalDispatcher:
    """Dispatch evals against the dedicated eval fleet without stalling training.

    With ``--eval-num-gpus 0`` this degrades to today's blocking ``eval.remote`` call.
    Otherwise each due eval gets an HF snapshot (exported to ``--eval-hf-dir``, or the
    ``--save-hf`` checkpoint in reuse mode) and is fired fire-and-forget; the pending
    set is bounded by ``--eval-max-in-flight`` via the ``--eval-overflow-policy``.
    Failures degrade to a skipped point logged at that rollout_id, never a crash.
    """

    def __init__(self, args, actor_model, rollout_manager):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, ray.ObjectRef]] = deque()

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None) -> None:
        if self.args.eval_num_gpus <= 0:
            await self.rollout_manager.eval.remote(rollout_id)
            return

        self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip":
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
        """Await every pending eval (before dispose, so the last points always land)."""
        while self.pending:
            rollout_id, ref = self.pending.popleft()
            await self._await_ref(rollout_id, ref)

    async def _ensure_snapshot(self, rollout_id: int) -> tuple[str, float | None]:
        if self.args.eval_hf_dir is None:
            # Reuse mode: the --save-hf checkpoint for this step was just written
            # (eval_interval is validated to be a multiple of save_interval).
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


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_manager)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        # The base checkpoint is the snapshot for the pre-train eval; no export needed.
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_curr_ref))
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref)
            await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                await critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            await rollout_manager.save.remote(rollout_id)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            await actor_model.update_weights(rollout_id=rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await eval_dispatcher.dispatch(rollout_id)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

    await eval_dispatcher.drain()
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
