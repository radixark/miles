import asyncio
import logging
import os
from typing import Any, TypeVar

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.train_batch_coordinator import TrainBatchCoordinator
from miles.utils import object_store
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _await_task_terminal(task: asyncio.Future[_T]) -> _T:
    """Await a remote task to completion even if the caller is cancelled."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _await_task_before_cancellation(task: asyncio.Future[_T]) -> _T:
    """Settle a remote task before propagating caller cancellation."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            await _await_task_terminal(task)
        except BaseException as terminal_error:
            raise cancellation from terminal_error
        raise


async def _await_remote_result(remote_ref: Any) -> Any:
    return await remote_ref


async def _await_remote_result_with_cancellation(remote_ref: Any) -> tuple[Any, asyncio.CancelledError | None]:
    task = asyncio.ensure_future(_await_remote_result(remote_ref))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), cancellation
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
            if task.done():
                try:
                    return task.result(), cancellation
                except BaseException as terminal_error:
                    raise cancellation from terminal_error
        except BaseException as terminal_error:
            if cancellation is not None:
                raise cancellation from terminal_error
            raise


async def _rollback_prefetched_pack_before_exit(coordinator: Any, pack: Any, *, preserve_primary_error: bool) -> bool:
    rollback_task = asyncio.ensure_future(coordinator.rollback_prefetched(pack))
    if preserve_primary_error:
        return await _await_task_terminal(rollback_task)
    return await _await_task_before_cancellation(rollback_task)


async def _update_weights_with_admission_hold(
    rollout_manager: Any,
    actor_model: Any,
    rollout_id: int | None,
) -> None:
    """Publish trainer weights behind one exact rollout admission hold."""
    acquire_task = asyncio.ensure_future(rollout_manager.acquire_train_admission_hold.remote())
    try:
        hold_id = await asyncio.shield(acquire_task)
    except asyncio.CancelledError as cancellation:
        try:
            hold_id = await _await_task_terminal(acquire_task)
            release_task = asyncio.ensure_future(rollout_manager.release_train_admission_hold.remote(hold_id))
            await _await_task_terminal(release_task)
        except BaseException as cleanup_error:
            raise cancellation from cleanup_error
        raise

    # Once the hold is published, an update or terminal-frontier failure
    # leaves it active and fail-closed for the caller to reconcile.
    await rollout_manager.wait_weight_update_admission.remote(hold_id)
    if rollout_id is None:
        await actor_model.update_weights()
    else:
        await actor_model.update_weights(rollout_id=rollout_id)

    async def commit_weight_update() -> None:
        await rollout_manager.record_train_weight_update.remote(hold_id)
        await rollout_manager.release_train_admission_hold.remote(hold_id)

    await _await_task_before_cancellation(asyncio.create_task(commit_weight_update()))


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
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
    await _update_weights_with_admission_hold(rollout_manager, actor_model, None)

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_manager)
    train_batch_coordinator = TrainBatchCoordinator(
        args=args,
        actor_model=actor_model,
        critic_model=critic_model,
        rollout_manager=rollout_manager,
    )

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    # async train loop.
    rollout_data_curr_ref = None
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    rollout_data_next_pack = None

    async def sync_prefetched_rollout() -> None:
        nonlocal rollout_data_curr_ref, rollout_data_next_future, rollout_data_next_pack
        if rollout_data_next_pack is not None:
            rollout_data_curr_ref = rollout_data_next_pack
            rollout_data_next_pack = None
            return
        if rollout_data_next_future is not None:
            prefetched_pack, cancellation = await _await_remote_result_with_cancellation(rollout_data_next_future)
            rollout_data_next_future = None
            if cancellation is not None:
                try:
                    rolled_back = await _rollback_prefetched_pack_before_exit(
                        train_batch_coordinator,
                        prefetched_pack,
                        preserve_primary_error=True,
                    )
                    if not rolled_back:
                        remove_rollout_data_refs(args, prefetched_pack)
                except BaseException as cleanup_error:
                    raise cancellation from cleanup_error
                raise cancellation
            rollout_data_curr_ref = prefetched_pack

    async def drain_prefetched_rollout(*, preserve_primary_error: bool) -> None:
        """Settle a prefetched result before a lifecycle fence or failure."""
        nonlocal rollout_data_next_future, rollout_data_next_pack
        cancellation: asyncio.CancelledError | None = None
        if rollout_data_next_pack is not None:
            prefetched_pack = rollout_data_next_pack
            rollout_data_next_pack = None
        elif rollout_data_next_future is not None:
            prefetched_future = rollout_data_next_future
            prefetched_pack, cancellation = await _await_remote_result_with_cancellation(prefetched_future)
            rollout_data_next_future = None
        else:
            return

        rolled_back = await _rollback_prefetched_pack_before_exit(
            train_batch_coordinator,
            prefetched_pack,
            preserve_primary_error=preserve_primary_error,
        )
        if not rolled_back:
            # Legacy packs have no manager-owned settlement and are retained for
            # the next iteration after a save fence.
            rollout_data_next_pack = prefetched_pack
        if cancellation is not None and not preserve_primary_error:
            if not rolled_back:
                rollout_data_next_pack = None
                remove_rollout_data_refs(args, prefetched_pack)
            raise cancellation

    def cleanup_terminal_prefetch() -> None:
        nonlocal rollout_data_next_pack
        if rollout_data_next_pack is None:
            return
        prefetched_pack = rollout_data_next_pack
        rollout_data_next_pack = None
        remove_rollout_data_refs(args, prefetched_pack)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        await sync_prefetched_rollout()

        external_save_before_train = args.save_trigger_sentinel is not None and os.path.exists(
            args.save_trigger_sentinel
        )
        periodic_save = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        update_weights = (rollout_id + 1) % args.update_weights_interval == 0
        periodic_eval = should_run_periodic_action(
            rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout
        )
        debug_exit = (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        )
        known_boundary = external_save_before_train or periodic_save or update_weights or periodic_eval or debug_exit

        if rollout_id + 1 < args.num_rollout and not known_boundary:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        try:
            await train_batch_coordinator.train(rollout_id, rollout_data_curr_ref)
        except BaseException as training_error:
            try:
                await drain_prefetched_rollout(preserve_primary_error=True)
                cleanup_terminal_prefetch()
            except BaseException as cleanup_error:
                raise training_error from cleanup_error
            raise

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        save_now = external_save or periodic_save
        if save_now:
            if external_save:
                await drain_prefetched_rollout(preserve_primary_error=False)
            force_sync = external_save or rollout_id == args.num_rollout - 1
            await save_training_model(actor_model, rollout_id, force_sync)
            if args.use_critic:
                await save_training_model(critic_model, rollout_id, force_sync)
            await rollout_manager.save.remote(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if update_weights:
            await _update_weights_with_admission_hold(rollout_manager, actor_model, rollout_id)

        if periodic_eval:
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        if debug_exit:
            cleanup_terminal_prefetch()
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

        if rollout_id + 1 < args.num_rollout and rollout_data_next_future is None and rollout_data_next_pack is None:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    await eval_dispatcher.drain()
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
