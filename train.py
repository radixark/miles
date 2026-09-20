import asyncio
import logging
import os

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import create_rollout_components, create_training_models, update_weights
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.wiring import launch_worker_manager
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.lora import lora_rollout_enabled
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)

_DISPOSE_TIMEOUT_SECONDS = 60.0
# Keep timed-out tasks alive until they settle. This bounds driver cleanup, not
# asyncio.run() shutdown if a disposer blocks the loop or refuses cancellation.
_pending_dispose_tasks: set[asyncio.Task] = set()


def _observe_dispose_result(task):
    _pending_dispose_tasks.discard(task)
    if not task.cancelled():
        # A timed-out disposer may finish later. Retrieve its exception so it
        # does not become an unhandled task exception during loop shutdown.
        task.exception()


async def _dispose_component(name, component):
    if name in ("rollout_executor", "worker_manager"):
        await component.dispose.remote()
        if name == "worker_manager":
            # Children must finish shutdown before their owning manager dies.
            # Use the handle returned by launch, never a global name lookup.
            ray.kill(component, no_restart=True)
    else:
        await component.dispose()


async def _dispose_training_components(
    rollout_executor, inference_controller, actor_model, critic_model, worker_manager
):
    """Release controllers, then the actual workers owned by this job."""
    first_error = None
    # Preserve the normal shutdown order: close the executor's data/eval
    # backends before unregistering inference services and stopping watchers.
    for name, component in (
        ("rollout_executor", rollout_executor),
        ("inference_controller", inference_controller),
        ("actor_model", actor_model),
        ("critic_model", critic_model),
        ("worker_manager", worker_manager),
    ):
        if component is None:
            continue
        task = asyncio.create_task(_dispose_component(name, component))
        _pending_dispose_tasks.add(task)
        task.add_done_callback(_observe_dispose_result)
        try:
            done, _ = await asyncio.wait([task], timeout=_DISPOSE_TIMEOUT_SECONDS)
            if not done:
                task.cancel()
                # wait_for() waits for cancellation acknowledgement, which can
                # hang and prevent all remaining components from being disposed.
                if name == "worker_manager":
                    logger.error("Worker teardown timed out; manager was not killed and worker children may remain")
                raise TimeoutError(f"Disposing {name} timed out after {_DISPOSE_TIMEOUT_SECONDS}s")
            task.result()
        except (Exception, asyncio.CancelledError) as error:
            logger.warning("Failed to dispose %s; continuing cleanup", name, exc_info=True)
            if first_error is None:
                first_error = error
    return first_error


async def train(args):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    if args.colocate_memory_peak_device == "gpu":
        assert (
            args.offload_train and args.offload_rollout
        ), "--colocate-memory-peak-device gpu requires --offload-train and --offload-rollout"
        assert not args.use_critic, "--colocate-memory-peak-device gpu is not wired for the critic path"

    worker_manager = inference_controller = rollout_executor = actor_model = critic_model = None
    original_error = None
    try:
        worker_manager = launch_worker_manager(args)
        object_store.init_instance(args, contribute_segment=False)
        init_tracking(args)
        # Initialize rollout first to calculate num_rollout. Only objects returned
        # by these factories are visible here; rollback of an object whose init
        # fails inside a factory must be implemented by that factory itself.
        inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
        actor_model, critic_model = await create_training_models(args, inference_controller, rollout_executor)
        await _train_with_components(
            args, inference_controller, rollout_executor, actor_model, critic_model, num_rollout_per_epoch
        )
    except BaseException as error:
        original_error = error
        raise
    finally:
        cleanup_task = asyncio.create_task(
            _dispose_training_components(
                rollout_executor, inference_controller, actor_model, critic_model, worker_manager
            )
        )
        cancellation = None
        # Defer even repeated cancellation until each known component has had a
        # bounded cleanup attempt. Never replace the original training exception.
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        cleanup_error = cleanup_task.result()
        if original_error is None:
            if cancellation is not None:
                raise cancellation
            if cleanup_error is not None:
                raise cleanup_error


async def _train_with_components(
    args, inference_controller, rollout_executor, actor_model, critic_model, num_rollout_per_epoch
):
    if args.api_server_port:
        start_api_server(
            args=args,
            actor_model=actor_model,
            inference_controller=inference_controller,
            host=args.api_server_host,
            port=args.api_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(actor_model, rollout_executor)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await inference_controller.onload_kv()

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def offload_train():
        if args.use_critic:
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False):
        force_sync = force_sync or rollout_id == args.num_rollout - 1

        async def save_training_model(model):
            if args.use_critic and args.offload_train:
                await model.onload()
            await model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic and args.offload_train:
                await model.offload()

        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            await save_training_model(actor_model)
        if args.use_critic:
            await save_training_model(critic_model)
        await rollout_executor.save.remote(rollout_id)

    if args.num_rollout > args.start_rollout_id and args.eval_interval is not None and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        if args.start_rollout_id == 0:
            await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)
        else:
            await eval_dispatcher.dispatch(args.start_rollout_id - 1)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        await inference_controller.prepare_rollout(rollout_id)
        rollout_data_pack = await rollout_executor.get.remote(rollout_id)

        if args.offload_rollout:
            if args.colocate_memory_peak_device == "gpu":
                await inference_controller.offload_kv()
                await actor_model.onload()
                await inference_controller.offload_weights()
            else:
                offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
                if "kv_cache" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
                if "weight" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
                await inference_controller.offload(tags=offload_tags)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_pack)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_pack, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
        else:
            await actor_model.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if args.colocate_memory_peak_device == "gpu":
            await actor_model.clear_memory()
            if lora_rollout_enabled(args):
                await actor_model.offload_grad_buffer()
            await inference_controller.onload_weights()
            await offload_train()
        else:
            await offload_train()
            if args.offload_rollout:
                await inference_controller.onload_weights()
        await update_weights(actor_model, rollout_executor, rollout_id=rollout_id)
        if args.offload_rollout:
            await inference_controller.onload_kv()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await inference_controller.prepare_eval()
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

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

    # Drain only on normal completion; an exceptional exit must not wait for
    # unrelated snapshot evals before attempting resource cleanup.
    await eval_dispatcher.drain()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
