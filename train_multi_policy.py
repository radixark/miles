import asyncio
import logging

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.ray.placement_group import create_rollout_components, maybe_start_api_server, update_weights
from miles.ray.wiring import launch_worker_manager
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.async_utils import wait_cancelling_pending_on_first_exception
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.multi_policy.utils import (
    TrainerInfo,
    create_trainers,
    define_policy_metric_groups,
    validate_multi_policy_args,
)
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)


async def train_multi_policy(args) -> None:
    megatron_config = resolve_megatron_config(args)
    validate_multi_policy_args(args, megatron_config=megatron_config)
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    maybe_start_periodic_pyspy_dump()
    init_tracking(args)
    define_policy_metric_groups(megatron_config)
    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)

    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)

    trainers = await create_trainers(args, rollout_executor=rollout_executor)

    maybe_start_api_server(
        args,
        actor_model=trainers[megatron_config.leader_model_id].handle,
        inference_controller=inference_controller,
    )

    maybe_start_mini_ft_controller(args)

    for model_id, trainer in trainers.items():
        await update_weights(trainer.handle, rollout_executor, trainer_model_id=model_id)
        if args.check_weight_update_equal:
            await inference_controller.check_weights(
                action="compare",
                allow_quant_error=args.check_weight_update_allow_quant_error,
                selector=args.check_weight_update_selector,
                skip_list=args.check_weight_update_skip_list,
                model_id=model_id,
            )

    tasks = [
        asyncio.create_task(
            _run_policy(
                args,
                trainer=trainer,
                inference_controller=inference_controller,
                rollout_executor=rollout_executor,
            )
        )
        for trainer in trainers.values()
    ]
    await wait_cancelling_pending_on_first_exception(tasks)

    await rollout_executor.dispose()
    await inference_controller.dispose()
    for trainer in trainers.values():
        await trainer.handle.dispose()


async def _run_policy(
    args,
    *,
    trainer: TrainerInfo,
    inference_controller: BaseWorkerHandle,
    rollout_executor: BaseWorkerHandle,
) -> None:
    model_id = trainer.model_id

    for rollout_id in range(trainer.start_rollout_id, args.num_rollout):
        await inference_controller.prepare_rollout(rollout_id, model_id=model_id)
        rollout_data_pack = await rollout_executor.get(rollout_id, trainer_model_id=model_id)
        await trainer.handle.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            await update_weights(trainer.handle, rollout_executor, rollout_id=rollout_id, trainer_model_id=model_id)

        if (x := args.debug_exit_after_rollout) is not None and (rollout_id - trainer.start_rollout_id + 1) >= x:
            logger.info(f"debug_exit_after_rollout={x} reached at rollout_id={rollout_id}, exiting")
            break


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train_multi_policy(args))
    finally:
        finish_tracking()
