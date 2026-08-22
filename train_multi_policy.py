import asyncio
import itertools
import logging
import os
from collections.abc import Awaitable
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.ray.placement_group import create_rollout_components, maybe_start_api_server, update_weights
from miles.ray.specs.train import compute_trainer_configs
from miles.ray.wiring import shutdown_worker_manager
from miles.utils.arguments import parse_args
from miles.utils.async_utils import wait_cancelling_pending_on_first_completion
from miles.utils.data import remove_rollout_data_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.misc import should_run_periodic_action
from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from miles.utils.multi_policy.parker import Parker
from miles.utils.multi_policy.utils import (
    TrainerInfo,
    assert_consistent_restore,
    create_trainers,
    define_policy_metric_groups,
    validate_multi_policy_args,
)
from miles.utils.orchestration_utils import init_orchestration_script
from miles.utils.tracking_utils.tracking import finish_tracking
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)


async def train_multi_policy(args) -> None:
    megatron_config = resolve_megatron_config(args)
    validate_multi_policy_args(args, megatron_config=megatron_config)
    worker_manager = init_orchestration_script(args)
    inference_controller: BaseWorkerHandle | None = None
    rollout_executor: BaseWorkerHandle | None = None
    trainers: dict[str, TrainerInfo] = {}
    try:
        define_policy_metric_groups(megatron_config)
        inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)

        trainers = await create_trainers(args, rollout_executor=rollout_executor)
        assert_consistent_restore(args, trainers=trainers, leader_model_id=megatron_config.leader_model_id)

        maybe_start_api_server(
            args,
            trainer_models={
                trainer_config.trainer_id: trainers[trainer_config.model_id].handle
                for trainer_config in compute_trainer_configs(args)
            },
            inference_controller=inference_controller,
        )
        maybe_start_mini_ft_controller(args)

        for model_id, trainer in trainers.items():
            await update_weights(
                args,
                trainer.handle,
                rollout_executor,
                inference_controller,
                trainer_model_id=model_id,
            )
            if args.check_weight_update_equal:
                await inference_controller.check_weights(
                    action="compare",
                    allow_quant_error=args.check_weight_update_allow_quant_error,
                    selector=args.check_weight_update_selector,
                    skip_list=args.check_weight_update_skip_list,
                    model_id=model_id,
                )

        parker = Parker(num_followers=len(trainers) - 1)
        run_ended = asyncio.Event()
        rollout_ids: dict[str, int] = {}
        tasks = [
            asyncio.create_task(
                _run_policy(
                    args,
                    trainer=trainer,
                    is_leader=trainer.model_id == megatron_config.leader_model_id,
                    trainers=trainers,
                    inference_controller=inference_controller,
                    rollout_executor=rollout_executor,
                    parker=parker,
                    run_ended=run_ended,
                    rollout_ids=rollout_ids,
                    num_rollout_per_epoch=num_rollout_per_epoch,
                )
            )
            for trainer in trainers.values()
        ]
        await wait_cancelling_pending_on_first_completion(tasks, on_first_completion=run_ended.set)
    except BaseException:
        try:
            await _dispose_components(
                rollout_executor=rollout_executor,
                inference_controller=inference_controller,
                trainers=trainers,
                worker_manager=worker_manager,
            )
        except BaseException as cleanup_error:
            logger.error("Additional failure while disposing the run", exc_info=cleanup_error)
        raise
    else:
        await _dispose_components(
            rollout_executor=rollout_executor,
            inference_controller=inference_controller,
            trainers=trainers,
            worker_manager=worker_manager,
        )


async def _dispose_components(
    *,
    rollout_executor: BaseWorkerHandle | None,
    inference_controller: BaseWorkerHandle | None,
    trainers: dict[str, TrainerInfo],
    worker_manager: object,
) -> None:
    dispose_awaitables: list[Awaitable[object]] = [*(trainer.handle.dispose() for trainer in trainers.values())]
    if rollout_executor is not None:
        dispose_awaitables.append(rollout_executor.dispose())
    if inference_controller is not None:
        dispose_awaitables.append(inference_controller.dispose())
    results = await asyncio.gather(*dispose_awaitables, return_exceptions=True)
    try:
        await shutdown_worker_manager(worker_manager)
    except BaseException as error:
        results.append(error)

    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        primary_error = errors[0]
        for error in errors[1:]:
            logger.error("Additional component disposal failure", exc_info=error)
        raise primary_error


async def _run_policy(
    args,
    *,
    trainer: TrainerInfo,
    is_leader: bool,
    run_ended: asyncio.Event,
    trainers: dict[str, TrainerInfo],
    inference_controller: BaseWorkerHandle,
    rollout_executor: BaseWorkerHandle,
    parker: Parker,
    rollout_ids: dict[str, int],
    num_rollout_per_epoch: int | None,
) -> None:
    model_id = trainer.model_id

    rollout_ids_iter = (
        range(trainer.start_rollout_id, args.num_rollout) if is_leader else itertools.count(trainer.start_rollout_id)
    )
    for rollout_id in rollout_ids_iter:
        if run_ended.is_set():
            return
        rollout_ids[model_id] = rollout_id
        await inference_controller.prepare_rollout(rollout_id, model_id=model_id)
        rollout_data_pack = await rollout_executor.get(rollout_id, trainer_model_id=model_id)
        await trainer.handle.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        if is_leader:
            await _maybe_save_globally(
                args,
                model_id=model_id,
                trainers=trainers,
                rollout_executor=rollout_executor,
                parker=parker,
                rollout_ids=rollout_ids,
                rollout_id=rollout_id,
                num_rollout_per_epoch=num_rollout_per_epoch,
            )
        else:
            await parker.maybe_park_follower()

        if (rollout_id + 1) % args.update_weights_interval == 0:
            await update_weights(
                args,
                trainer.handle,
                rollout_executor,
                inference_controller,
                rollout_id=rollout_id,
                trainer_model_id=model_id,
            )

        if (
            is_leader
            and (x := args.debug_exit_after_rollout) is not None
            and (rollout_id - trainer.start_rollout_id + 1) >= x
        ):
            logger.info(f"debug_exit_after_rollout={x} reached at rollout_id={rollout_id}, exiting")
            break


async def _maybe_save_globally(
    args,
    *,
    model_id: str,
    trainers: dict[str, TrainerInfo],
    rollout_executor: BaseWorkerHandle,
    parker: Parker,
    rollout_ids: dict[str, int],
    rollout_id: int,
    num_rollout_per_epoch: int | None,
) -> None:
    external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
    if not external_save and not should_run_periodic_action(
        rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
    ):
        return

    async with parker.with_all_parked():
        await asyncio.gather(
            *(
                trainer.handle.save_model(rollout_ids[trainer_model_id], force_sync=True)
                for trainer_model_id, trainer in trainers.items()
            )
        )
        await rollout_executor.save(rollout_id)
        if args.save is not None:
            MultiPolicyCheckpointState(leader_model_id=model_id, rollout_ids=dict(rollout_ids)).save(Path(args.save))

    if external_save:
        os.remove(args.save_trigger_sentinel)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train_multi_policy(args))
    finally:
        finish_tracking()
