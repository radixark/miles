import asyncio
import logging
import os

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.lora import is_lora_enabled
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


async def train(args, *, disposer: Disposer):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    if args.colocate_memory_peak_device == "gpu":
        assert (
            args.offload_train and args.offload_rollout
        ), "--colocate-memory-peak-device gpu requires --offload-train and --offload-rollout"
        assert not args.use_critic, "--colocate-memory-peak-device gpu is not wired for the critic path"
        if is_lora_enabled(args):
            raise NotImplementedError("--colocate-memory-peak-device gpu only supports full-parameter training")

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)

    maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(args, actor_model, rollout_executor, inference_controller)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await inference_controller.onload_kv()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await inference_controller.prepare_eval()
        await rollout_executor.eval(rollout_id=0)

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
        await rollout_executor.save(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            await inference_controller.prepare_eval()
            await rollout_executor.eval(rollout_id)

        await inference_controller.prepare_rollout(rollout_id)
        rollout_data_pack = await rollout_executor.get(rollout_id)

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
            remove_train_output_refs(values)
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
            await inference_controller.onload_weights()
            await offload_train()
        else:
            await offload_train()
            if args.offload_rollout:
                await inference_controller.onload_weights()
        await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
        if args.offload_rollout:
            await inference_controller.onload_kv()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await inference_controller.prepare_eval()
            await rollout_executor.eval(rollout_id)

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


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train, args))
