import asyncio
import logging

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.multi_lora_controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.adapter_config import ADAPTER_INACTIVE_STATES, ADAPTER_ROLLOUT_STATES
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


async def main(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    controller = create_multi_lora_controller(args)
    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"
    args.custom_generate_state_path = "miles.ray.multi_lora_controller.MultiLoRAGenerateState"

    # For cli adapters
    for name, path in args.multi_lora_adapters:
        await controller.register_adapter.remote(name, path)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # sync starting weights to sglang
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    async def offload_train():
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def offload_rollout():
        # Offload if need to train or if need to update adapters
        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

    async def save(rollout_id):
        await actor_model.save_model(
            rollout_id,
            force_sync=rollout_id == args.num_rollout - 1,
        )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    rollout_id = args.start_rollout_id

    # Note: in colocated, rollout is inherently tied to train (1 rollout means 1 train) --
    # In async, we should have a run_rollout to gate the rollout.
    def should_run_train(adapters):
        return any(a.state in ADAPTER_ROLLOUT_STATES for a in adapters.values())

    def should_update_adapters(adapters):
        return any(a.state in ADAPTER_INACTIVE_STATES for a in adapters.values())

    has_seen_adapters = False
    is_idle = False

    while True:
        adapters = await controller.active_adapters.remote()
        run_train = should_run_train(adapters)
        update_adapters = should_update_adapters(adapters)

        if adapters:
            has_seen_adapters = True

        # Run training
        if run_train:
            rollout_data_ref = await rollout_manager.generate.remote(rollout_id)
            await offload_rollout()

            await actor_model.train(rollout_id, rollout_data_ref)
            await controller.report_training_completed.remote(rollout_id)

        # Load/unload adapters
        if update_adapters:
            # Train already offloads the rollout
            if not run_train:
                await offload_rollout()
                await actor_model.onload()

            await actor_model.load_pending_adapters()
            await actor_model.unload_drained_adapters()

        # Both cases need to push weights
        if run_train or update_adapters:
            is_idle = False
            # For run train, at the end, update rollout id and checkpoint if needed
            if run_train:
                if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                    # NOTE: rollout_id in save doesn't do that much for multilora since each adapter
                    # tracks its own steps, but pass anyway as a dummy arg
                    await save(rollout_id)
                rollout_id += 1

            # Push the weights to sglang
            await offload_train()
            if args.offload_rollout:
                await rollout_manager.onload_weights.remote()
            await actor_model.update_weights()
            if args.offload_rollout:
                await rollout_manager.onload_kv.remote()
        # Nothing to do: either waiting for first adapter or all work is done
        else:
            if not is_idle:
                logger.info("Idle: waiting for adapters...")
                is_idle = True
            if not args.multi_lora_service_mode and has_seen_adapters:
                break
            await asyncio.sleep(5)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
