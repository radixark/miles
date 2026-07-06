"""Custom fully-async multi-LoRA trainer.

Drives the new controller (Ray + HTTP proxy) with no drain and no rollout-id
tracking:
  - create the controller, point rollout requests at its HTTP proxy,
  - register adapters from CLI (parse YAML -> AdapterConfig) + load into Megatron
    slots via ``actor_model.load_adapters``,
  - loop: read ``active_adapters``, reconcile (cleanup adapters no longer active),
    drain a batch from the continuous rollout, train, upsert via ``update_weights``,
  - the data source deregisters adapters at num_row; the trainer cleans them up
    (save ckpt + clear slot + free) on reconcile.

Compile-checked only; needs torch + Ray + SGLang to run.
"""

import asyncio
import logging
from pathlib import Path

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.adapter_config import parse_adapter_yaml
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking

from miles.ray.multi_lora_controller import create_controller, get_multi_lora_controller

logger = logging.getLogger(__name__)

ROLLOUT_FUNCTION_PATH = "examples.multi_lora.multi_lora_async_rollout.generate_rollout_multi_lora"
DATA_SOURCE_PATH = "examples.multi_lora.multi_lora_data_source_async.MultiLoRAAsyncDataSource"


async def main(args):
    assert not args.colocate, "Colocation is not supported for fully-async training (generation needs continuous GPU; colocate time-shares)."
    configure_logger()

    upstream_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    controller = create_controller(args, upstream_url)
    port = await controller.start.remote()
    host = await controller.http_host.remote()
    args.sglang_router_ip = host
    args.sglang_router_port = port

    args.rollout_function_path = ROLLOUT_FUNCTION_PATH
    args.data_source_path = DATA_SOURCE_PATH
    args.rollout_global_dataset = True

    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    # Register adapters from CLI. The loop's first reconcile_adapters loads them
    # into Megatron slots and the first update_weights upserts their initial
    # weights, so the first rollout uses the adapter weights. In service mode the
    # trainer idle-waits for registrations; with --multi-lora-disable-service-mode
    # it exits when no adapters are active.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_yaml(Path(path))
        await controller.register_adapter.remote(name, config)

    rollout_id = 0
    while True:
        active = await get_multi_lora_controller().active_adapters.remote()
        if not active:
            if getattr(args, "multi_lora_disable_service_mode", False):
                logger.info("No active adapters; exiting.")
                break
            await asyncio.sleep(getattr(args, "multi_lora_idle_poll_s", 5.0))
            continue

        # Reconcile (load new + cleanup gone) then upsert BEFORE generate, so the
        # batch is fetched with loaded = active (batch ⊆ loaded) and SGLang has
        # the latest weights. Reconcile runs after the previous train (gone
        # adapters' last batch already trained) and before update_weights.
        await actor_model.reconcile_adapters()
        await actor_model.update_weights()

        rollout_data = await rollout_manager.generate(rollout_id)
        await actor_model.train(rollout_id, rollout_data)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await actor_model.save_model(rollout_id, force_sync=False)

        rollout_id += 1

    await rollout_manager.dispose.remote()
    await controller.stop.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
