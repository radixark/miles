"""Fully-async multi-LoRA trainer driver."""

import asyncio
import logging
from pathlib import Path

from miles.ray.multi_lora.controller import get_multi_lora_controller
from miles.ray.placement_group import create_rollout_components, create_training_models, update_weights
from miles.ray.wiring import launch_worker_manager
from miles.utils import object_store
from miles.utils.adapter_config import parse_adapter_run_yaml
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.logging_utils import configure_logger
from miles.utils.multi_lora import define_new_adapter_metrics
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


async def main(args):
    assert (
        not args.colocate
    ), "Colocation is not supported for fully-async training (generation needs continuous GPU; colocate time-shares)."
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    maybe_start_periodic_pyspy_dump()

    # The multi-LoRA rollout fn / data source / global dataset flags are
    # defaulted by miles_validate_args when --multi-lora-n-adapters > 0.
    init_tracking(args)
    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)
    inference_controller, rollout_executor, _num_rollout_per_epoch = await create_rollout_components(args)

    # Create a controller nclusing MultiLoRAController and MultiLoRAHTTPServer to manage lora
    controller = get_multi_lora_controller()
    await controller.init()
    host = await controller.http_host()
    api_port = await controller.api_port()
    logger.info(f"Multi-LoRA control API listening on http://{host}:{api_port} (head node)")

    actor_model, _ = await create_training_models(args, rollout_executor)

    # CLI-registered adapters are loaded and pushed by the loop's first
    # reconcile + update_weights.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_run_yaml(Path(path))
        await controller.register_adapter(name, config)

    rollout_id = 0
    while True:
        snapshot = await get_multi_lora_controller().snapshot()

        # handle dynamic metrics in tracking backend
        define_new_adapter_metrics(snapshot)
        if not (snapshot["pending"] or snapshot["active"] or snapshot["retiring"] or snapshot["cleanup"]):
            if not args.multi_lora_service_mode:
                logger.info("No adapters; exiting.")
                break
            logger.info(f"No adapters; sleeping for {args.multi_lora_idle_poll_s}s...")
            await asyncio.sleep(args.multi_lora_idle_poll_s)
            continue

        # Reconcile + push before generate: the push promotes pending adapters,
        # and only then does the data source sample them. The actor pushes only
        # stale adapter weights (newly loaded, or stepped by the last batch).
        await actor_model.reconcile_adapters()
        await update_weights(args, actor_model, rollout_executor, inference_controller)

        # With nothing active, generate would wait forever.
        post_update = await get_multi_lora_controller().snapshot()
        if not (post_update["active"] or post_update["retiring"]):
            continue

        await inference_controller.prepare_rollout(rollout_id)
        rollout_data = await rollout_executor.get(rollout_id)
        if rollout_data.empty_batch_timeout:
            logger.warning("Generate timed out with no trainable groups; retrying reconcile/update.")
            continue
        await actor_model.train(rollout_id, rollout_data)
        remove_rollout_data_refs(args, rollout_data)

        # Per-adapter save cadence decided inside save_model.
        await actor_model.save_model(rollout_id)
        # TODO: support rollout_executor.save

        rollout_id += 1

    await rollout_executor.dispose()
    await inference_controller.dispose()
    await actor_model.dispose()
    await controller.stop()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
