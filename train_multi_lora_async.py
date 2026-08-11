"""Fully-async multi-LoRA trainer driver."""

import asyncio
import logging
import os
from pathlib import Path

import ray

from miles.ray.multi_lora.controller import create_multilora_controller, get_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils import object_store
from miles.utils.adapter_config import parse_adapter_run_yaml
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.logging_utils import configure_logger
from miles.utils.multi_lora import EmptyBatchTimeoutError, define_new_adapter_metrics
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _is_empty_batch_timeout(task_error: ray.exceptions.RayTaskError) -> bool:
    cause = getattr(task_error, "cause", None)
    if isinstance(cause, EmptyBatchTimeoutError):
        return True
    return isinstance(task_error.as_instanceof_cause(), EmptyBatchTimeoutError)


async def main(args):
    assert (
        not args.colocate
    ), "Colocation is not supported for fully-async training (generation needs continuous GPU; colocate time-shares)."
    configure_logger(args, source=MainProcessIdentity())

    # The multi-LoRA rollout fn / data source / global dataset flags are
    # defaulted by miles_validate_args when --multi-lora-n-adapters > 0.
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    rollout_manager, _num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Create a controller nclusing MultiLoRAController and MultiLoRAHTTPServer to manage lora
    router_ip, router_port = await rollout_manager.get_router_address.remote()
    args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
    controller = create_multilora_controller(args, f"http://{router_ip}:{router_port}")
    await controller.start.remote()
    host = await controller.http_host.remote()
    api_port = await controller.api_port.remote()
    logger.info(f"Multi-LoRA control API listening on http://{host}:{api_port} (head node)")

    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    tinker_server = None
    tinker_task = None
    if os.environ.get("MILES_TINKER_API") == "1":
        import uvicorn

        from miles.tinker.api import create_app
        from miles.tinker.backend import MilesTinkerBackend

        tinker_backend = MilesTinkerBackend(
            args,
            actor_model,
            controller,
            f"http://{router_ip}:{router_port}",
        )
        app = create_app(
            tinker_backend,
            os.environ.get("MILES_TINKER_BASE_MODEL", args.hf_checkpoint),
            max_lora_rank=args.lora_rank,
        )
        tinker_port = int(os.environ.get("MILES_TINKER_API_PORT", "8000"))
        tinker_server = uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=tinker_port, log_level="info", access_log=False)
        )
        tinker_task = asyncio.create_task(tinker_server.serve())
        while not tinker_server.started:
            if tinker_task.done():
                tinker_task.result()
                raise RuntimeError("Miles Tinker API exited before startup")
            await asyncio.sleep(0.05)
        logger.info(f"Tinker-compatible API listening on http://{host}:{tinker_port}")

    # CLI-registered adapters are loaded and pushed by the loop's first
    # reconcile + update_weights.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_run_yaml(Path(path))
        await controller.register_adapter.remote(name, config)

    # Tinker owns the training cadence in request-driven mode.  Entering the
    # autonomous rollout loop here would try to construct Miles datasets for
    # externally registered adapters (which intentionally have no prompt
    # path) and would also race Tinker's split forward/backward + optimizer
    # calls.  Keep the Ray actors alive until the API server is stopped; the
    # backend methods perform their own adapter reconciliation and weight push.
    if tinker_task is not None:
        try:
            await tinker_task
        finally:
            await rollout_manager.dispose.remote()
            await controller.stop.remote()
        return

    rollout_id = 0
    while True:
        snapshot = await get_multi_lora_controller().snapshot.remote()

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
        await actor_model.update_weights()

        # With nothing active, generate would wait forever.
        post_update = await get_multi_lora_controller().snapshot.remote()
        if not (post_update["active"] or post_update["retiring"]):
            continue

        try:
            rollout_data = await rollout_manager.generate.remote(rollout_id)
        except ray.exceptions.RayTaskError as e:
            if _is_empty_batch_timeout(e):
                logger.warning(f"Generate timed out with no trainable groups; retrying reconcile/update. {e}")
                continue
            raise
        await actor_model.train(rollout_id, rollout_data)
        remove_rollout_data_refs(args, rollout_data)

        # Per-adapter save cadence decided inside save_model.
        await actor_model.save_model(rollout_id)

        rollout_id += 1

    await rollout_manager.dispose.remote()
    if tinker_server is not None:
        tinker_server.should_exit = True
        await tinker_task
    await controller.stop.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
