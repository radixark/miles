"""Fully-async multi-LoRA trainer driver."""

import asyncio
import logging
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
from miles.utils.multi_lora import EmptyBatchTimeoutError, define_new_adapter_metrics, serving_lora_name
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

    # CLI-registered adapters are loaded and pushed by the loop's first
    # reconcile + update_weights.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_run_yaml(Path(path))
        await controller.register_adapter.remote(name, config)

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

        # Control phase: data-less thinker operations execute every iteration —
        # including the idle paths below — so a client waiting on a step never
        # depends on another adapter generating data. publish_snapshot is the
        # exception: its barrier semantics complete only after this iteration's
        # weight push is live, with the new serving version in the result.
        control_ops = await get_multi_lora_controller().claim_ready_control_operations.remote()
        pending_publishes: list[dict] = []
        if control_ops:
            results = await actor_model.execute_adapter_controls(control_ops)
            publish_ids = {op["operation_id"] for op in control_ops if op["kind"] == "publish_snapshot"}
            deferred = {
                op_id: outcome for op_id, outcome in results.items() if op_id in publish_ids and outcome.get("ok")
            }
            immediate = {op_id: outcome for op_id, outcome in results.items() if op_id not in deferred}
            if immediate:
                await get_multi_lora_controller().complete_control_operations.remote(immediate)
            pending_publishes = [op for op in control_ops if op["operation_id"] in deferred]

        await actor_model.update_weights()

        if pending_publishes:
            post_push = await get_multi_lora_controller().snapshot.remote()
            live = {**post_push["active"], **post_push["retiring"]}
            completions = {}
            for op in pending_publishes:
                run = live.get(op["name"])
                if run is None or run.registration_id != op["registration_id"]:
                    completions[op["operation_id"]] = dict(
                        ok=False, error=f"adapter '{op['name']}' retired before the publish landed", category="user"
                    )
                else:
                    completions[op["operation_id"]] = dict(
                        ok=True,
                        result=dict(
                            serving_version=run.version,
                            serving_name=serving_lora_name(op["name"], op["registration_id"]),
                        ),
                    )
            await get_multi_lora_controller().complete_control_operations.remote(completions)

        # With nothing active, generate would wait forever.
        post_update = await get_multi_lora_controller().snapshot.remote()
        if not (post_update["active"] or post_update["retiring"]):
            continue

        try:
            rollout_data = await rollout_manager.generate.remote(rollout_id)
        except ray.exceptions.RayTaskError as e:
            if _is_empty_batch_timeout(e):
                logger.warning(f"Generate timed out with no trainable groups; retrying reconcile/update. {e}")
                # Pace the retry so an admission-empty selection cannot busy-loop.
                await asyncio.sleep(args.multi_lora_idle_poll_s)
                continue
            raise

        # Execute the selection's bind plan on every trainer rank, then commit.
        # Bind failure is fail-stop: a half-executed swap cannot be rolled back
        # in-process, so abort the reservations and let restart rebuild.
        control_metadata = rollout_data.get("control_metadata") or {}
        if bind_plan := control_metadata.get("batch_plan"):
            txn_id = control_metadata["train_txn_id"]
            try:
                await actor_model.bind_adapters(bind_plan)
                await get_multi_lora_controller().commit_bind.remote(txn_id)
            except Exception:
                logger.error("bind_adapters failed; aborting the bind transaction and terminating (fail-stop)")
                await get_multi_lora_controller().abort_bind.remote(txn_id)
                raise
        await actor_model.train(rollout_id, rollout_data)
        remove_rollout_data_refs(args, rollout_data)

        # Per-adapter save cadence decided inside save_model.
        await actor_model.save_model(rollout_id)

        rollout_id += 1

    await rollout_manager.dispose.remote()
    await controller.stop.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
