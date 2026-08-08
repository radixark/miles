"""Driver for the tinker-compatible backend.

One loop, two phases. The CONTROL phase claims data-less operations
(optim_step, save_weights_for_sampler, save_state, load_state) — at most one
per adapter, in strict per-registration order — executes them on every
training rank, pushes any staged weights, and only then completes deferred
publishes (the publish barrier: a save_weights_for_sampler result is visible
strictly after its weights are live on the engines). The DATA phase runs
generate/train over whole client batches; an empty-queue timeout is a yield
back to the control phase, not an error.
"""

import asyncio
import logging
from pathlib import Path

import ray

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.tinker_backend.config import parse_adapter_run_yaml
from miles.ray.tinker_backend.controller import create_tinker_controller
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.logging_utils import configure_logger
from miles.utils.tinker_backend import EmptyBatchTimeoutError
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _is_empty_batch_timeout(task_error: ray.exceptions.RayTaskError) -> bool:
    cause = getattr(task_error, "cause", None)
    if isinstance(cause, EmptyBatchTimeoutError):
        return True
    return isinstance(task_error.as_instanceof_cause(), EmptyBatchTimeoutError)


async def run_control_phase(actor_model, controller) -> None:
    """Claim → execute → complete, with the publish barrier in the middle."""
    operations = await controller.claim_ready_control_operations.remote()
    deferred: list[str] = []
    if operations:
        results = await actor_model.execute_tinker_controls(operations)
        deferred = [op_id for op_id, outcome in results.items() if outcome.get("deferred") == "publish"]
        immediate = {op_id: outcome for op_id, outcome in results.items() if op_id not in deferred}
        if immediate:
            await controller.complete_control_operations.remote(immediate)

    # Push staged weights (publishes and load_state re-publishes); a no-op
    # when nothing is staged. Serving versions bump as the push commits.
    await actor_model.update_weights()

    if deferred:
        # The barrier held: these weights are now live, so the operations may
        # complete (the backend stamps the authoritative serving identity).
        await controller.complete_control_operations.remote({op_id: dict(ok=True, result={}) for op_id in deferred})


async def main(args):
    assert (
        not args.colocate
    ), "Colocation is not supported for the tinker backend (generation needs continuous GPU; colocate time-shares)."
    configure_logger(args, source=MainProcessIdentity())

    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    rollout_manager, _num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    router_ip, router_port = await rollout_manager.get_router_address.remote()
    args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
    controller = create_tinker_controller(args, f"http://{router_ip}:{router_port}")
    await controller.start.remote()
    host = await controller.http_host.remote()
    api_port = await controller.api_port.remote()
    logger.info(f"Tinker control API listening on http://{host}:{api_port} (head node)")

    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    # CLI-registered adapters; loaded and marked READY by the first reconcile.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_run_yaml(Path(path))
        await controller.register_adapter.remote(name, config)

    rollout_id = 0
    while True:
        # The handle from create_tinker_controller is the actor's only owning
        # reference (it is not detached): rebinding it — e.g. to the weak
        # ray.get_actor handle — would let Ray reap the controller mid-run.
        snapshot = await controller.snapshot.remote()
        if not (snapshot["pending"] or snapshot["ready"] or snapshot["retiring"] or snapshot["cleanup"]):
            if not args.multi_lora_service_mode:
                logger.info("No adapters; exiting.")
                break
            logger.info(f"No adapters; sleeping for {args.multi_lora_idle_poll_s}s...")
            await asyncio.sleep(args.multi_lora_idle_poll_s)
            continue

        # Residency first: retire deregistered adapters (final states), then
        # load bound registrations and open their READY gates.
        await actor_model.reconcile_tinker_adapters()

        await run_control_phase(actor_model, controller)

        post_control = await controller.snapshot.remote()
        if not post_control["ready"]:
            continue

        try:
            rollout_data = await rollout_manager.generate.remote(rollout_id)
        except ray.exceptions.RayTaskError as e:
            if _is_empty_batch_timeout(e):
                # The data queue is idle; loop back to the control phase so
                # queued optim/save/load operations never wait behind it.
                continue
            raise
        await actor_model.train(rollout_id, rollout_data)
        remove_rollout_data_refs(args, rollout_data)
        rollout_id += 1

    await rollout_manager.dispose.remote()
    await controller.stop.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
