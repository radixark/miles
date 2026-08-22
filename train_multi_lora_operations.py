import asyncio
import logging

import ray

from miles.ray.multi_lora.controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_training_models
from miles.ray.rollout.components import create_rollout_components
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.logging_utils import configure_logger
from miles.utils.operation_contract import EmptyBatchTimeoutError
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _is_empty_batch_timeout(task_error: ray.exceptions.RayTaskError) -> bool:
    cause = getattr(task_error, "cause", None)
    if isinstance(cause, EmptyBatchTimeoutError):
        return True
    return isinstance(task_error.as_instanceof_cause(), EmptyBatchTimeoutError)


class ActorGroupWeightUpdater:
    def __init__(self, actor_model) -> None:
        self._actor_model = actor_model

    async def update_weights(self) -> None:
        await self._actor_model.update_weights()


async def train_data_batch(actor_model, controller, rollout_id: int, rollout_data) -> None:
    from miles.backends.megatron_utils.ft.types import TrainStepOutcome

    dispatch = rollout_data.get("tinker_dispatch") or {}
    operation_ids = list(dispatch.get("operation_ids") or [])
    lease = dispatch.get("lease")

    try:
        outcomes = await actor_model.train(rollout_id, rollout_data)
    except Exception as e:
        await controller.fail_tinker_batch.remote(
            operation_ids,
            f"train dispatch raised on the trainer: {e}; the batch did not commit and its "
            "gradient window is poisoned — resubmit the batch and optim_step again",
            lease,
        )
        raise
    outcomes = outcomes if isinstance(outcomes, list) else [outcomes]
    abnormal = sorted({str(outcome) for outcome in outcomes if outcome != TrainStepOutcome.NORMAL})
    if abnormal:
        await controller.fail_tinker_batch.remote(
            operation_ids,
            f"train step finished without committing (outcome {', '.join(abnormal)}); the batch's "
            "gradient window is poisoned — resubmit the batch and optim_step again",
            lease,
        )


async def run_control_phase(actor_model, controller, weight_updater) -> None:
    claimed = await controller.claim_ready_control_operations.remote()
    operations, lease = claimed["operations"], claimed["lease"]
    released = lease is None
    try:
        deferred: list[str] = []
        if operations:
            results = await actor_model.execute_tinker_controls(operations, lease)
            deferred = [op_id for op_id, outcome in results.items() if outcome.get("deferred") == "publish"]
            immediate = {op_id: outcome for op_id, outcome in results.items() if op_id not in deferred}
            if immediate:
                await controller.complete_control_operations.remote(immediate)
            if not deferred and not released:
                released = True
                await controller.release_batch_lease.remote(lease)

        # Push staged weights (publishes and load_state re-publishes); a no-op
        # when nothing is staged. Serving versions bump as the push commits.
        await weight_updater.update_weights()

        if deferred:
            await controller.complete_control_operations.remote(
                {
                    op_id: {key: value for key, value in results[op_id].items() if key != "deferred"}
                    for op_id in deferred
                }
            )
            released = True
            await controller.release_batch_lease.remote(lease)
    finally:
        if not released:
            await controller.release_batch_lease.remote(lease)


async def generate_with_failure_cap(rollout_executor, rollout_id: int, failure_streak: int, cap: int):
    """One tolerated generate attempt; returns (rollout_data or None, updated consecutive-failure streak)."""
    try:
        return await rollout_executor.generate(rollout_id), 0
    except ray.exceptions.RayTaskError as e:
        if _is_empty_batch_timeout(e):
            # The data queue is idle; yield to the control phase so queued optim/save/load never wait behind it.
            return None, failure_streak
        failure_streak += 1
        if failure_streak >= cap:
            raise
        # Skipping the round self-heals: failure paths restore unconsumed claims to READY for re-dispatch.
        logger.exception(
            f"[tinker] generate failed ({failure_streak} consecutive, cap {cap}); "
            f"keeping the multi-tenant service alive: {e}"
        )
        return None, failure_streak


async def main(args):
    assert (
        not args.colocate
    ), "Colocation is not supported for Multi-LoRA operations (generation needs continuous GPU; colocate time-shares)."
    configure_logger(args, source=MainProcessIdentity())

    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    rollout_components = create_rollout_components(args, pgs["rollout"])
    inference_controller = rollout_components.inference_controller
    rollout_executor = rollout_components.rollout_executor

    inference_endpoint = await inference_controller.get_inference_endpoint()
    args.sglang_router_ip, args.sglang_router_port = inference_endpoint.host, inference_endpoint.port
    multi_lora_controller = create_multi_lora_controller(args, inference_endpoint.base_url)
    await multi_lora_controller.start.remote()
    host = await multi_lora_controller.http_host.remote()
    api_port = await multi_lora_controller.api_port.remote()
    logger.info(f"Tinker control API listening on http://{host}:{api_port} (head node)")

    # As in train_async.py, actor_model is the actor RayTrainGroup, with the weight-update owner wired in.
    actor_model, _ = await create_training_models(args, pgs, rollout_components.weight_update_owner)
    weight_updater = ActorGroupWeightUpdater(actor_model)

    # The trainer is up: flip readiness so /api/v1/healthz stops answering 503.
    await multi_lora_controller.set_trainer_ready.remote()

    rollout_id = 0
    generate_failures = 0
    while True:
        # This handle is the controller's only owning reference; rebinding it would let Ray reap the actor.
        snapshot = await multi_lora_controller.snapshot.remote()
        if not (snapshot["pending"] or snapshot["ready"] or snapshot["retiring"] or snapshot["cleanup"]):
            logger.info(f"No adapters; sleeping for {args.multi_lora_idle_poll_s}s...")
            await asyncio.sleep(args.multi_lora_idle_poll_s)
            continue

        # Residency first: retire deregistered adapters, then load bound registrations.
        await actor_model.reconcile_tinker_adapters()

        await run_control_phase(actor_model, multi_lora_controller, weight_updater)

        post_control = await multi_lora_controller.snapshot.remote()
        if not post_control["ready"]:
            continue

        # Per-rollout engine preparation; a no-op behind today's combined manager.
        await inference_controller.prepare_rollout(rollout_id)
        rollout_data, generate_failures = await generate_with_failure_cap(
            rollout_executor, rollout_id, generate_failures, args.multi_lora_max_consecutive_generate_failures
        )
        if rollout_data is None:
            continue
        await train_data_batch(actor_model, multi_lora_controller, rollout_id, rollout_data)
        remove_rollout_data_refs(args, rollout_data)
        rollout_id += 1


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
