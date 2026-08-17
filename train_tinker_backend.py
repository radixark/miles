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

from miles.ray.placement_group import create_placement_groups, create_training_models
from miles.ray.rollout.components import create_rollout_components
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


class ActorGroupWeightPublisher:
    """Physical publish-barrier seam (codex-rollout-fullparameter-design-0810
    §4.7): one parameterless call that lands whatever the training actors
    staged. It carries no tinker operation IDs, no lease, and no second
    binding list — the actor keeps sole authority over pending-push
    coalescing, the has_new_engines trigger, and the resident push-set
    selection. PR #1842 integration swaps only what sits behind this call."""

    def __init__(self, actor_model) -> None:
        self._actor_model = actor_model

    async def publish_staged_weights(self) -> None:
        await self._actor_model.update_weights()


async def train_data_batch(actor_model, controller, rollout_id: int, rollout_data) -> None:
    """Dispatch one claimed data batch to the trainer and finalize it on
    abnormal outcomes.

    A NORMAL train commits rank-side (``commit_batch`` completes the batch's
    operations with their logprobs and releases the lease). Every other exit —
    a non-NORMAL ``TrainStepOutcome`` (e.g. DISCARDED_SHOULD_RETRY) or a
    raised train error — used to leave the operations CLAIMED forever and the
    lease unreleased: the SDK future never resolved. The finalizer terminal-
    fails the still-CLAIMED operations typed server and releases the lease;
    the FAILED forward_backwards stay in the ledger as poison evidence, so
    the window's possibly-partial gradients are discarded by the next
    optim_step. Retry ownership is explicit: the client resubmits as NEW
    operations."""
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


async def run_control_phase(actor_model, controller, weight_publisher) -> None:
    """Claim → execute → complete, with the publish barrier in the middle.

    The claim carries one BatchExecutionLease for the whole control batch
    (the single binding truth the trainer validates before mutating). Its
    lifecycle follows the operations' completion boundary: an immediate-only
    batch releases after its completions land; a batch with deferred
    publish/load operations holds the lease through the physical publish
    barrier and releases only after their terminal completion. Failure paths
    release in ``finally`` — a no-op under fixed residency, so nothing can
    leak either way."""
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
        await weight_publisher.publish_staged_weights()

        if deferred:
            # The barrier held: these weights are now live, so the operations may
            # complete with their original execution results (a deferred load_state
            # carries its restored step; the backend stamps a publish's
            # authoritative serving identity).
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


async def main(args):
    assert (
        not args.colocate
    ), "Colocation is not supported for the tinker backend (generation needs continuous GPU; colocate time-shares)."
    configure_logger(args, source=MainProcessIdentity())

    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    # Role-separated views over the (currently combined) rollout plane: the
    # inference controller owns the router/engines, the rollout executor runs
    # operation batches. PR #1842 swaps only the factory's construction.
    rollout_components = create_rollout_components(args, pgs["rollout"])
    inference_controller = rollout_components.inference_controller
    rollout_executor = rollout_components.rollout_executor

    inference_endpoint = await inference_controller.get_inference_endpoint()
    args.sglang_router_ip, args.sglang_router_port = inference_endpoint.host, inference_endpoint.port
    controller = create_tinker_controller(args, inference_endpoint.base_url)
    await controller.start.remote()
    host = await controller.http_host.remote()
    api_port = await controller.api_port.remote()
    logger.info(f"Tinker control API listening on http://{host}:{api_port} (head node)")

    # Engine/weight-update plumbing wires the factory's opaque weight-update
    # owner into the training actors; the driver never reaches through the
    # controller role for it.
    actor_model, _ = await create_training_models(args, pgs, rollout_components.weight_update_owner)
    weight_publisher = ActorGroupWeightPublisher(actor_model)

    # CLI-registered adapters; loaded and marked READY by the first reconcile.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_run_yaml(Path(path))
        await controller.register_adapter.remote(name, config)

    # The trainer exists and the driver loop is about to run: flip readiness
    # so /api/v1/healthz stops answering 503 (liveness /health was up earlier,
    # but a probe must never see "ok" while trainer init can still fail).
    await controller.set_trainer_ready.remote()

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

        await run_control_phase(actor_model, controller, weight_publisher)

        post_control = await controller.snapshot.remote()
        if not post_control["ready"]:
            continue

        # Per-rollout engine preparation (the PR #1842 controller boundary):
        # a no-op behind today's combined manager, the real health/prepare
        # step once the split controller lands.
        await inference_controller.prepare_rollout(rollout_id)
        try:
            rollout_data = await rollout_executor.generate(rollout_id)
        except ray.exceptions.RayTaskError as e:
            if _is_empty_batch_timeout(e):
                # The data queue is idle; loop back to the control phase so
                # queued optim/save/load operations never wait behind it.
                continue
            raise
        await train_data_batch(actor_model, controller, rollout_id, rollout_data)
        remove_rollout_data_refs(args, rollout_data)
        rollout_id += 1

    await rollout_components.dispose()
    await controller.stop.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
