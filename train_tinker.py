"""Serve Miles training and sampling through the official Tinker SDK API."""

from __future__ import annotations

import asyncio
import logging
import os

import ray

from miles.ray.multi_lora.controller import create_multilora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.tinker.protocol import TinkerError
from miles.utils import object_store
from miles.utils.arguments import _RedactedString, parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)

_TINKER_BACKEND = "miles.ray.tinker.backend.TinkerBackend"
_TINKER_SERVER = "miles.ray.tinker.http_server.TinkerHTTPServer"


async def main(args) -> None:
    """Launch a long-running Tinker-compatible training service."""
    _apply_api_key_environment(args)
    _validate_args(args)
    _configure_megatron_batch_placeholder(args)

    args.multi_lora_backend_path = _TINKER_BACKEND
    args.multi_lora_http_server_path = _TINKER_SERVER
    # Tinker checkpoints are explicit API operations. Avoid an unrelated
    # final save when a model is unloaded.
    args.save_interval = None

    configure_logger(args, source=MainProcessIdentity())
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])

    router_ip, router_port = await rollout_manager.get_router_address.remote()
    args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
    controller = create_multilora_controller(args, f"http://{router_ip}:{router_port}")
    actor_model = None
    try:
        await controller.start.remote()
        host = await controller.http_host.remote()
        api_port = await controller.api_port.remote()
        logger.info(f"Tinker-compatible API listening on http://{host}:{api_port}")

        actor_model, _ = await create_training_models(args, pgs, rollout_manager)
        await controller.mark_external_ready.remote()
        while True:
            operation = await controller.next_external_operation.remote()
            if operation is None:
                continue
            request_id = operation["request_id"]
            try:
                if operation["kind"] == "create_model":
                    await actor_model.reconcile_adapters()
                    await actor_model.update_weights()
                    result = {"_operation_kind": "create_model"}
                elif operation["kind"] == "unload_model":
                    await actor_model.reconcile_adapters()
                    result = {"_operation_kind": "unload_model"}
                else:
                    result = await actor_model.tinker_execute(operation)
                await controller.complete_external_operation.remote(request_id, result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                cause = _unwrap_ray_error(exc)
                category = cause.category if isinstance(cause, TinkerError) else "server"
                logger.exception(f"Tinker operation {operation['kind']} failed")
                await controller.fail_external_operation.remote(request_id, str(cause), category)
                if operation["kind"] in {"create_model", "unload_model"}:
                    try:
                        await actor_model.reconcile_adapters()
                    except Exception:
                        logger.exception("Tinker adapter cleanup after failed lifecycle operation also failed")
    finally:
        if rollout_manager is not None:
            await rollout_manager.dispose.remote()
        await controller.stop.remote()


def _unwrap_ray_error(exc: Exception) -> Exception:
    if isinstance(exc, ray.exceptions.RayTaskError):
        cause = getattr(exc, "cause", None)
        if isinstance(cause, Exception):
            return cause
        unwrapped = exc.as_instanceof_cause()
        if isinstance(unwrapped, Exception):
            return unwrapped
    return exc


def _apply_api_key_environment(args) -> None:
    """Prefer an environment secret so it does not appear in process arguments."""
    if api_key := os.environ.get("TINKER_API_KEY"):
        args.tinker_api_key = _RedactedString(api_key)


def _configure_megatron_batch_placeholder(args) -> None:
    """Set an initialization-only batch shape divisible by the trainer DP size."""
    total_trainers = args.actor_num_nodes * args.actor_num_gpus_per_node
    model_parallel_size = compute_megatron_world_size_except_dp(args)
    data_parallel_size = total_trainers // model_parallel_size
    micro_batch_size = args.micro_batch_size or 1
    args.micro_batch_size = micro_batch_size
    args.global_batch_size = micro_batch_size * data_parallel_size
    # The regular Miles scheduler still initializes even though Tinker owns
    # every optimizer step. Keep its one dummy iteration internally valid.
    args.num_rollout = 1
    args.rollout_batch_size = args.global_batch_size
    args.n_samples_per_prompt = 1
    args.over_sampling_batch_size = args.rollout_batch_size
    args.num_steps_per_rollout = 1


def _validate_args(args) -> None:
    requirements = [
        (args.train_backend == "megatron", "--train-backend must be megatron"),
        (args.multi_lora_n_adapters > 0, "--multi-lora-n-adapters must be positive"),
        (not getattr(args, "indep_dp", False), "--indep-dp is not supported"),
        (not args.colocate, "trainer and rollout GPUs must be disaggregated"),
        (not args.use_fault_tolerance, "fault-tolerant trainer replicas are not supported"),
        (args.pipeline_model_parallel_size == 1, "pipeline parallel size must be 1"),
        (args.context_parallel_size == 1, "context parallel size must be 1"),
        (args.qkv_format == "thd", "--qkv-format must be thd"),
        (not args.calculate_per_token_loss, "per-token-loss mode must be disabled for sum-reduced losses"),
        (args.lora_dropout == 0.0, "--lora-dropout must be 0"),
        (args.attention_dropout == 0.0, "--attention-dropout must be 0"),
        (args.hidden_dropout == 0.0, "--hidden-dropout must be 0"),
        (args.tinker_max_concurrent_samples > 0, "--tinker-max-concurrent-samples must be positive"),
    ]
    for valid, message in requirements:
        if not valid:
            raise ValueError(f"Tinker service: {message}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
