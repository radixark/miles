import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.backends.megatron_utils.lora.slot_capacity import (
    AUTO_SLOT_CAPACITY,
    PROBE_SLOTS,
    probe_slot_capacity,
    resolve_slot_capacity,
)
from miles.backends.megatron_utils.lora.utils import convert_target_modules_to_megatron
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.train.group import TrainerController
from miles.ray.wiring import launch_worker_manager
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


async def serve(args):
    assert args.multi_lora, "serve_tinker requires --multi-lora-n-adapters (a count, or 'auto')"
    configure_logger(args, source=MainProcessIdentity())

    init_http_client(args)

    auto_capacity = args.multi_lora_n_adapters == AUTO_SLOT_CAPACITY
    if auto_capacity:
        # the trainer sizes its slot pool at construction: probe with one slot, rebuild at the measured count
        args.multi_lora_n_adapters = PROBE_SLOTS
    worker_manager = launch_worker_manager(args, trainer_only=auto_capacity)
    object_store.init_instance(args, contribute_segment=False)

    inference_controller = InferenceController(args)
    if auto_capacity:
        probe_trainer = _trainer_controller(args, inference_controller)
        await probe_trainer.init()
        # the probe only trains; the router address exists once the engines launch
        dp_size = _data_parallel_size(args)
        probe_backend = MilesBackend(probe_trainer, router_url="", dp_size=dp_size)
        probes = await probe_slot_capacity(args, probe_backend, probe_trainer, dp_size)
        args.multi_lora_n_adapters = resolve_slot_capacity(args, probes)
        await probe_trainer.dispose()
        # fresh worker processes rebuild the pool at the resolved size; the engine specs read it from args
        await worker_manager.restart_with_specs.remote(compute_specs(args))
    await inference_controller.init()

    trainer = _trainer_controller(args, inference_controller)
    await trainer.init()

    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    # args.target_modules holds HF names (q_proj, ...); classify on the megatron names
    target_modules = set(convert_target_modules_to_megatron(args.target_modules or ()))
    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        n_slots=args.multi_lora_n_adapters,
        checkpoint_root=checkpoint_root,
        lora_alpha=args.lora_alpha,
        max_lora_rank=args.lora_rank,
        trains_attn=bool(target_modules & {"linear_qkv", "linear_proj"}),
        trains_mlp=bool(target_modules & {"linear_fc1", "linear_fc2"}),
        trains_unembed="output_layer" in target_modules,
    )
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    service = TinkerService(MilesBackend(trainer, router_url, dp_size=_data_parallel_size(args)), config)

    server = uvicorn.Server(
        uvicorn.Config(build_app(service), host="0.0.0.0", port=args.tinker_server_port, log_level="info")
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    # supervise both: a crashed dispatcher must take the HTTP server down with it,
    # not keep answering /healthz while every training future pends forever
    service_task = asyncio.create_task(service.run())
    server_task = asyncio.create_task(server.serve())
    try:
        done, _ = await asyncio.wait({service_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (service_task, server_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _trainer_controller(args, inference_controller) -> TrainerController:
    return TrainerController(
        args=args,
        role="actor",
        with_ref=False,
        with_opd_teacher=False,
        inference_controller=inference_controller,
        rollout_executor=None,
    )


def _data_parallel_size(args) -> int:
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    return actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )


if __name__ == "__main__":
    args = parse_args(entry="serve")
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(serve(args))
