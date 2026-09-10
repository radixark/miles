import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.backends.megatron_utils.lora.utils import convert_target_modules_to_megatron
from miles.ray.rollout.inference_controller import InferenceController
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
    assert args.multi_lora, "serve_tinker requires --multi-lora-n-adapters > 0"
    configure_logger(args, source=MainProcessIdentity())

    init_http_client(args)

    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)

    inference_controller = InferenceController(args)
    await inference_controller.init()

    trainer = TrainerController(
        args=args,
        role="actor",
        with_ref=False,
        with_opd_teacher=False,
        inference_controller=inference_controller,
        rollout_executor=None,
    )
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
        trains_attn=bool(target_modules & {"linear_qkv", "linear_proj"}),
        trains_mlp=bool(target_modules & {"linear_fc1", "linear_fc2"}),
        trains_unembed="output_layer" in target_modules,
    )
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    dp_size = actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    service = TinkerService(MilesBackend(trainer, router_url, dp_size=dp_size), config)

    server = uvicorn.Server(
        uvicorn.Config(build_app(service), host="0.0.0.0", port=args.tinker_server_port, log_level="info")
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    service_task = asyncio.create_task(service.run())
    try:
        await server.serve()
    finally:
        # service.run() loops forever; cancel it so the gateway can exit
        service_task.cancel()
        with suppress(asyncio.CancelledError):
            await service_task


if __name__ == "__main__":
    args = parse_args(entry="serve")
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(serve(args))
