import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import create_trainer_handles, create_training_model, take_over_trainers
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.specs.inference import compute_router_providers, create_inference_controller_handle
from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs
from miles.ray.wiring import get_backend_capability
from miles.tinker.arguments import add_tinker_arguments, configure_tinker_args
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.hf_config import load_hf_config
from miles.utils.hot_restart import init_or_reset_inference_controller
from miles.utils.http_utils import init_http_client
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


async def serve(args, *, disposer: Disposer):
    assert args.multi_lora, "serve_tinker requires --multi-lora-n-adapters > 0"
    assert args.load == args.hf_checkpoint, "Tinker trainers and engines must load the same frozen HF base"
    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    hf_config = load_hf_config(args.hf_checkpoint)
    max_tokens_per_datum = hf_config.max_position_embeddings
    if args.max_tokens_per_gpu is not None:
        # The trainer pads each packed microbatch to this multiple.
        pad_size = args.tensor_model_parallel_size * args.data_pad_size_multiplier
        trainer_token_limit = args.max_tokens_per_gpu // pad_size * pad_size
        max_tokens_per_datum = min(max_tokens_per_datum, trainer_token_limit)
    assert max_tokens_per_datum > 0, "trainer token budget must fit at least one padding block"
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    init_http_client(args)

    capability = get_backend_capability(args)
    await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))
    inference_controller = create_inference_controller_handle(capability=capability)
    await init_or_reset_inference_controller(inference_controller, args=args)
    disposer.add(inference_controller)

    trainer_configs = compute_trainer_configs(args)
    handles = create_trainer_handles(args, trainer_configs=trainer_configs)
    resumed = await take_over_trainers(args, handles=handles)
    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    actor_info = await create_training_model(
        compute_trainer_args(args, actor_config),
        handle=handles[actor_config.trainer_id],
        trainer_id=actor_config.trainer_id,
        resumed=resumed,
    )
    trainer = actor_info.handle
    disposer.add(trainer)

    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        n_slots=args.multi_lora_n_adapters,
        checkpoint_root=checkpoint_root,
        vocab_size=hf_config.vocab_size,
        max_tokens_per_datum=max_tokens_per_datum,
        lora_alpha=args.lora_alpha,
        max_lora_rank=args.lora_rank,
        trains_attn=args.tinker_train_attn,
        trains_mlp=args.tinker_train_mlp,
        trains_unembed=args.tinker_train_unembed,
    )
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    dp_size = actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    service = TinkerService(MilesBackend(trainer, router_url, dp_size=dp_size), config)

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(service), host=args.tinker_server_host, port=args.tinker_server_port, log_level="info"
        )
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


if __name__ == "__main__":
    args = parse_args(add_tinker_arguments, entry="serve", preprocess_args=configure_tinker_args)
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(with_disposer(serve, args))
