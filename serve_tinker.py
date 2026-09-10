import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import uvicorn

from miles.backends.megatron_utils.lora.slot_capacity import (
    AUTO_SLOT_CAPACITY,
    probe_slot_capacity,
    resolve_slot_capacity,
)
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


async def _init_trainer(args, inference_controller):
    trainer = TrainerController(
        args=args,
        role="actor",
        with_ref=False,
        with_opd_teacher=False,
        inference_controller=inference_controller,
        rollout_executor=None,
    )
    await trainer.init()
    return trainer


def _backend(args, trainer, router_url):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    dp_size = actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    return MilesBackend(trainer, router_url, dp_size=dp_size)


async def _probe_and_resize(args, trainer, worker_manager, checkpoint_root):
    # The probe only trains; engine initialization resolves the router address.
    probes = await probe_slot_capacity(args, _backend(args, trainer, ""), trainer)
    # Retired versions can be recovered through request-carried disk backfill.
    keep_k = 2
    args.multi_lora_n_adapters = resolve_slot_capacity(args, probes, keep_k)
    args.sglang_max_loaded_loras = keep_k * args.multi_lora_n_adapters
    report = {
        "n_slots": args.multi_lora_n_adapters,
        "keep_k": keep_k,
        "max_tokens_per_gpu": args.max_tokens_per_gpu,
        "train_memory_margin_bytes": args.train_memory_margin_bytes,
        "engine_host_lora_budget_bytes": args.engine_host_lora_budget_bytes,
        "ranks": [asdict(probe) for probe in probes],
    }
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_root, "slot-capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    await trainer.dispose()
    # Rebuild the fixed pool in fresh processes, releasing the probe's CUDA
    # context and DDP hooks. Engine specs now receive a positive slot count.
    await worker_manager.restart_with_specs.remote(compute_specs(args))


async def serve(args):
    assert args.multi_lora, "serve_tinker requires --multi-lora-n-adapters (a count, or 'auto')"
    configure_logger(args, source=MainProcessIdentity())
    init_http_client(args)

    auto_capacity = args.multi_lora_n_adapters == AUTO_SLOT_CAPACITY
    if auto_capacity:
        # Bridge and the LayerWise optimizer allocate fixed pools at construction.
        args.multi_lora_n_adapters = 1
    worker_manager = launch_worker_manager(args, trainer_only=auto_capacity)
    object_store.init_instance(args, contribute_segment=False)
    inference_controller = InferenceController(args)
    trainer = await _init_trainer(args, inference_controller)

    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    if auto_capacity:
        await _probe_and_resize(args, trainer, worker_manager, checkpoint_root)
        trainer = await _init_trainer(args, inference_controller)
    await inference_controller.init()
    backend = _backend(args, trainer, f"http://{args.sglang_router_ip}:{args.sglang_router_port}")

    target_modules = set(args.target_modules or ())
    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        n_slots=args.multi_lora_n_adapters,
        checkpoint_root=checkpoint_root,
        lora_alpha=args.lora_alpha,
        trains_attn=bool(target_modules & {"linear_qkv", "linear_proj"}),
        trains_mlp=bool(target_modules & {"linear_fc1", "linear_fc2"}),
        trains_unembed="output_layer" in target_modules,
    )
    service = TinkerService(backend, config)
    server = uvicorn.Server(
        uvicorn.Config(build_app(service), host="0.0.0.0", port=args.tinker_server_port, log_level="info")
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    service_task = asyncio.create_task(service.run())
    try:
        await server.serve()
    finally:
        service_task.cancel()
        with suppress(asyncio.CancelledError):
            await service_task


if __name__ == "__main__":
    args = parse_args(entry="serve")
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(serve(args))
