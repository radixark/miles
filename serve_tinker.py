import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.train.group import TrainerController
from miles.ray.wiring import launch_worker_manager
from miles.tinker.arguments import add_tinker_arguments, configure_tinker_args
from miles.tinker.core.service import TinkerService
from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.core.types import GatewayConfig
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app
from miles.tinker.server.oai_routes import install_session_routes
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.hf_config import load_hf_config
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

_SWEEP_INTERVAL_S = 60.0


def _build_collector(args, service: TinkerService) -> TrajectoryCollector:
    """The collector over the running service: loads the HF tokenizer, TTL from the flag, optional TITOTokenizer."""
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    tito_tokenizer = None
    if args.tinker_tito_model is not None:
        tito_tokenizer = get_tito_tokenizer(
            tokenizer, args.tinker_tito_model, chat_template_kwargs=args.apply_chat_template_kwargs
        )
    return TrajectoryCollector(
        service,
        tokenizer,
        session_ttl_s=args.tinker_session_ttl_s,
        chat_template_kwargs=args.apply_chat_template_kwargs,
        tito_tokenizer=tito_tokenizer,
    )


async def _sweep_collector(collector: TrajectoryCollector, interval_s: float) -> None:
    """Every interval_s drop recorded sessions idle past their TTL; lives and dies with service.run()."""
    while True:
        await asyncio.sleep(interval_s)
        if dropped := collector.sweep():
            logger.info(f"swept {dropped} idle recorded session(s)")


async def serve(args):
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
        inference_controller=None,
        rollout_executor=None,
    )
    await trainer.init()

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
    # --tinker-session-server off: no tokenizer, no /oai routes, no sweep; the gateway behaves exactly as before
    collector = _build_collector(args, service) if args.tinker_session_server else None

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(service), host=args.tinker_server_host, port=args.tinker_server_port, log_level="info"
        )
    )
    if collector is not None:
        # the four /oai/sessions routes ride on the app uvicorn holds; the Tinker routes are untouched
        install_session_routes(server.config.app, collector)
    logger.info(
        f"tinker gateway serving {config.base_model} on :{args.tinker_server_port} (session server {'on' if collector else 'off'})"
    )
    # supervise both: a crashed dispatcher must take the HTTP server down with it,
    # not keep answering /healthz while every training future pends forever
    service_task = asyncio.create_task(service.run())
    if collector is not None:
        # the sweep lives exactly as long as the dispatcher; nothing else needs to know about it
        sweep_task = asyncio.create_task(_sweep_collector(collector, _SWEEP_INTERVAL_S))
        service_task.add_done_callback(lambda _: sweep_task.cancel())
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

    await inference_controller.dispose()
    await trainer.dispose()


if __name__ == "__main__":
    args = parse_args(add_tinker_arguments, entry="serve", preprocess_args=configure_tinker_args)
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(serve(args))
