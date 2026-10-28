import logging
from typing import Any

from miles.router.backend import make_router_backend, request_router_model
from miles.utils.output_store.config import output_store_backend_config
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class SessionServerConfig(FrozenStrictBaseModel):
    host: str
    port: int
    instance_id: str | None
    backend_url: str
    # header the router places a session by; None = turns are not pinned to one engine
    routing_key_header: str | None
    timeout: float | None
    hf_checkpoint: str | None
    chat_template_path: str | None
    tito_model: str
    apply_chat_template_kwargs: dict[str, Any] | None
    use_rollout_routing_replay: bool
    use_rollout_indexer_replay: bool
    rollout_indexer_topk_num_streams: int | None
    use_sampling_support_replay: bool
    sglang_speculative_algorithm: str | None
    num_layers: int | None
    moe_router_topk: int | None
    save_debug_trajectory_data: str | None
    lora_rank: int
    lora_adapter_path: str | None
    lora_train_only: bool
    use_session_server: bool | str | None
    session_message_matcher: str
    pause_generation_mode: str | None
    session_sample_picker_path: str | None
    session_sample_postprocessor_path: str | None
    rollout_top_logprobs_num: int
    rollout_sampling_logprobs_mode: str
    rollout_temperature: float
    # The run's output store with --use-output-store, else None; the server resolves its own local_hostname.
    output_store_backend_config: dict[str, Any] | None


def compute_session_server_config(
    args, *, host: str, port: int, instance_id: str | None, backend_url: str
) -> SessionServerConfig:
    return SessionServerConfig(
        host=host,
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        routing_key_header=_routing_key_header(args),
        timeout=args.miles_router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        tito_model=args.tito_model,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        use_rollout_routing_replay=args.use_rollout_routing_replay,
        use_rollout_indexer_replay=args.use_rollout_indexer_replay,
        rollout_indexer_topk_num_streams=args.rollout_indexer_topk_num_streams,
        use_sampling_support_replay=args.use_sampling_support_replay,
        sglang_speculative_algorithm=args.sglang.common_value("speculative_algorithm"),
        num_layers=args.num_layers,
        moe_router_topk=args.raw_megatron.base_args.get("moe_router_topk"),
        save_debug_trajectory_data=args.save_debug_trajectory_data,
        lora_rank=args.lora_rank,
        lora_adapter_path=args.lora_adapter_path,
        lora_train_only=args.lora_train_only,
        use_session_server=args.use_session_server,
        session_message_matcher=args.session_message_matcher,
        pause_generation_mode=args.pause_generation_mode,
        session_sample_picker_path=args.session_sample_picker_path,
        session_sample_postprocessor_path=args.session_sample_postprocessor_path,
        rollout_top_logprobs_num=args.rollout_top_logprobs_num,
        rollout_sampling_logprobs_mode=args.rollout_sampling_logprobs_mode,
        rollout_temperature=getattr(
            args, "rollout_temperature", 1.0
        ),  # config-access-exempt: legacy session inputs omit rollout_temperature
        output_store_backend_config=(output_store_backend_config(args) if args.use_output_store else None),
    )


def _routing_key_header(args) -> str | None:
    backend = make_router_backend(args)
    if not backend.routes_by_key(request_router_model(args)):
        # affinity only saves prefix-cache work, so a router without it is allowed
        logger.warning(
            "--use-session-server with a router that does not route by session: "
            "a session's turns may land on different engines and lose prefix-cache reuse"
        )
        return None
    return backend.features.routing_key_header
