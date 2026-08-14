from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SessionServerConfig(FrozenStrictBaseModel):
    host: str
    port: int
    instance_id: str | None
    backend_url: str
    timeout: float | None
    hf_checkpoint: str | None
    chat_template_path: str | None
    tito_model: str
    apply_chat_template_kwargs: dict[str, Any] | None
    use_rollout_routing_replay: bool
    use_rollout_indexer_replay: bool
    sglang_speculative_algorithm: str | None
    num_layers: int | None
    moe_router_topk: int | None
    save_debug_trajectory_data: str | None
    lora_rank: int
    lora_adapter_path: str | None
    use_session_server: str | None
    session_message_matcher: str
    pause_generation_mode: str | None
    session_sample_picker_path: str | None
    session_sample_postprocessor_path: str | None


def compute_session_server_config(
    args, *, host: str, port: int, instance_id: str | None, backend_url: str
) -> SessionServerConfig:
    return SessionServerConfig(
        host=host,
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        timeout=args.miles_router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        tito_model=args.tito_model,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        use_rollout_routing_replay=args.use_rollout_routing_replay,
        use_rollout_indexer_replay=args.use_rollout_indexer_replay,
        sglang_speculative_algorithm=args.sglang_speculative_algorithm,
        num_layers=getattr(args, "num_layers", None),
        moe_router_topk=getattr(args, "moe_router_topk", None),
        save_debug_trajectory_data=args.save_debug_trajectory_data,
        lora_rank=args.lora_rank,
        lora_adapter_path=args.lora_adapter_path,
        use_session_server=getattr(args, "use_session_server", None),
        session_message_matcher=getattr(args, "session_message_matcher", "strict"),
        pause_generation_mode=getattr(args, "pause_generation_mode", None),
        session_sample_picker_path=getattr(args, "session_sample_picker_path", None),
        session_sample_postprocessor_path=getattr(args, "session_sample_postprocessor_path", None),
    )
