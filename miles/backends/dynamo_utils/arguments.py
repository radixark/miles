import logging
from pathlib import Path

from miles.backends.dynamo_utils.config import load_dynamo_config
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.utils.environ import use_legacy_rollout_v1

logger = logging.getLogger(__name__)


def configure_dynamo(args) -> None:
    if not getattr(args, "dynamo_config", None):
        args.dynamo_config = None
        return
    args.dynamo_config = load_dynamo_config(Path(args.dynamo_config))
    logger.info("Required Dynamo/SGLang build revisions (not runtime-verified): %s", args.dynamo_config.pins)
    if args.use_miles_router or args.rollout_external:
        raise ValueError("--dynamo-config selects a Miles-managed router and SGLang engines")
    config = resolve_sglang_config(args)
    if len(config.models) != 1 or any(
        group.worker_type != WorkerType.REGULAR for model in config.models for group in model.server_groups
    ):
        raise ValueError("Dynamo currently supports one model with aggregated SGLang engines")
    if args.eval_uses_snapshots:
        raise ValueError("Dynamo snapshot evaluation needs a separately managed router")
    if args.fully_async and args.colocate:
        raise ValueError("Dynamo fully-async colocation needs admission deferred until the driver's resume")
    if args.recompute_logprobs_via_prefill:
        raise ValueError("Dynamo native generate does not yet support batched prefill scoring")
    if args.sglang_dp_size != 1:
        raise ValueError("Dynamo managed launch currently requires SGLang dp_size=1; TP engines are supported")
    if any(group.overrides.get("dp_size", 1) != 1 for model in config.models for group in model.server_groups):
        raise ValueError("Dynamo managed launch requires per-engine dp_size=1")
    if args.sglang_router_policy in ("consistent_hashing", "manual"):
        raise ValueError("Dynamo does not consume X-SMG-Routing-Key; configure frontend.router_mode instead")
    if getattr(args, "sglang_grpc_mode", False) or getattr(args, "sglang_smg_grpc_mode", False):
        raise ValueError("Dynamo requires native gRPC alongside HTTP, not legacy gRPC mode")
    if getattr(args, "sglang_sidecar", None) not in (None, "dynamo.sglang.sidecar"):
        raise ValueError("--dynamo-config owns the SGLang sidecar provider")
    for name in ("sidecar_args", "grpc_port", "kv_events_config"):
        if getattr(args, f"sglang_{name}", None):
            raise ValueError(f"Dynamo owns --sglang-{name.replace('_', '-')}; use dynamo-config for sidecar options")
    if use_legacy_rollout_v1():
        raise ValueError("Dynamo requires the standard InferenceRolloutFn lifecycle")
