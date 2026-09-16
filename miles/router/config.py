from miles.utils.pydantic_utils import FrozenStrictBaseModel


class MilesRouterConfig(FrozenStrictBaseModel):
    host: str
    port: int
    max_connections: int
    timeout: float | None
    health_check_interval: float
    health_check_failure_threshold: int


def compute_miles_router_config(args, *, host: str, port: int, num_engines: int) -> MilesRouterConfig:
    if args.miles_router_max_connections is not None:
        max_connections = args.miles_router_max_connections
    else:
        max_connections = args.sglang_server_concurrency * num_engines

    return MilesRouterConfig(
        host=host,
        port=port,
        max_connections=max_connections,
        timeout=args.miles_router_timeout,
        health_check_interval=args.rollout_health_check_interval,
        health_check_failure_threshold=args.miles_router_health_check_failure_threshold,
    )
