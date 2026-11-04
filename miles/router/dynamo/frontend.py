from collections.abc import Mapping, Sequence

from miles.router.dynamo.config import Address, DynamoConfig, EtcdDiscovery, FileDiscovery, LaunchOptions


def frontend_launch(
    config: DynamoConfig,
    *,
    address: Address | None = None,
    interpreter_prefix: Sequence[str],
    inherited_env: Mapping[str, str],
) -> tuple[list[str], dict[str, str]]:
    if not interpreter_prefix:
        raise ValueError("frontend requires a Python interpreter")
    if address is None:
        raise ValueError("frontend requires a configured or allocated address")
    managed = {
        "--http-host": address.host,
        "--http-port": str(address.port),
        "--namespace": config.namespace,
        # Frontend startup must not depend on worker discovery or initial weight sync.
        "--router-min-initial-workers": "0",
        "--discovery-backend": config.discovery.backend,
        "--request-plane": config.request_plane,
        "--response-plane": config.response_plane,
        "--event-plane": config.event_plane,
    }
    if config.frontend.router_mode is not None:
        managed["--router-mode"] = config.frontend.router_mode
    argv = [*interpreter_prefix, "-m", "dynamo.frontend"]
    argv.extend(launch_args(config.frontend, managed=managed, reserved=("--namespace-prefix", "--interactive")))
    env = frontend_env(config, inherited_env=inherited_env, address=address)
    return argv, env


def frontend_env(
    config: DynamoConfig,
    *,
    inherited_env: Mapping[str, str],
    address: Address | None = None,
) -> dict[str, str]:
    managed = {
        "DYN_ROUTER_MIN_INITIAL_WORKERS": "0",
        "DYN_SGLANG_ENABLE_GENERATE": "1",
        "DYN_HTTP_SVC_SGLANG_GENERATE_PATH": "/generate",
        "DYN_INTERACTIVE": "false",
    }
    if config.frontend.router_mode is not None:
        managed["DYN_ROUTER_MODE"] = config.frontend.router_mode
    if address is not None:
        managed.update(DYN_HTTP_HOST=address.host, DYN_HTTP_PORT=str(address.port))
    env = runtime_env(config, inherited_env=inherited_env, options=config.frontend, managed_env=managed)
    if address is None:
        # Worker env is built before port allocation. The later launch callback
        # validates explicit listener settings and supplies the resolved CLI args.
        env.pop("DYN_HTTP_HOST", None)
        env.pop("DYN_HTTP_PORT", None)
    return env


def runtime_env(
    config: DynamoConfig,
    *,
    inherited_env: Mapping[str, str],
    options: LaunchOptions | None = None,
    managed_env: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    managed = dict(
        DYN_NAMESPACE=config.namespace,
        DYN_DISCOVERY_BACKEND=config.discovery.backend,
        DYN_REQUEST_PLANE=config.request_plane,
        DYN_RESPONSE_PLANE=config.response_plane,
        DYN_EVENT_PLANE=config.event_plane,
        NATS_SERVER=config.nats_server,
        DYN_NAMESPACE_PREFIX=None,
        DYN_NAMESPACE_WORKER_SUFFIX=None,
        ETCD_ENDPOINTS=None,
        DYN_FILE_KV=None,
    )
    match config.discovery:
        case EtcdDiscovery(endpoints=endpoints):
            managed["ETCD_ENDPOINTS"] = ",".join(address.url for address in endpoints)
        case FileDiscovery(root=root):
            managed["DYN_FILE_KV"] = str(root)
    managed.update(managed_env or {})
    explicit = {**config.env, **(options.env if options is not None else {})}
    for key, value in explicit.items():
        if key in managed and value != managed[key]:
            raise ValueError(f"{key} conflicts with the resolved Dynamo launch configuration")
    env = {**inherited_env, **explicit}
    # Run-owned identity/protocol values replace inherited shell settings. All other
    # env survives; explicitly conflicting config values above are errors, not ignored.
    for key, value in managed.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def launch_args(
    options: LaunchOptions, *, managed: Mapping[str, str | None], reserved: Sequence[str] = ()
) -> list[str]:
    """Keep native flags extensible without allowing a second source for owned settings."""
    protected = {*managed, *reserved, "--help", "--version"}
    for token in options.extra_args:
        flag = token.partition("=")[0]
        canonical = "--" + flag[5:] if flag.startswith("--no-") else flag
        if flag in ("--", "-h", "-i") or (
            flag.startswith("--") and any(name.startswith(canonical) for name in protected)
        ):
            raise ValueError(f"{flag} is managed by Miles; use its configuration field instead")
    argv = []
    for flag, value in managed.items():
        argv.append(flag)
        if value is not None:
            argv.append(value)
    return [*argv, *options.extra_args]
