import ipaddress
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class Address(FrozenStrictBaseModel):
    host: str
    port: Annotated[int, Field(strict=True, ge=1, le=65535)]

    @field_validator("host")
    @classmethod
    def _validate_host(cls, host: str) -> str:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if (
                not host
                or len(host) > 253
                or any(
                    not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                    for label in host.rstrip(".").split(".")
                )
            ):
                raise ValueError("host must be a bare IP address or DNS name") from None
        else:
            if "%" in host:
                raise ValueError("scoped IP addresses are not supported")
            return address.compressed
        return host.lower().rstrip(".")

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


class EtcdDiscovery(FrozenStrictBaseModel):
    backend: Literal["etcd"] = "etcd"
    endpoints: Annotated[tuple[Address, ...], Field(min_length=1)]


class FileDiscovery(FrozenStrictBaseModel):
    backend: Literal["file"] = "file"
    root: Path

    @field_validator("root")
    @classmethod
    def _absolute_root(cls, root: Path) -> Path:
        if not root.is_absolute():
            raise ValueError("file discovery requires an absolute shared directory")
        return root


class SourcePins(FrozenStrictBaseModel):
    dynamo: Revision
    sglang: Revision
    sglang_branch: Literal["sglang-miles"] = "sglang-miles"


class LaunchOptions(FrozenStrictBaseModel):
    """Native argv tokens (no shell expansion); explicit env overrides inherited env."""

    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)


class FrontendConfig(LaunchOptions):
    # None leaves the choice to native CLI/env/default precedence.
    router_mode: (
        Literal["round-robin", "random", "power-of-two", "kv", "direct", "least-loaded", "device-aware-weighted"]
        | None
    ) = None


class SidecarConfig(LaunchOptions):
    pass


class DynamoConfig(FrozenStrictBaseModel):
    request_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 600.0
    control_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 120.0
    namespace: Name
    discovery: Annotated[EtcdDiscovery | FileDiscovery, Field(discriminator="backend")]
    pins: SourcePins
    nats_server: Annotated[str, Field(min_length=1)] = "nats://127.0.0.1:4222"
    request_plane: Literal["tcp", "nats"] = "tcp"
    response_plane: Literal["tcp", "quic"] = "tcp"
    event_plane: Literal["zmq", "nats"] = "zmq"
    env: dict[str, str] = Field(default_factory=dict)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    sidecar: SidecarConfig = Field(default_factory=SidecarConfig)

    @field_validator("namespace")
    @classmethod
    def _private_namespace(cls, value: str) -> str:
        if value == "dynamo":
            raise ValueError("namespace 'dynamo' is global; choose a run-specific namespace")
        return value


def load_dynamo_config(path: Path) -> DynamoConfig:
    return DynamoConfig.model_validate_json(path.read_text())


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
