import ipaddress
import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

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
    nats_server: Annotated[str, Field(min_length=1)] | None = None
    request_plane: Literal["tcp", "nats"] = "tcp"
    response_plane: Literal["tcp", "quic"] = "tcp"
    event_plane: Literal["zmq", "nats"] = "zmq"
    env: dict[str, str] = Field(default_factory=dict)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    sidecar: SidecarConfig = Field(default_factory=SidecarConfig)

    @model_validator(mode="after")
    def _require_nats_server(self) -> Self:
        if (self.request_plane == "nats" or self.event_plane == "nats") and self.nats_server is None:
            raise ValueError("nats_server is required when request_plane or event_plane is nats")
        return self

    @field_validator("namespace")
    @classmethod
    def _private_namespace(cls, value: str) -> str:
        if value == "dynamo":
            raise ValueError("namespace 'dynamo' is global; choose a run-specific namespace")
        return value


def load_dynamo_config(path: Path) -> DynamoConfig:
    return DynamoConfig.model_validate_json(path.read_text())
