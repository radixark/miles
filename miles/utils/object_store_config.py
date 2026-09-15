import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MOONCAKE_MASTER_PORT = 50051
MOONCAKE_MASTER_ADDRESS_KEY = "master_server_address"

_GLOBAL_SEGMENT_SIZE_KEY = "global_segment_size"
_MISSING = object()


def compute_mooncake_init_kwargs_vanilla(*, host: str = "127.0.0.1", master_port: int = MOONCAKE_MASTER_PORT) -> dict:
    return {
        "protocol": "tcp",
        MOONCAKE_MASTER_ADDRESS_KEY: f"{host}:{master_port}",
        "global_segment_size": "2gb",
        "local_buffer_size": "2gb",
    }


def compute_mooncake_init_kwargs_from_env() -> dict:
    defaulted_init_kwargs = compute_mooncake_init_kwargs_vanilla()
    return {
        field.init_kwarg: value
        for field in _MOONCAKE_STORE_FIELDS
        if field.init_kwarg in defaulted_init_kwargs and (value := os.environ.get(field.env_var)) is not None
    }


def compute_mooncake_store_config(init_kwargs: dict[str, Any], *, contribute_segment: bool) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for field in _MOONCAKE_STORE_FIELDS:
        if field.config_key == _GLOBAL_SEGMENT_SIZE_KEY and not contribute_segment:
            config[field.config_key] = 0
        else:
            config[field.config_key] = field.parse(_resolve_field(field, init_kwargs=init_kwargs))
    return config


@dataclass(frozen=True)
class _MooncakeStoreField:
    init_kwarg: str
    config_key: str
    env_var: str
    parse: Callable[[Any], Any]
    default: Any = None
    default_factory: Callable[[], Any] | None = None
    treats_falsy_as_unset: bool = True


def _resolve_field(field: _MooncakeStoreField, *, init_kwargs: dict[str, Any]) -> Any:
    for value in (init_kwargs.get(field.init_kwarg, _MISSING), os.environ.get(field.env_var, _MISSING)):
        if value is not _MISSING and (value or not field.treats_falsy_as_unset):
            return value
    if (default_factory := field.default_factory) is not None:
        return default_factory()
    return field.default


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    units = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "k": 1024, "m": 1024**2, "g": 1024**3}
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def _local_hostname() -> str:
    import ray

    return ray.util.get_node_ip_address()


_MOONCAKE_STORE_FIELDS: tuple[_MooncakeStoreField, ...] = (
    _MooncakeStoreField(
        init_kwarg="local_hostname",
        config_key="local_hostname",
        env_var="MOONCAKE_LOCAL_HOSTNAME",
        parse=str,
        default_factory=lambda: _local_hostname(),
    ),
    _MooncakeStoreField(
        init_kwarg="metadata_server",
        config_key="metadata_server",
        env_var="MOONCAKE_TE_META_DATA_SERVER",
        parse=str,
        default="P2PHANDSHAKE",
    ),
    _MooncakeStoreField(
        init_kwarg="local_buffer_size",
        config_key="local_buffer_size",
        env_var="MOONCAKE_LOCAL_BUFFER_SIZE",
        parse=_parse_size,
        default=32 * 1024**3,
        treats_falsy_as_unset=False,
    ),
    _MooncakeStoreField(
        init_kwarg="protocol",
        config_key="protocol",
        env_var="MOONCAKE_PROTOCOL",
        parse=str,
        default="rdma",
    ),
    _MooncakeStoreField(
        init_kwarg="device_name",
        config_key="rdma_devices",
        env_var="MOONCAKE_DEVICE",
        parse=str,
        default="",
    ),
    _MooncakeStoreField(
        init_kwarg=MOONCAKE_MASTER_ADDRESS_KEY,
        config_key="master_server_addr",
        env_var="MOONCAKE_MASTER",
        parse=str,
        default="",
    ),
    _MooncakeStoreField(
        init_kwarg=_GLOBAL_SEGMENT_SIZE_KEY,
        config_key=_GLOBAL_SEGMENT_SIZE_KEY,
        env_var="MOONCAKE_GLOBAL_SEGMENT_SIZE",
        parse=_parse_size,
        default=8 * 1024**3,
        treats_falsy_as_unset=False,
    ),
)
