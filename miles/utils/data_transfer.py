"""Thin rollout data transport helpers."""

import os
from functools import cache
from typing import Any

import ray

from miles.utils.ray_utils import Box

ROLLOUT_DATA_TRANSPORT_OBJECT_STORE = "object-store"
ROLLOUT_DATA_TRANSPORT_MOONCAKE = "mooncake"
ROLLOUT_DATA_TRANSPORT_CHOICES = (
    ROLLOUT_DATA_TRANSPORT_OBJECT_STORE,
    ROLLOUT_DATA_TRANSPORT_MOONCAKE,
)

_MOONCAKE_IMPORT_ERROR: ImportError | None = None

try:
    from mooncake.store import MooncakeDistributedStore
    from mooncake.structured_object_store import FieldSchema, MooncakeBundleTransfer, export_ref, import_ref

    _MOONCAKE_AVAILABLE = True
except ImportError as exc:
    _MOONCAKE_AVAILABLE = False
    _MOONCAKE_IMPORT_ERROR = exc
    FieldSchema = None


def get_rollout_data_transport(args: Any) -> str:
    transport = getattr(args, "rollout_data_transport", ROLLOUT_DATA_TRANSPORT_OBJECT_STORE)
    if transport not in ROLLOUT_DATA_TRANSPORT_CHOICES:
        raise ValueError(f"Unsupported rollout data transport: {transport!r}")
    return transport


def is_mooncake_rollout_data_transport(args: Any) -> bool:
    return get_rollout_data_transport(args) == ROLLOUT_DATA_TRANSPORT_MOONCAKE


def _check_mooncake_available_if_needed(args: Any) -> None:
    if is_mooncake_rollout_data_transport(args):
        check_mooncake_available()


def check_mooncake_available() -> None:
    if not _MOONCAKE_AVAILABLE:
        raise ImportError("rollout-data-transport='mooncake' requires the mooncake package") from _MOONCAKE_IMPORT_ERROR


def validate_rollout_data_transport(args: Any) -> None:
    _check_mooncake_available_if_needed(args)


def put_rollout_data_ref(
    args: Any,
    data: dict[str, Any],
    *,
    partition: str,
    field_schema_specs: dict[str, tuple] | None = None,
) -> Box:
    _check_mooncake_available_if_needed(args)
    if is_mooncake_rollout_data_transport(args):
        return _put_mooncake_rollout_data(
            args,
            data,
            partition=partition,
            field_schemas=_rollout_field_schemas_for_data(data, field_schema_specs),
        )
    return Box(ray.put(data))


def get_rollout_data_ref(args: Any, ref: Box) -> dict[str, Any]:
    _check_mooncake_available_if_needed(args)
    if is_mooncake_rollout_data_transport(args):
        return _mooncake_transfer(args, contribute_segment=_should_contribute_segment()).get_legacy_dict(
            import_ref(ref.inner)
        )
    return ray.get(ref.inner)


def release_rollout_data(args: Any, data: dict[str, Any]) -> None:
    _check_mooncake_available_if_needed(args)
    if is_mooncake_rollout_data_transport(args):
        MooncakeBundleTransfer.release_result(data)


def cleanup_rollout_data_refs(args: Any, refs: Any) -> None:
    _check_mooncake_available_if_needed(args)
    if not is_mooncake_rollout_data_transport(args):
        return
    if isinstance(refs, dict) and "data_ref" in refs:
        refs = refs["data_ref"]
    if isinstance(refs, Box):
        _cleanup_mooncake_rollout_data(args, refs)
        return
    for ref in refs:
        _cleanup_mooncake_rollout_data(args, ref)


def _put_mooncake_rollout_data(
    args: Any,
    data: dict[str, Any],
    partition: str,
    field_schemas: dict | None = None,
) -> Box:
    config = getattr(args, "mooncake_store_init_kwargs", None) or {}
    ref = _mooncake_transfer(args, contribute_segment=True).put_legacy_dict(
        data,
        namespace="miles",
        partition=partition,
        stage="rollout",
        chunk_bytes=config.get("chunk_bytes"),
        field_schemas=field_schemas,
    )
    return Box(export_ref(ref))


def _cleanup_mooncake_rollout_data(args: Any, ref: Box) -> None:
    _mooncake_transfer(args, contribute_segment=False).remove_legacy_dict(import_ref(ref.inner))


def _rollout_field_schemas_for_data(data: dict[str, Any], field_schema_specs: dict[str, tuple] | None) -> dict | None:
    if field_schema_specs is None:
        return None
    if FieldSchema is None:
        check_mooncake_available()
        raise ImportError("rollout-data-transport='mooncake' requires mooncake.structured_object_store.FieldSchema")

    schemas = {}
    for field, spec in field_schema_specs.items():
        if field not in data:
            continue
        codec, dtype, section = (*spec, "non_tensor_batch")[:3]
        metadata = {"section": section}
        if dtype:
            metadata["dtype"] = dtype
        schemas[field] = FieldSchema(codec=codec, nullable=False, metadata=metadata)
    return schemas


def _should_contribute_segment() -> bool:
    local_rank = os.getenv("LOCAL_RANK")
    return local_rank is None or int(local_rank) == 0


def _mooncake_transfer(args: Any, contribute_segment: bool):
    return _mooncake_transfer_cached(tuple(sorted(_mooncake_store_config(args, contribute_segment).items())))


@cache
def _mooncake_transfer_cached(config_items: tuple[tuple[str, Any], ...]):
    store = MooncakeDistributedStore()
    ret = store.setup(dict(config_items))
    if ret:
        raise RuntimeError(f"Mooncake store setup failed: {ret}")
    return MooncakeBundleTransfer(store, key_prefix="miles-rollout")


def _mooncake_store_config(args: Any, contribute_segment: bool) -> dict[str, Any]:
    config = getattr(args, "mooncake_store_init_kwargs", None) or {}
    global_segment_size = _parse_size(
        config.get("global_segment_size", os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", 8 * 1024**3))
    )
    return {
        "local_hostname": str(config.get("local_hostname") or os.getenv("MOONCAKE_LOCAL_HOSTNAME") or _local_hostname()),
        "metadata_server": str(config.get("metadata_server") or os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")),
        "global_segment_size": global_segment_size if contribute_segment else 0,
        "local_buffer_size": _parse_size(
            config.get("local_buffer_size", os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", 32 * 1024**3))
        ),
        "protocol": str(config.get("protocol") or os.getenv("MOONCAKE_PROTOCOL", "rdma")),
        "rdma_devices": str(config.get("device_name") or os.getenv("MOONCAKE_DEVICE", "")),
        "master_server_addr": str(config.get("master_server_address") or os.getenv("MOONCAKE_MASTER", "")),
    }


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
    return ray.util.get_node_ip_address()
