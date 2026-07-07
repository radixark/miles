import os
from functools import cache
from typing import Any

from miles.utils.ray_utils import Box

try:
    from mooncake.store import MooncakeDistributedStore
    from mooncake.structured_object_store import MooncakeBundleTransfer, export_ref, import_ref

    _MOONCAKE_AVAILABLE = True
except ImportError:
    _MOONCAKE_AVAILABLE = False


def check_mooncake_available() -> None:
    if not _MOONCAKE_AVAILABLE:
        raise ImportError("transfer_backend='mooncake' requires the mooncake package")


def is_mooncake_backend(args: Any) -> bool:
    backend = getattr(args, "transfer_backend", "ray")
    if backend not in {"ray", "mooncake"}:
        raise ValueError(f"Unsupported transfer backend: {backend}")
    if backend == "mooncake":
        check_mooncake_available()
    return backend == "mooncake"


def put_mooncake_rollout_data(
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


def get_mooncake_rollout_data(args: Any, ref: Box) -> dict[str, Any]:
    return _mooncake_transfer(args, contribute_segment=_should_contribute_segment()).get_legacy_dict(import_ref(ref.inner))


def release_mooncake_rollout_data(args: Any, data: dict[str, Any]) -> None:
    MooncakeBundleTransfer.release_result(data)


def cleanup_mooncake_rollout_data(args: Any, ref: Box) -> None:
    _mooncake_transfer(args, contribute_segment=False).remove_legacy_dict(import_ref(ref.inner))


def cleanup_mooncake_rollout_refs(args: Any, refs: Any) -> None:
    if isinstance(refs, dict) and "data_ref" in refs:
        refs = refs["data_ref"]
    if isinstance(refs, Box):
        cleanup_mooncake_rollout_data(args, refs)
        return
    for ref in refs:
        cleanup_mooncake_rollout_data(args, ref)


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
    global_segment_size = _parse_size(config.get("global_segment_size", os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", 8 * 1024**3)))
    return {
        "local_hostname": str(config.get("local_hostname") or os.getenv("MOONCAKE_LOCAL_HOSTNAME") or _local_hostname()),
        "metadata_server": str(config.get("metadata_server") or os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")),
        "global_segment_size": global_segment_size if contribute_segment else 0,
        "local_buffer_size": _parse_size(config.get("local_buffer_size", os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", 32 * 1024**3))),
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
    import ray

    return ray.util.get_node_ip_address()
