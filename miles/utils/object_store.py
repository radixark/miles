from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Annotated, Any, Literal

import ray
import ray._private.internal_api
from pydantic import AfterValidator, ConfigDict, Field, PlainSerializer

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.types import WorkerCommBackend

_MOONCAKE_IMPORT_ERROR: ImportError | None = None

try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
    from mooncake.structured_object_store import FieldSchema, MooncakeBundleTransfer, export_ref, import_ref

    _MOONCAKE_AVAILABLE = True
except ImportError as exc:
    _MOONCAKE_AVAILABLE = False
    _MOONCAKE_IMPORT_ERROR = exc
    FieldSchema = None
    ReplicateConfig = None


# ============================== types ==============================


class ObjectStoreBackend(Enum):
    RAY = "ray"
    MOONCAKE = "mooncake"


class _BaseStoreObjectRef(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


@dataclass(frozen=True)
class ValueSpec:
    codec: str
    dtype: str | None = None


class ObjectStoreGetResult:
    def __init__(self, value: Any, release_fn: Callable[[Any], None]) -> None:
        self._value = value
        self._release_fn = release_fn

    @property
    def value(self) -> Any:
        return self._value

    def __enter__(self) -> Any:
        return self._value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._release_fn(self._value)


# ============================ singleton ============================

_INSTANCE: BaseObjectStore | None = None


def init_instance(args: Namespace, *, contribute_segment: bool | None = None) -> BaseObjectStore:
    global _INSTANCE
    assert _INSTANCE is None, "object store instance is already initialized"
    _INSTANCE = _create_instance(args, contribute_segment=contribute_segment)
    return _INSTANCE


def get_instance() -> BaseObjectStore:
    assert _INSTANCE is not None, "object store instance is not initialized; call init_instance first"
    return _INSTANCE


def _create_instance(args: Namespace, *, contribute_segment: bool | None) -> BaseObjectStore:
    backend = ObjectStoreBackend(args.object_store_backend)
    if backend == ObjectStoreBackend.MOONCAKE:
        if contribute_segment is None:
            contribute_segment = _default_contribute_segment()
        return MooncakeObjectStore(args, contribute_segment=contribute_segment)
    return RayObjectStore(frees_objects=WorkerCommBackend(args.worker_comm_backend) == WorkerCommBackend.RPC)


def _default_contribute_segment() -> bool:
    local_rank = os.getenv("LOCAL_RANK")
    return local_rank is None or int(local_rank) == 0


# ============================ base class ===========================


class BaseObjectStore(ABC):
    @abstractmethod
    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        raise NotImplementedError

    @abstractmethod
    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        raise NotImplementedError

    @abstractmethod
    def remove(self, ref: StoreObjectRef) -> None:
        raise NotImplementedError


# ============================ ray backend ==========================


def _decode_ray_object_ref(payload: str) -> ray.ObjectRef:
    return ray.cloudpickle.loads(base64.b64decode(payload))


def _encode_ray_object_ref(payload: ray.ObjectRef) -> str:
    return base64.b64encode(ray.cloudpickle.dumps(payload)).decode()


class _RayStoreObjectRef(_BaseStoreObjectRef):
    backend: Literal[ObjectStoreBackend.RAY.value] = ObjectStoreBackend.RAY.value

    payload: Annotated[
        ray.ObjectRef | Annotated[str, AfterValidator(_decode_ray_object_ref)],
        PlainSerializer(_encode_ray_object_ref, return_type=str),
    ]


class RayObjectStore(BaseObjectStore):
    def __init__(self, *, frees_objects: bool) -> None:
        self._frees_objects = frees_objects

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        return _RayStoreObjectRef(payload=ray.put(value))

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        return ObjectStoreGetResult(value=ray.get(ref.payload), release_fn=_release_noop)

    def remove(self, ref: StoreObjectRef) -> None:
        if self._frees_objects:
            ray._private.internal_api.free([ref.payload])


def _release_noop(value: Any) -> None:
    pass


# ========================= mooncake backend ========================


class _MooncakeStoreObjectRef(_BaseStoreObjectRef):
    backend: Literal[ObjectStoreBackend.MOONCAKE.value] = ObjectStoreBackend.MOONCAKE.value

    payload: Any


class MooncakeObjectStore(BaseObjectStore):
    def __init__(self, args: Namespace, *, contribute_segment: bool) -> None:
        _check_mooncake_available()

        self._init_kwargs: dict[str, Any] = args.mooncake_store_init_kwargs or {}
        self._replica_num: int = args.mooncake_replica_num
        if self._replica_num < 1:
            raise ValueError("--mooncake-replica-num must be >= 1")

        store = MooncakeDistributedStore()
        setup_error = store.setup(_mooncake_store_config(self._init_kwargs, contribute_segment=contribute_segment))
        if setup_error:
            raise RuntimeError(f"Mooncake store setup failed: {setup_error}")
        self._transfer = MooncakeBundleTransfer(store, key_prefix="miles-object-store")

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        ref = self._transfer.put(
            value,
            type="dict",
            namespace="miles",
            chunk_bytes=self._init_kwargs.get("chunk_bytes"),
            config=self._replicate_config(),
            field_schemas=_field_schemas_for_value(value, value_spec),
        )
        return _MooncakeStoreObjectRef(payload=export_ref(ref))

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        value = self._transfer.get(import_ref(ref.payload), type="dict")
        return ObjectStoreGetResult(value=value, release_fn=MooncakeBundleTransfer.release_result)

    def remove(self, ref: StoreObjectRef) -> None:
        self._transfer.cleanup_dataproto(import_ref(ref.payload))

    def _replicate_config(self) -> Any:
        if self._replica_num == 1:
            return None
        config = ReplicateConfig()
        config.replica_num = self._replica_num
        return config


def _check_mooncake_available() -> None:
    if not _MOONCAKE_AVAILABLE:
        raise ImportError("object-store-backend='mooncake' requires the mooncake package") from _MOONCAKE_IMPORT_ERROR


def _field_schemas_for_value(value: Any, value_spec: dict[str, ValueSpec] | None) -> dict[str, Any] | None:
    if value_spec is None:
        return None
    return {
        field: FieldSchema(
            codec=spec.codec,
            nullable=False,
            metadata={
                "section": "meta_info" if spec.codec == "auto" else "non_tensor_batch",
                **({"dtype": spec.dtype} if spec.dtype is not None else {}),
            },
        )
        for field, spec in value_spec.items()
        if field in value
    }


def _mooncake_store_config(init_kwargs: dict[str, Any], *, contribute_segment: bool) -> dict[str, Any]:
    return {
        "local_hostname": str(
            init_kwargs.get("local_hostname") or os.getenv("MOONCAKE_LOCAL_HOSTNAME") or _local_hostname()
        ),
        "metadata_server": str(
            init_kwargs.get("metadata_server") or os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
        ),
        "local_buffer_size": _parse_size(
            init_kwargs.get("local_buffer_size", os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", 32 * 1024**3))
        ),
        "protocol": str(init_kwargs.get("protocol") or os.getenv("MOONCAKE_PROTOCOL", "rdma")),
        "rdma_devices": str(init_kwargs.get("device_name") or os.getenv("MOONCAKE_DEVICE", "")),
        "master_server_addr": str(init_kwargs.get("master_server_address") or os.getenv("MOONCAKE_MASTER", "")),
        "global_segment_size": (
            _parse_size(init_kwargs.get("global_segment_size", os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", 8 * 1024**3)))
            if contribute_segment
            else 0
        ),
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


StoreObjectRef = Annotated[_RayStoreObjectRef | _MooncakeStoreObjectRef, Field(discriminator="backend")]
