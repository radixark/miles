from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

try:
    from tensordict._td import TensorDict
except ImportError:

    class TensorDict(dict):
        def __init__(self, source=None, batch_size=None, device=None):
            super().__init__(source or {})
            self.batch_size = torch.Size(batch_size or [])
            self.device = device

        def __len__(self):
            return self.batch_size[0] if len(self.batch_size) > 0 else dict.__len__(self)

        def contiguous(self):
            return self

        def consolidate(self):
            return self


ALLOWED_SETUP_METHODS = {"setup", "setup_dummy"}
_STORE_CACHE: dict[tuple[tuple[str, str], ...], Any] = {}
_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def normalize_store_init_kwargs(store_init_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if store_init_kwargs is None:
        raise ValueError("mooncake transfer requires --mooncake-dataproto-store-init-kwargs")
    if not store_init_kwargs:
        return {"setup_method": "setup"}
    setup_method = store_init_kwargs.get("setup_method", "setup")
    if setup_method not in ALLOWED_SETUP_METHODS:
        raise ValueError(f"unsupported Mooncake store setup_method {setup_method!r}; allowed: {sorted(ALLOWED_SETUP_METHODS)}")
    return dict(store_init_kwargs)


def create_mooncake_store(store_init_kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = normalize_store_init_kwargs(store_init_kwargs or {})
    setup_method = kwargs.get("setup_method", "setup")
    if setup_method == "setup_dummy":
        return InMemoryMooncakeStore()

    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    setup_kwargs = {key: val for key, val in kwargs.items() if key != "setup_method"}
    setup = getattr(store, setup_method)
    try:
        ret = setup(**setup_kwargs)
    except TypeError:
        if setup_method != "setup":
            raise
        ret = setup(_env_store_config() | setup_kwargs)
    if ret != 0:
        raise RuntimeError(f"Mooncake store {setup_method} failed with retcode {ret}")
    return store


def get_cached_mooncake_store(store_init_kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = normalize_store_init_kwargs(store_init_kwargs)
    cache_key = tuple(sorted((key, repr(val)) for key, val in kwargs.items()))
    if cache_key not in _STORE_CACHE:
        _STORE_CACHE[cache_key] = create_mooncake_store(kwargs)
    return _STORE_CACHE[cache_key]


def remove_mooncake_keys(store: Any, keys: list[str]) -> None:
    errors = []
    for key in sorted(set(keys)):
        try:
            ret = store.remove(key, True)
        except TypeError:
            ret = store.remove(key)
        if ret != 0:
            errors.append((key, ret))
    if errors:
        raise RuntimeError(f"Mooncake key cleanup failed: {errors}")


def _env_store_config() -> dict[str, Any]:
    return {
        "local_hostname": os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
        "metadata_server": os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        "global_segment_size": int(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(16 * 1024 * 1024 * 1024))),
        "local_buffer_size": int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(16 * 1024 * 1024 * 1024))),
        "protocol": os.getenv("MOONCAKE_PROTOCOL", "rdma"),
        "rdma_devices": os.getenv("MOONCAKE_DEVICE", ""),
        "master_server_addr": os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051"),
    }


class InMemoryMooncakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.tensor_batches: dict[str, dict[str, torch.Tensor]] = {}

    def put(self, key: str, value: Any) -> int:
        self.objects[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def remove(self, key: str, force: bool = False) -> int:
        self.objects.pop(key, None)
        self.tensors.pop(key, None)
        self.tensor_batches.pop(key, None)
        return 0

    def put_tensor(self, key: str, tensor: torch.Tensor) -> int:
        self.tensors[key] = tensor.detach().cpu().clone()
        return 0

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key].clone()

    def put_tensor_batch(self, key: str, tensors: dict[str, torch.Tensor]) -> int:
        self.tensor_batches[key] = {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}
        return 0

    def get_tensor_batch(self, key: str, fields: list[str] | None = None) -> dict[str, torch.Tensor]:
        tensors = self.tensor_batches[key]
        selected = tensors.keys() if fields is None else fields
        return {name: tensors[name].clone() for name in selected}


def _import_mooncake_transfer():
    try:
        from mooncake.structured_object_store import MooncakeBundleTransfer
    except ImportError as exc:
        raise ImportError("Mooncake structured object DataProto helpers are required for mooncake transfer") from exc
    return MooncakeBundleTransfer


@dataclass
class MooncakeRemoteBatch:
    ref: Any
    store_init_kwargs: dict[str, Any] = field(default_factory=dict)
    key_prefix: str = "miles-rollout"

    @classmethod
    def from_dataproto(
        cls,
        proto: Any,
        store: Any,
        prefix: str,
        store_init_kwargs: dict[str, Any] | None = None,
        use_hard_pin: bool = True,
    ) -> MooncakeRemoteBatch:
        del use_hard_pin
        _validate_prefix(prefix)
        for name in (proto.batch or {}).keys():
            _validate_field_name(name)
        if isinstance(store, InMemoryMooncakeStore):
            key = f"{prefix}/batch"
            store.put_tensor_batch(key, {name: tensor for name, tensor in proto.batch.items()})
            ref = SimpleNamespace(
                key=key,
                batch_size=proto.batch.batch_size[0],
                field_index={name: SimpleNamespace(section="batch") for name in proto.batch.keys()},
            )
            return cls(ref=ref, store_init_kwargs=store_init_kwargs or {}, key_prefix=prefix)

        transfer = _import_mooncake_transfer()(store, key_prefix=prefix)
        ref = transfer.put_dataproto(proto, namespace="miles", partition="rollout", stage="batch")
        return cls(ref=ref, store_init_kwargs=store_init_kwargs or {}, key_prefix=prefix)

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        store: Any,
        prefix: str,
        store_init_kwargs: dict[str, Any] | None = None,
        use_hard_pin: bool = True,
    ) -> MooncakeRemoteBatch:
        from miles.utils.rollout_dataproto import DataProto

        proto = DataProto.from_dict(tensors={key: tensor.detach().cpu().contiguous() for key, tensor in tensors.items()})
        return cls.from_dataproto(
            proto,
            store,
            prefix,
            store_init_kwargs=store_init_kwargs,
            use_hard_pin=use_hard_pin,
        )

    def __len__(self) -> int:
        return int(self.ref.batch_size)

    def keys(self) -> list[str]:
        return [name for name, location in self.ref.field_index.items() if location.section == "batch"]

    @property
    def keys_to_cleanup(self) -> list[str]:
        keys = getattr(self.ref, "keys_to_cleanup", None)
        if keys is not None:
            return list(keys)
        key = getattr(self.ref, "key", None)
        if key is not None:
            return [key]
        cleanup_keys = []
        for location in getattr(self.ref, "field_index", {}).values():
            for attr in ("key", "object_key", "data_key"):
                value = getattr(location, attr, None)
                if value is not None:
                    cleanup_keys.append(value)
        return cleanup_keys

    def materialize(self, fields: list[str] | None = None) -> TensorDict:
        store = get_cached_mooncake_store(self.store_init_kwargs)
        if isinstance(store, InMemoryMooncakeStore):
            return TensorDict(source=store.get_tensor_batch(self.ref.key, fields), batch_size=(len(self),))
        transfer = _import_mooncake_transfer()(store, key_prefix=self.key_prefix)
        try:
            result = transfer.get_dataproto(self.ref, batch_fields=fields, non_tensor_fields=[])
            return TensorDict(source=result["batch"], batch_size=(len(self),))
        except Exception as exc:
            requested = self.keys() if fields is None else fields
            raise RuntimeError(f"MooncakeRemoteBatch materialize failed for fields={list(requested)}") from exc

    def cleanup(self) -> None:
        store = get_cached_mooncake_store(self.store_init_kwargs)
        if isinstance(store, InMemoryMooncakeStore):
            store.remove(self.ref.key, True)
            return
        transfer = _import_mooncake_transfer()(store, key_prefix=self.key_prefix)
        transfer.cleanup_dataproto(self.ref)


def _validate_prefix(prefix: str) -> None:
    if not prefix or len(prefix) > 256 or ".." in prefix or any(ord(ch) < 32 for ch in prefix):
        raise ValueError(f"invalid Mooncake key prefix: {prefix!r}")


def _validate_field_name(name: str) -> None:
    if _FIELD_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid Mooncake tensor field name: {name!r}")
