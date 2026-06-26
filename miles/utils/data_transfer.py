import os
import pickle
import uuid
from abc import ABC, abstractmethod
from typing import Any

_PICKLE_PROTOCOL = 5


class DataTransferBackend(ABC):
    """Abstract base class for data transfer backends."""

    @abstractmethod
    def put(self, data: Any) -> Any:
        pass

    @abstractmethod
    def get(self, handle: Any) -> Any:
        pass

    def cleanup(self, handle: Any):  # noqa: B027
        pass


class RayDataTransfer(DataTransferBackend):
    """Default data transfer using Ray Object Store."""

    def put(self, data: Any) -> Any:
        import ray

        from miles.utils.ray_utils import Box

        return Box(ray.put(data))

    def get(self, handle: Any) -> Any:
        import ray

        from miles.utils.ray_utils import Box

        if isinstance(handle, Box):
            return ray.get(handle.inner)
        return ray.get(handle)


class MooncakeDataProtoTransfer(DataTransferBackend):
    """Mooncake DataProto rollout transfer handle adapter."""

    def put(self, data: Any) -> Any:
        return data

    def get(self, handle: Any) -> Any:
        from miles.utils.rollout_dataproto import DataProto, dataproto_to_rollout_data

        if not isinstance(handle, DataProto):
            raise TypeError(f"expected DataProto handle, got {type(handle)}")
        return dataproto_to_rollout_data(handle, preserve_remote_tensors=True)

    def cleanup(self, handle: Any):
        from miles.utils.rollout_dataproto import cleanup_dataproto_refs

        cleanup_dataproto_refs([handle])


class DiskDataTransfer(DataTransferBackend):
    """Local disk transfer backend used by tests and benchmarks."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def put(self, data: Any) -> Any:
        path = os.path.join(self._base_dir, f"rollout_data_{uuid.uuid4().hex}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=_PICKLE_PROTOCOL)
        return path

    def get(self, handle: Any) -> Any:
        with open(handle, "rb") as f:
            return pickle.load(f)

    def cleanup(self, handle: Any):
        try:
            os.remove(handle)
        except FileNotFoundError:
            pass


def _pack_rollout_to_tensor_buffer(data: Any) -> bytes:
    return pickle.dumps(data, protocol=_PICKLE_PROTOCOL)


def _unpack_rollout_from_tensor_buffer(buffer: bytes) -> Any:
    return pickle.loads(buffer)


def get_data_transfer_backend(args):
    """Factory function to get the appropriate backend."""
    backend_name = getattr(args, "transfer_backend", "ray")
    if backend_name == "mooncake_dataproto":
        return MooncakeDataProtoTransfer()
    return RayDataTransfer()
