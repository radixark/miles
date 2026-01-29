import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any

import ray
import os
from dataclasses import dataclass
from typing import Optional


from miles.utils.ray_utils import Box

logger = logging.getLogger(__name__)


class DataTransferBackend(ABC):
    """Abstract base class for data transfer backends."""

    @abstractmethod
    def put(self, data: Any) -> Any:
        """
        Store data and return a handle/key.
        """
        pass

    @abstractmethod
    def get(self, handle: Any) -> Any:
        """
        Retrieve data using the handle/key.
        """
        pass

    def cleanup(self, handle: Any):
        """
        Clean up data associated with the handle (optional).
        """
        pass


class RayDataTransfer(DataTransferBackend):
    """Default data transfer using Ray Object Store."""

    def put(self, data: Any) -> Any:
        return Box(ray.put(data))

    def get(self, handle: Any) -> Any:
        if isinstance(handle, Box):
            return ray.get(handle.inner)
        return ray.get(handle)


DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB


def _parse_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid segment size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: Optional[str]
    master_server_address: str

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        if not os.getenv("MOONCAKE_MASTER"):
            raise ValueError("Neither the environment variable 'MOONCAKE_CONFIG_PATH' nor 'MOONCAKE_MASTER' is set.")
        return MooncakeStoreConfig(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
            global_segment_size=_parse_segment_size(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_segment_size(
                os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE", ""),
            master_server_address=os.getenv("MOONCAKE_MASTER"),
        )


class MooncakeDataTransfer(DataTransferBackend):
    """
    Data transfer using Mooncake Store.
    """

    def __init__(self):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
                "to run SGLang with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()

            # Load from environment variables
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded from env successfully.")

            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    def put(self, data: Any) -> Any:
        serialized_data = pickle.dumps(data)
        
        import uuid
        key = f"rollout_data_{uuid.uuid4().hex}"
        
        self.store.put(key, serialized_data)
        return key

    def get(self, handle: Any) -> Any:        
        key = handle
        serialized_data = self.store.get(key)
        if serialized_data is None:
            raise ValueError(f"Data not found in Mooncake for key: {key}")
            
        return pickle.loads(serialized_data)

    def clear(self) -> None:
        self.store.remove_all()

    def cleanup(self, handle: Any):
        """
        Clean up data associated with the handle (optional).
        """
        self.store.remove(handle)


def get_data_transfer_backend(args):
    """Factory function to get the appropriate backend."""
    backend_name = getattr(args, "transfer_backend", "ray")
    # Here just hack to force use mooncake
    return MooncakeDataTransfer()
    if backend_name == "mooncake":
        return MooncakeDataTransfer()
    else:
        return RayDataTransfer()
