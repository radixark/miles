import ctypes
import logging
import os
import pickle
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Pickle protocol 5: same as Ray for fair comparison
_PICKLE_PROTOCOL = 5

# Whitelist of (module, name) for safe pickle unpickling. Prevents RCE when loading
# data from Mooncake distributed store. Rollout data uses: dict, list, numpy arrays.
_SAFE_PICKLE_WHITELIST = frozenset(
    {
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "bool"),
        ("builtins", "slice"),
        ("builtins", "range"),
        ("builtins", "type"),
        ("builtins", "object"),
        ("builtins", "NoneType"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows whitelisted classes. Mitigates RCE from malicious pickle."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _SAFE_PICKLE_WHITELIST:
            raise pickle.UnpicklingError(f"Unpickling of {module}.{name} is disabled for security. Only whitelisted types (builtins, numpy ndarray/dtype) are allowed.")
        return super().find_class(module, name)


def _safe_pickle_loads(data: bytes) -> Any:
    """
    Deserialize pickle data using a restricted unpickler to prevent RCE.
    Use when loading data from external/distributed store (e.g. Mooncake).
    """
    if os.environ.get("MILES_UNSAFE_PICKLE", "").lower() in ("1", "true", "yes"):
        return pickle.loads(data)
    import io

    return _RestrictedUnpickler(io.BytesIO(data)).load()


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

    def cleanup(self, handle: Any):  # noqa: B027
        """
        Clean up data associated with the handle (optional).
        """
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
                raise ValueError("Invalid segment size: missing number before 'gb'")
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
    device_name: str | None
    master_server_address: str

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        Set MOONCAKE_PROTOCOL=rdma for best performance (requires InfiniBand/RoCE).
        """
        if not os.getenv("MOONCAKE_MASTER"):
            raise ValueError("Neither the environment variable 'MOONCAKE_CONFIG_PATH' nor 'MOONCAKE_MASTER' is set.")
        return MooncakeStoreConfig(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
            global_segment_size=_parse_segment_size(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", DEFAULT_GLOBAL_SEGMENT_SIZE)),
            local_buffer_size=_parse_segment_size(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),  # use "rdma" for RDMA
            device_name=os.getenv("MOONCAKE_DEVICE", ""),
            master_server_address=os.getenv("MOONCAKE_MASTER"),
        )


class MooncakeDataTransfer(DataTransferBackend):
    """
    Data transfer using Mooncake Store with automatic cleanup.

    Uses delayed asynchronous deletion to clean up data after it's been retrieved.
    Keys are queued for deletion after get() is called, and a background thread
    periodically removes them to free up storage space.
    """

    def __init__(self, cleanup_delay_seconds: float = 5.0, cleanup_batch_size: int = 100, enable_auto_cleanup: bool = True):
        """
        Initialize MooncakeDataTransfer.

        Args:
            cleanup_delay_seconds: Delay before deleting keys after get() is called.
                                  This ensures data is fully processed before deletion.
            cleanup_batch_size: Maximum number of keys to delete in one batch.
            enable_auto_cleanup: If True, automatically schedule keys for deletion after get().
                                If False, keys must be manually cleaned up via cleanup().
        """
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError("Please install mooncake by following the instructions at https://kvcache-ai.github.io/Mooncake/getting_started/build.html to run SGLang with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()

            # Load from environment variables
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake config loaded from env.")

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
                raise RuntimeError(f"Failed to setup Mooncake store, error code: {ret_code}")
            logger.info("Mooncake store setup complete.")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        # Cleanup configuration
        self.cleanup_delay_seconds = cleanup_delay_seconds
        self.cleanup_batch_size = cleanup_batch_size
        self.enable_auto_cleanup = enable_auto_cleanup

        # Priority queue for keys pending deletion: (deletion_time, key). Lower time = higher priority.
        self._pending_deletion = queue.PriorityQueue()

        # Reusable buffers: avoid repeated alloc when sizes are similar
        self._get_buffer = None
        self._get_buffer_size = 0
        self._get_buffer_registered = False  # persistent reg to avoid ~17ms/round overhead
        self._put_buffer = None
        self._put_buffer_size = 0
        self._put_buffer_registered = False  # persistent reg for put

        # Background cleanup thread (only start if auto cleanup is enabled)
        self._cleanup_thread = None
        self._cleanup_thread_lock = threading.Lock()
        self._cleanup_stop_event = threading.Event()
        if self.enable_auto_cleanup:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start the background cleanup thread."""
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, name="MooncakeCleanupThread", daemon=True)
        self._cleanup_thread.start()
        logger.info("Mooncake cleanup thread started")

    def _cleanup_worker(self):
        """Background worker that deletes keys when their deletion time is reached.
        Uses PriorityQueue to block until the next item is due, avoiding polling.
        """
        while not self._cleanup_stop_event.is_set():
            try:
                keys_to_delete: list[str] = []
                sleep_until: float | None = None

                # Collect keys ready for deletion (up to batch_size)
                while len(keys_to_delete) < self.cleanup_batch_size:
                    # Block up to 0.5s when queue is empty; otherwise get next item by priority
                    try:
                        deletion_time, key = self._pending_deletion.get(timeout=0.5)
                    except queue.Empty:
                        break

                    current_time = time.time()
                    if current_time >= deletion_time:
                        keys_to_delete.append(key)
                    else:
                        # Put back and sleep until this item is due
                        self._pending_deletion.put((deletion_time, key))
                        sleep_until = deletion_time
                        break

                # Batch delete keys
                if keys_to_delete:
                    deleted_count = 0
                    for key in keys_to_delete:
                        try:
                            result = self.store.remove(key)
                            if result == 0:
                                deleted_count += 1
                            else:
                                logger.warning(f"Failed to delete key {key}, error code: {result}")
                        except Exception as e:
                            logger.warning(f"Exception while deleting key {key}: {e}")

                    if deleted_count > 0:
                        logger.debug(f"Deleted {deleted_count} keys from Mooncake store")

                # Sleep until next item is due, or 0.5s if queue was empty
                if sleep_until is not None:
                    sleep_time = min(0.5, max(0.1, sleep_until - time.time()))
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}", exc_info=True)
                time.sleep(1.0)  # Sleep longer on error

    def _schedule_deletion(self, key: str):
        """Schedule a key for deletion after the delay period."""
        deletion_time = time.time() + self.cleanup_delay_seconds
        self._pending_deletion.put((deletion_time, key))

    def put(self, data: Any) -> Any:
        """
        Store data using zero-copy put_from with persistent buffer registration.
        Same pickle format as Ray for fair comparison.
        """
        serialized_data = pickle.dumps(data, protocol=_PICKLE_PROTOCOL)
        data_size = len(serialized_data)

        key = f"rollout_data_{uuid.uuid4().hex}"

        buffer_size = data_size + 1024
        if self._put_buffer is None or self._put_buffer_size < buffer_size:
            if self._put_buffer_registered:
                self.store.unregister_buffer(ctypes.addressof(self._put_buffer))
                self._put_buffer_registered = False
            self._put_buffer = (ctypes.c_ubyte * buffer_size)()
            self._put_buffer_size = buffer_size
        buffer = self._put_buffer
        buffer_ptr = ctypes.addressof(buffer)

        ctypes.memmove(buffer, serialized_data, data_size)

        if not self._put_buffer_registered:
            result = self.store.register_buffer(buffer_ptr, buffer_size)
            if result != 0:
                raise RuntimeError(f"Failed to register buffer for put_from: {result}")
            self._put_buffer_registered = True

        result = self.store.put_from(key, buffer_ptr, data_size)
        if result != 0:
            raise RuntimeError(f"put_from failed with code: {result}")

        return key

    def get(self, handle: Any, auto_cleanup: bool | None = None) -> Any:
        """
        Retrieve data using zero-copy get_into.

        Args:
            handle: The key/handle returned by put()
            auto_cleanup: If True, schedule the key for automatic deletion after retrieval.
                         If False, skip automatic deletion (manual cleanup required).
                         If None (default), use the instance-level enable_auto_cleanup setting.

        Returns:
            The deserialized data object.
        """
        key = handle

        data_size = self.store.get_size(key)
        if data_size < 0:
            raise ValueError(f"Data not found in Mooncake for key: {key}, error code: {data_size}")

        buffer_size = data_size + 1024
        if self._get_buffer is None or self._get_buffer_size < buffer_size:
            if self._get_buffer_registered:
                self.store.unregister_buffer(ctypes.addressof(self._get_buffer))
                self._get_buffer_registered = False
            self._get_buffer = (ctypes.c_ubyte * buffer_size)()
            self._get_buffer_size = buffer_size
        buffer = self._get_buffer
        buffer_ptr = ctypes.addressof(buffer)

        if not self._get_buffer_registered:
            result = self.store.register_buffer(buffer_ptr, buffer_size)
            if result != 0:
                raise RuntimeError(f"Failed to register buffer for get_into: {result}")
            self._get_buffer_registered = True

        bytes_read = self.store.get_into(key, buffer_ptr, buffer_size)
        if bytes_read < 0:
            raise RuntimeError(f"get_into failed with code: {bytes_read}")
        if bytes_read != data_size:
            raise RuntimeError(f"Data size mismatch: expected {data_size}, got {bytes_read}")

        # Use restricted unpickler to mitigate RCE from malicious data in distributed store.
        # Set MILES_UNSAFE_PICKLE=1 to use raw pickle in fully trusted environments.
        data = _safe_pickle_loads(bytes(memoryview(buffer)[:bytes_read]))

        should_cleanup = auto_cleanup if auto_cleanup is not None else self.enable_auto_cleanup
        if should_cleanup:
            with self._cleanup_thread_lock:
                if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
                    self._start_cleanup_thread()
            self._schedule_deletion(key)

        return data

    def clear(self) -> None:
        """Remove all data from the store."""
        self.store.remove_all()
        # Clear pending deletion queue
        while not self._pending_deletion.empty():
            try:
                self._pending_deletion.get_nowait()
            except queue.Empty:
                break

    # Mooncake error -706: key not found (e.g. already removed by get_into or replication)
    _CLEANUP_KEY_NOT_FOUND = -706

    def cleanup(self, handle: Any):
        """
        Immediately clean up data associated with the handle.
        This bypasses the delayed deletion mechanism.
        """
        key = handle
        result = self.store.remove(key)
        if result != 0 and result != self._CLEANUP_KEY_NOT_FOUND:
            logger.warning(f"Failed to cleanup key {key}, error code: {result}")

    def shutdown(self):
        """Shutdown the cleanup thread gracefully and unregister persistent buffers."""
        if self._put_buffer_registered and self._put_buffer is not None:
            try:
                self.store.unregister_buffer(ctypes.addressof(self._put_buffer))
            except Exception as e:
                logger.warning("Failed to unregister put buffer on shutdown: %s", e)
            self._put_buffer_registered = False
        if self._get_buffer_registered and self._get_buffer is not None:
            try:
                self.store.unregister_buffer(ctypes.addressof(self._get_buffer))
            except Exception as e:
                logger.warning("Failed to unregister get buffer on shutdown: %s", e)
            self._get_buffer_registered = False
        if self._cleanup_thread is not None:
            self._cleanup_stop_event.set()
            self._cleanup_thread.join(timeout=5.0)
            if self._cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not stop within timeout")
            else:
                logger.info("Cleanup thread stopped successfully")


def get_data_transfer_backend(args):
    """Factory function to get the appropriate backend."""
    backend_name = getattr(args, "transfer_backend", "ray")
    if backend_name == "mooncake":
        return MooncakeDataTransfer()
    else:
        return RayDataTransfer()
