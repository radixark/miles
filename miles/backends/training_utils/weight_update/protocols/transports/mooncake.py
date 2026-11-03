import os
from typing import Any

import ray
import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import RemoteShard


class MooncakeTransport:
    """The Mooncake `NativeTransport` of `P2PTransport`: one Mooncake transfer engine for this trainer process."""

    name = "mooncake"
    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self) -> None:
        self._transfer_engine = _create_transfer_engine()
        # a failed write does not prove Mooncake stopped reading its source
        self._has_failed_writes = False

    def connect_peer(self, transfer_engine_info: dict) -> str:
        return transfer_engine_info["session_id"]

    def remove_peers(self) -> None:
        # this side keeps no state per Mooncake session
        pass

    def register_tensor(self, tensor: torch.Tensor) -> int:
        ret = self._transfer_engine.register_memory(tensor.data_ptr(), tensor.nbytes)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake could not register {tensor.nbytes} bytes at {tensor.data_ptr():#x}: error {ret}"
            )
        return tensor.data_ptr()

    def unregister_tensor(self, address: int) -> None:
        if self._has_failed_writes:
            raise RuntimeError(
                f"Cannot release registered memory after a {self.name} write with unknown native completion"
            )
        ret = self._transfer_engine.unregister_memory(address)
        if ret != 0:
            raise RuntimeError(f"Mooncake could not unregister the memory at {address:#x}: error {ret}")

    def write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        names = list(tensors_by_name)
        succeeded = False
        try:
            ret = self._transfer_engine.batch_transfer_sync_write(
                remote_shard.transport_peer,
                [tensors_by_name[name].data_ptr() for name in names],
                [remote_shard.weight_locations_by_name[name].address for name in names],
                [tensors_by_name[name].nbytes for name in names],
            )
            succeeded = ret >= 0
        finally:
            # write threads share this flag: only ever set it, so one thread's success never clears another's failure
            if not succeeded:
                self._has_failed_writes = True
        if not succeeded:
            raise RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")


def _create_transfer_engine() -> Any:
    # not in miles' requirements
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    transfer_engine.initialize(
        ray._private.services.get_node_ip_address(),
        "P2PHANDSHAKE",
        os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
        os.environ.get("MOONCAKE_DEVICE", ""),
    )
    return transfer_engine
