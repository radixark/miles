import base64
import bisect
import os
import time
import uuid
from typing import Any, NamedTuple

import ray
import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import RemoteShard
from miles.platforms import current_platform

# polled: TransferStatus.WaitFor sleeps to its timeout on some MoRI backends (ROCm/mori#873)
_POLL_SECONDS = 0.0005
# one QP fills an AMD AI NIC's send queue under load; unchunked, a large transfer gets about half the line rate;
# nothing pops the inbound status a notification leaves. MORI_IO_* variables override these.
_RDMA_BACKEND_CONFIG = dict(qp_per_transfer=4, enable_notification=False, enable_transfer_chunking=True)


class _Region(NamedTuple):
    """Registered memory [start, start + nbytes) and the MoRI MemoryDesc that names it."""

    start: int
    nbytes: int
    memory_desc: Any


def _find(regions: list[_Region], address: int, nbytes: int) -> _Region | None:
    """The region of `regions`, sorted by start, that holds [address, address + nbytes)."""
    index = bisect.bisect_right(regions, address, key=lambda region: region.start) - 1
    if index >= 0 and address + nbytes <= regions[index].start + regions[index].nbytes:
        return regions[index]
    return None


class MoriTransport:
    """The MoRI-IO `NativeTransport` of `P2PTransport`: one MoRI `IOEngine` with the RDMA backend per trainer process.

    Each runner of a rollout engine rank publishes its own MoRI engine and a MemoryDesc per registered block of its
    weights (sglang `--remote-instance-weight-loader-start-seed-via-mori`). A write is one MoRI batch write per shard,
    with one entry per (transfer buffer, remote block) pair, failed if it does not finish within `write_timeout`.
    """

    name = "mori"
    transfer_buffer_device = torch.device("cpu")

    def __init__(self, write_timeout: float, transfer_buffer: str) -> None:
        """`transfer_buffer` is --p2p-mori-transfer-buffer (gpu or cpu)."""
        if transfer_buffer == "gpu":
            _assert_no_expandable_segments()
            self.transfer_buffer_device = current_platform.current_device()
        self._engine = _create_io_engine()
        self._write_timeout = write_timeout
        self._peers: dict[str, tuple[Any, list[_Region]]] = {}
        # replaced, never mutated: write threads read it without a lock
        self._local_regions: list[_Region] = []
        # the statuses of writes that failed, kept for good: MoRI may still read their source, and completes a status
        # through a raw pointer
        self._failed_writes: list[list[Any]] = []

    def connect_peer(self, transfer_engine_info: dict) -> str:
        from mori.io import EngineDesc, MemoryDesc

        engine_desc = EngineDesc.unpack(base64.b64decode(transfer_engine_info["engine_desc"]))
        blocks = sorted(
            (
                _Region(int(address), int(nbytes), MemoryDesc.unpack(base64.b64decode(memory_desc)))
                for address, nbytes, memory_desc in transfer_engine_info["memory_blocks"]
            ),
            key=lambda region: region.start,
        )
        if engine_desc.key not in self._peers:
            self._engine.register_remote_engine(engine_desc)
        self._peers[engine_desc.key] = (engine_desc, blocks)
        return engine_desc.key

    def remove_peers(self) -> None:
        for engine_desc, _ in self._peers.values():
            self._engine.deregister_remote_engine(engine_desc)
        self._peers = {}

    def register_tensor(self, tensor: torch.Tensor) -> _Region:
        from mori.io import MemoryLocationType

        on_cpu = tensor.device.type == "cpu"
        memory_desc = self._engine.register_memory(
            tensor.data_ptr(),
            tensor.nbytes,
            -1 if on_cpu else tensor.device.index,
            MemoryLocationType.CPU if on_cpu else MemoryLocationType.GPU,
        )
        region = _Region(tensor.data_ptr(), tensor.nbytes, memory_desc)
        self._local_regions = sorted([*self._local_regions, region], key=lambda region: region.start)
        return region

    def unregister_tensor(self, region: _Region) -> None:
        if self._failed_writes:
            raise RuntimeError(
                f"Cannot release registered memory after a {self.name} write with unknown native completion"
            )
        self._engine.deregister_memory(region.memory_desc)
        self._local_regions = [kept for kept in self._local_regions if kept is not region]

    def write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        peer = remote_shard.transport_peer
        _, blocks = self._peers[peer]
        local_regions = self._local_regions
        # one batch entry per (local region, remote block): local offsets, remote offsets, sizes
        entries: dict[tuple[int, int], tuple[_Region, _Region, list[int], list[int], list[int]]] = {}
        for name, tensor in tensors_by_name.items():
            source, nbytes = tensor.data_ptr(), tensor.nbytes
            target = remote_shard.weight_locations_by_name[name].address
            if nbytes == 0:
                continue
            if (local := _find(local_regions, source, nbytes)) is None:
                raise RuntimeError(f"{name} ({nbytes} bytes at {source:#x}) is not in memory registered with MoRI")
            if (remote := _find(blocks, target, nbytes)) is None:
                raise RuntimeError(
                    f"{name} ({nbytes} bytes at {target:#x}) is not in any weight block MoRI engine {peer} registered"
                )
            entry = entries.setdefault((local.start, remote.start), (local, remote, [], [], []))
            entry[2].append(source - local.start)
            entry[3].append(target - remote.start)
            entry[4].append(nbytes)
        if not entries:
            return
        if self.transfer_buffer_device.type != "cpu":
            # the loads into a GPU buffer are kernels the loading thread queued; the NIC must not read before they ran
            torch.accelerator.synchronize(self.transfer_buffer_device)

        locals_, remotes, local_offsets, remote_offsets, sizes = zip(*entries.values(), strict=True)
        statuses: list[Any] = []
        try:
            statuses = self._engine.batch_write(
                [local.memory_desc for local in locals_],
                list(local_offsets),
                [remote.memory_desc for remote in remotes],
                list(remote_offsets),
                list(sizes),
                [self._engine.allocate_transfer_uid() for _ in entries],
            )
            deadline = time.monotonic() + self._write_timeout
            while any(status.InProgress() for status in statuses) and time.monotonic() < deadline:
                time.sleep(_POLL_SECONDS)
        finally:
            succeeded = bool(statuses) and all(status.Succeeded() for status in statuses)
            if not succeeded:
                self._failed_writes.append(statuses)
        if not succeeded:
            running = sum(status.InProgress() for status in statuses)
            reason = (
                f"{running} of {len(statuses)} batch entries still running after {self._write_timeout}s"
                if running
                else f"errors {sorted({status.Message() for status in statuses if status.Failed()})}"
            )
            raise RuntimeError(f"the MoRI WRITE of {len(tensors_by_name)} weights to {peer} failed: {reason}")


def _assert_no_expandable_segments() -> None:
    alloc_conf = ",".join(
        os.environ.get(name, "")
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
    )
    if "expandable_segments:true" in alloc_conf.replace(" ", "").lower():
        raise ValueError(
            f"MoRI cannot register expandable-segment GPU memory ({alloc_conf.strip(',')}): turn expandable_segments "
            "off for the trainer, or keep the MoRI transfer buffers in host memory (--p2p-mori-transfer-buffer cpu)"
        )


def _create_io_engine() -> Any:
    # not in miles' requirements: the ROCm images ship it with sglang
    from mori.io import BackendType, IOEngine, IOEngineConfig, RdmaBackendConfig

    engine = IOEngine(
        f"miles-p2p-{uuid.uuid4()}", IOEngineConfig(host=ray._private.services.get_node_ip_address(), port=0)
    )
    engine.create_backend(BackendType.RDMA, RdmaBackendConfig(**_RDMA_BACKEND_CONFIG))
    return engine
