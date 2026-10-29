import base64
import contextlib
import io
import shutil
import socket
from argparse import Namespace
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from urllib.parse import urlparse

import torch
import torch.multiprocessing
from torch.multiprocessing import reductions

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import (
    OwnedBuckets,
    get_data_replica_rank_and_size,
)
from miles.backends.training_utils.weight_update.protocols.utils.flattened_bucket import flatten_weight_bucket
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.platforms import current_platform
from miles.utils import async_utils


class UpdateWeightFromSharedMemory(WeightTransferProtocol):
    """
    Host weight sync through shared memory, for engines on the trainer's host. Each gathered bucket goes
    to the replica rank with the fewest bytes so far, which copies it from the device straight into that bucket's
    shared-memory tensor, in the engine's flattened layout, and hands the engines its handle through
    /update_weights_from_tensor; each engine maps it in place, with no file written or parsed.
    The shared-memory tensors are kept across syncs (`_SharedImage`), so a sync neither allocates nor
    copies through other host memory; /dev/shm holds each rank's share of the model for the run.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._image = _SharedImage()
        # this sync's engine calls; the next sync rewrites the shared buckets only once every engine has loaded them
        self._in_flight: list[Future] = []
        self._send_error: Exception | None = None

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        engines_elsewhere = _engines_off_this_host(rollout_engines)
        if engines_elsewhere:
            raise ValueError(
                "--update-weight-transfer-mode shm needs every rollout engine on the trainer's host, which they "
                f"open the shared-memory buckets on; {engines_elsewhere} run elsewhere"
            )
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, self.rollout_engines, engine_cell_ids)
        self.group_name = "miles-shm"
        self.is_sender = True
        self._selector = selector
        self._engine_tp_sizes = engine_gpu_counts
        replica_rank, replica_size = get_data_replica_rank_and_size(parallel_state, placement)
        self._owned_buckets = OwnedBuckets(
            replica_rank=replica_rank, replica_size=replica_size, host_bucket=self._image.host_bucket
        )

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        self._owned_buckets.reset()
        self._send_error = None
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # raising mid-stream would strand the other ranks in its gathers; every rank raises it after the stream
        if self._send_error is not None:
            return
        try:
            self._owned_buckets.offer(bucket)
        except Exception as error:
            self._send_error = error

    def after_base_weights(self) -> None:
        # after the stream: take() is where the non-blocking device copies are known to have landed
        cell_updaters = list(self.cell_updaters_of_cell_id.values())
        with fail_on_every_rank("[shm] bucket publish"):
            if self._send_error is not None:
                raise self._send_error
            for index, _ in self._owned_buckets.take():
                shared = self._image.bucket(index)
                with _file_system_sharing():
                    payload = {"flattened_tensor": shared.flat, "metadata": shared.metadata}
                    self._in_flight += self._publish(cell_updaters, payload)

    def _publish(self, cell_updaters: list, bucket_payload: dict) -> list[Future]:
        return [
            updater.submit_client_call(
                "update_weights_from_tensor",
                # one serialization per receiving process: each takes its own reference to the /dev/shm file,
                # which is unlinked once the references are released, so a shared payload fails a later reader
                serialized_named_tensors=[_serialize_payload(bucket_payload) for _ in range(tp_size)],
                load_format="flattened_bucket",
                selector=self._selector,
            )
            for updater, tp_size in zip(cell_updaters, self._engine_tp_sizes, strict=True)
            # a payload no engine opens keeps its reference, so the file would outlive the sync
            if not updater.is_errored
        ]

    def finalize(self, weight_version: int) -> None:
        with fail_on_every_rank("[shm] engine load"):
            futures, self._in_flight = self._in_flight, []
            check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)


def _serialize_payload(payload: dict) -> str:
    """sglang's `MultiprocessingSerializer.serialize(payload, output_str=True)`, with torch's own tensor reducer.

    sglang's `monkey_patch_torch_reductions` (installed by Megatron's weight iterator) rewrites every reduced
    tensor's args as a CUDA tensor's and fails on these shared-memory CPU tensors; the override is per pickler.
    """
    buffer = io.BytesIO()
    pickler = ForkingPickler(buffer)
    pickler.dispatch_table[torch.Tensor] = getattr(  # config-access-exempt: set only once sglang patched torch
        reductions, "_reduce_tensor_original", reductions.reduce_tensor
    )
    pickler.dump(payload)
    return base64.b64encode(buffer.getvalue()).decode()


def _engines_off_this_host(rollout_engines: Sequence[SGLangApiClient]) -> list[str]:
    return [engine.server_url for engine in rollout_engines if not _is_this_host(urlparse(engine.server_url).hostname)]


def _is_this_host(host: str) -> bool:
    # only an address of one of this machine's interfaces can be bound, IPv4 or IPv6
    for family, _, _, _, address in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((address[0], 0))
            except OSError:
                continue
        return True
    return False


@dataclass(frozen=True)
class _SharedBucket:
    """One bucket as a pinned shared-memory byte tensor in the engine's flattened layout, with a view per weight."""

    layout: tuple[tuple[str, tuple[int, ...], torch.dtype], ...]
    flat: torch.Tensor
    metadata: list
    views: list[torch.Tensor]


class _SharedImage:
    """This rank's owned buckets in shared memory, by stream index, reused while a bucket's layout is unchanged.

    Rewriting in place never races an engine: the next sync starts only after every load has finished.
    """

    def __init__(self) -> None:
        self._buckets: dict[int, _SharedBucket] = {}

    def host_bucket(self, index: int, bucket: list[tuple[str, torch.Tensor]]) -> list[torch.Tensor]:
        layout = tuple((name, tuple(tensor.shape), tensor.dtype) for name, tensor in bucket)
        shared = self._buckets.get(index)
        if shared is None or shared.layout != layout:
            if shared is not None:
                current_platform.unpin_host_memory(shared.flat)
            shared = self._buckets[index] = _allocate_shared_bucket(bucket, layout)
        return shared.views

    def bucket(self, index: int) -> _SharedBucket:
        return self._buckets[index]


def _allocate_shared_bucket(bucket: list[tuple[str, torch.Tensor]], layout) -> _SharedBucket:
    metadata = flatten_weight_bucket([(name, tensor.to("meta")) for name, tensor in bucket]).get_metadata()
    nbytes = metadata[-1].end_idx
    # a full tmpfs SIGBUSes on write; the earlier buckets' pages are already in (pinned or copied)
    free = shutil.disk_usage("/dev/shm").free
    if nbytes > free:
        raise RuntimeError(
            f"--update-weight-transfer-mode shm keeps each rank's share of the model in /dev/shm; a {nbytes} byte "
            f"bucket does not fit in its {free} free bytes. Enlarge /dev/shm or use more trainer ranks"
        )
    with _file_system_sharing():
        flat = torch.empty(nbytes, dtype=torch.uint8).share_memory_()
    current_platform.pin_host_memory(flat)
    views = [
        flat[meta.start_idx : meta.end_idx].view(dtype).view(shape)
        for (_, shape, dtype), meta in zip(layout, metadata, strict=True)
    ]
    return _SharedBucket(layout=layout, flat=flat, metadata=metadata, views=views)


@contextlib.contextmanager
def _file_system_sharing() -> Iterator[None]:
    # the payload must name the /dev/shm file, which an unrelated process (the engine) can open; the strategy is
    # process-wide, so it is set only while allocating and serializing
    previous = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        torch.multiprocessing.set_sharing_strategy(previous)
