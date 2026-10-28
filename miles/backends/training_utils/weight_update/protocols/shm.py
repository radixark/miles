import base64
import contextlib
import io
import socket
from argparse import Namespace
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
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
from miles.utils import async_utils


# buckets in flight across the host, enough to keep the engines loading while the next one is copied out; arbitrary
# otherwise. Each replica rank gets its share but at least one, so /dev/shm holds max(this, replica ranks) buckets.
_MAX_BUCKETS_IN_FLIGHT_PER_HOST = 4


class UpdateWeightFromSharedMemory(WeightTransferProtocol):
    """
    Host weight sync through shared memory, for engines on the trainer's host. Each gathered bucket goes
    to the replica rank with the fewest bytes so far, which copies it flattened into a shared-memory tensor
    once the collective gathers are done, and hand the engines its handle through
    /update_weights_from_tensor; each engine maps it in place, with no file written or parsed.
    A bucket is published as soon as it is in shared memory, so the copies overlap the engine loads, and only
    a few shared-memory buckets are alive at once (`_MAX_BUCKETS_IN_FLIGHT_PER_HOST`).
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        # each published host copy with its engine calls; a copy lives until every engine has loaded it
        self._in_flight: deque[tuple[torch.Tensor, list[Future]]] = deque()

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
        self._owned_buckets = OwnedBuckets(replica_rank=replica_rank, replica_size=replica_size)
        self._max_in_flight = max(1, _MAX_BUCKETS_IN_FLIGHT_PER_HOST // replica_size)

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        self._owned_buckets.reset()
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        self._owned_buckets.offer(bucket)

    def after_base_weights(self) -> None:
        # published only after the collective gathers finish, so the ranks' copies run in parallel
        cell_updaters = list(self.cell_updaters_of_cell_id.values())
        with fail_on_every_rank("[shm] bucket publish"):
            for _, bucket in self._owned_buckets.take():
                if len(self._in_flight) == self._max_in_flight:
                    self._wait_for_oldest_bucket()
                host_tensor, metadata = _copy_to_shared_bucket(bucket)
                with _file_system_sharing():
                    futures = self._publish(cell_updaters, {"flattened_tensor": host_tensor, "metadata": metadata})
                self._in_flight.append((host_tensor, futures))

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
            while self._in_flight:
                self._wait_for_oldest_bucket()

    def _wait_for_oldest_bucket(self) -> None:
        _, futures = self._in_flight.popleft()
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


def _copy_to_shared_bucket(bucket: list[tuple[str, torch.Tensor]]) -> tuple[torch.Tensor, list]:
    """The host bucket as one shared-memory byte tensor in the engine's flattened layout, with one copy per tensor."""
    metadata = flatten_weight_bucket([(name, tensor.to("meta")) for name, tensor in bucket]).get_metadata()
    with _file_system_sharing():
        host_tensor = torch.empty(metadata[-1].end_idx, dtype=torch.uint8).share_memory_()
    for (_, tensor), meta in zip(bucket, metadata, strict=True):
        host_tensor[meta.start_idx : meta.end_idx].view(tensor.dtype).view(tensor.shape).copy_(tensor)
    return host_tensor, metadata


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
