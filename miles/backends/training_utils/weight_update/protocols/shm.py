import base64
import contextlib
import glob
import io
import json
import os
import socket
from argparse import Namespace
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
from miles.backends.training_utils.weight_update.protocols.utils.owner import owner_is_gone, owner_tag
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.platforms import current_platform
from miles.utils import async_utils

_SHM_DIR = "/dev/shm"


class UpdateWeightFromSharedMemory(WeightTransferProtocol):
    """
    Host weight sync through shared memory, for engines on the trainer's host. Each gathered bucket goes to the
    replica rank with the fewest bytes so far, which copies it from the device straight into that bucket's
    shared-memory tensor, in the engine's flattened layout, and hands the engines its handle through
    /update_weights_from_tensor; each engine maps it in place, with no file written or parsed.
    The shared-memory tensors are kept across syncs (`_SharedImage`), so a sync neither allocates nor copies
    through other host memory; /dev/shm holds each rank's share of the model for the run.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
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
        # /update_weights_from_tensor takes one payload per engine process and picks it by TP rank
        self._engine_gpu_counts = engine_gpu_counts
        self._image = _SharedImage(_SHM_DIR)
        replica_rank, replica_size = get_data_replica_rank_and_size(parallel_state, placement)
        self._owned_buckets = OwnedBuckets(
            replica_rank=replica_rank, replica_size=replica_size, host_buffer=self._image.host_buffer
        )

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        self._owned_buckets.reset()
        self._send_error = None
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # raising mid-stream would strand the other ranks in its gathers; every rank raises it after the stream
        if self._send_error is not None:
            bucket.clear()
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
            for index, host_bucket in self._owned_buckets.take():
                metadata = flatten_weight_bucket([(name, tensor.to("meta")) for name, tensor in host_bucket])
                payload = {"flattened_tensor": self._image.flat(index), "metadata": metadata.get_metadata()}
                with _file_system_sharing():
                    self._in_flight += self._publish(cell_updaters, payload)

    def _publish(self, cell_updaters: list, payload: dict) -> list[Future]:
        return [
            updater.submit_client_call(
                "update_weights_from_tensor",
                # one serialization per receiving process: each takes its own reference to the /dev/shm file, and
                # a payload opened by several processes would release that one reference several times
                serialized_named_tensors=[_serialize_payload(payload) for _ in range(gpu_count)],
                load_format="flattened_bucket",
                selector=self._selector,
            )
            for updater, gpu_count in zip(cell_updaters, self._engine_gpu_counts, strict=True)
            # a payload no engine opens keeps its reference until the trainer exits
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
    # sglang's patch keeps torch's reducer under this name; without the patch, reduce_tensor is torch's own
    pickler.dispatch_table[torch.Tensor] = getattr(  # config-access-exempt: set only once sglang patched torch
        reductions, "_reduce_tensor_original", reductions.reduce_tensor
    )
    pickler.dump(payload)
    return base64.b64encode(buffer.getvalue()).decode()


def _engines_off_this_host(rollout_engines: Sequence[SGLangApiClient]) -> list[str]:
    return [engine.server_url for engine in rollout_engines if not _is_this_host(urlparse(engine.server_url).hostname)]


def _is_this_host(host: str) -> bool:
    # only an address of one of this machine's interfaces can be bound, IPv4 or IPv6
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for family, _, _, _, address in addresses:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((address[0], 0))
            except OSError:
                continue
        return True
    return False


class _SharedImage:
    """This rank's owned buckets in shared memory, by stream index, kept across syncs while a bucket keeps its size.

    Rewriting in place never races an engine: the next sync starts only after every load of this one has returned.
    The files live as long as the trainer, and Ray ends it, and torch's shm manager with it, by a signal no handler
    sees. So each trainer lists its files in a manifest named after it (`owner_tag`), and the next trainer on the
    host deletes the files of every manifest whose owner is gone.
    """

    def __init__(self, manifest_dir: str) -> None:
        for manifest in glob.glob(os.path.join(manifest_dir, "miles-shm-*.manifest")):
            if owner_is_gone(os.path.basename(manifest).removeprefix("miles-shm-").removesuffix(".manifest")):
                _delete_listed_files(manifest)
        self._manifest = os.path.join(manifest_dir, f"miles-shm-{owner_tag()}.manifest")
        self._buckets: dict[int, torch.Tensor] = {}
        self._files: list[str] = []

    def host_buffer(self, index: int, nbytes: int) -> torch.Tensor:
        flat = self._buckets.get(index)
        if flat is None or flat.nbytes != max(nbytes, 1):
            if flat is not None:
                current_platform.unpin_host_memory(flat)
            flat = self._buckets[index] = self._allocate(nbytes)
        return flat[:nbytes]

    def flat(self, index: int) -> torch.Tensor:
        return self._buckets[index]

    def _allocate(self, nbytes: int) -> torch.Tensor:
        with _file_system_sharing():
            # created shared, not moved there: nothing is written before the pages are reserved below
            storage = torch.UntypedStorage._new_shared(max(nbytes, 1))  # a zero-size mapping is rejected
            name = os.fsdecode(storage._share_filename_cpu_()[1])
        flat = torch.empty(0, dtype=torch.uint8).set_(storage)
        path = os.path.join(_SHM_DIR, name.lstrip("/"))
        self._files.append(path)
        # replaced whole, so a later run never reads it half-written
        with open(f"{self._manifest}.tmp", "w") as manifest:
            json.dump(self._files, manifest)
        os.replace(f"{self._manifest}.tmp", self._manifest)
        # reserve the pages now: a full /dev/shm then fails here, not with a SIGBUS on the first write
        fd = os.open(path, os.O_RDWR)
        try:
            os.posix_fallocate(fd, 0, flat.nbytes)
        finally:
            os.close(fd)
        current_platform.pin_host_memory(flat)
        return flat


def _delete_listed_files(manifest: str) -> None:
    try:
        with open(manifest) as listed:
            paths = json.load(listed)
    except FileNotFoundError:
        return
    for path in paths:
        _remove_if_present(path)
    _remove_if_present(manifest)


def _remove_if_present(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def _file_system_sharing() -> Iterator[None]:
    # the payload must name the /dev/shm file, which an unrelated process (the engine) can open; the strategy is
    # process-wide, so it is set only while allocating and serializing, and another thread sharing a tensor in
    # that window would share it this way too
    previous = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        torch.multiprocessing.set_sharing_strategy(previous)
