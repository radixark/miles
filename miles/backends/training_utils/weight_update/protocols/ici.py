"""Device-to-device weight sync over the ICI links of a TPU slice the trainer shares with one engine.

Trainer rank i sends each bucket to engine TP rank i as one packed buffer, and posts the bucket's names to the
slice's store, where the engine's receiver (sglang_tpu/shared_slice.py) reads them. The slice comes from
miles.platforms.tpu_shared_slice.
"""

import json
import time
import zlib
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.platforms import current_platform
from miles.platforms.tpu_shared_slice import ENGINE_BACKEND_NAME
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group

# Many small sends crash libtpu's CopyToRemoteDevice, so a bucket travels as one buffer, each tensor in its own
# 4 KiB-aligned region. int16, not bytes: a bf16 tensor viewed as a 1-byte type pads to a full TPU tile.
_BUFFER_DTYPE = torch.int16
_REGION_WORDS = 4096 // _BUFFER_DTYPE.itemsize


class UpdateWeightFromIci(WeightTransferProtocol):
    # every trainer rank sends whole tensors, so it must hold every PP stage's
    required_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-ici"
        # buffers of sends the engine never received: a posted send's buffer must outlive the process
        self._unreceived_buffers: list[torch.Tensor] = []

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
        shared_slice = current_platform.shared_slice
        assert shared_slice is not None, "the trainer did not join the shared slice"
        trainer_size = dist.get_world_size()
        # miles_validate_args checked the layout; discovery checked the engine against it
        assert list(engine_gpu_counts) == [trainer_size], f"engine devices {engine_gpu_counts}"
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.is_sender = True
        self._selector = selector
        self._shared_slice = shared_slice
        self._src_slice_rank = dist.get_rank()
        self._dst_slice_rank = trainer_size + dist.get_rank()
        # a fresh engine receiver numbers its buckets from 1; torch_tpu sequences repeated tags per peer
        self._tag = 0
        if dist.get_rank() == 0:
            futures = [
                updater.submit_client_call(
                    "init_weights_update_group",
                    master_address=shared_slice.spec.store_host,
                    master_port=shared_slice.spec.store_port,
                    rank_offset=0,
                    world_size=shared_slice.spec.size,
                    group_name=self.group_name,
                    backend=ENGINE_BACKEND_NAME,
                )
                for updater in self.cell_updaters_of_cell_id.values()
            ]
            check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        names = [name for name, _ in bucket]
        dtypes = [tensor.dtype for _, tensor in bucket]
        shapes = [tensor.shape for _, tensor in bucket]
        _check_same_layout_on_every_rank(dtypes, shapes)
        buffer = pack_bucket([tensor for _, tensor in bucket])
        self._tag += 1
        futures = []
        if dist.get_rank() == 0:
            # posts the engine's receives; the send below blocks until they are posted
            futures = [
                updater.submit_client_call(
                    "update_weights_from_distributed",
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=self.group_name,
                    selector=self._selector,
                )
                for updater in self.cell_updaters_of_cell_id.values()
            ]
        self._shared_slice.store.set(
            bucket_names_key(self._src_slice_rank, self._dst_slice_rank, self._tag), json.dumps(names)
        )
        send = self._shared_slice.backend.send([buffer], self._dst_slice_rank, self._tag)
        error = None
        try:
            self._wait_until_received(send, buffer)
            check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)
        except Exception as send_or_engine_error:
            error = send_or_engine_error
        # all ranks fail together, and keep their buffer until the engine replied: send's wait() returns early
        any_rank_failed = torch.tensor([error is not None], dtype=torch.int32)
        dist.all_reduce(any_rank_failed, group=get_gloo_group())
        if error is not None:
            raise error
        if any_rank_failed.item():
            raise RuntimeError("ICI weight sync failed on another trainer rank, which raised the cause")

    def _wait_until_received(self, send: dist.Work, buffer: torch.Tensor) -> None:
        """Wait for the engine rank to take the send; an engine that rejected rank 0's request never posts its receive.

        torch_tpu's ``Work.wait`` ignores its timeout, so a send to such an engine would block for the slice
        backend's 30 minutes.
        """
        timeout = self.args.update_weight_engine_request_timeout
        deadline = time.monotonic() + timeout
        while not send.is_completed():
            if time.monotonic() > deadline:
                self._unreceived_buffers.append(buffer)
                raise TimeoutError(
                    f"engine slice rank {self._dst_slice_rank} did not receive bucket {self._tag} within {timeout} s"
                )
            time.sleep(0.001)
        send.wait()  # raises the transport error of a failed send


def _check_same_layout_on_every_rank(dtypes: list[torch.dtype], shapes: list[torch.Size]) -> None:
    """Every engine rank reads its paired rank's buffer with the dtypes and shapes of rank 0's request, and only the
    names come per rank (from the store). A rank packing another layout, such as its own experts, would load another
    tensor's bytes; every rank raises before anything is sent."""
    layout = zlib.crc32(
        repr([(str(dtype), tuple(shape)) for dtype, shape in zip(dtypes, shapes, strict=True)]).encode()
    )
    # one MAX reduce gives both the largest and the smallest checksum
    bounds = torch.tensor([layout, -layout], dtype=torch.int64)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=get_gloo_group())
    if bounds[0].item() != -bounds[1].item():
        raise RuntimeError(
            "ICI weight sync: trainer ranks packed this bucket with different dtypes or shapes, but every engine rank "
            "reads it with rank 0's"
        )


def bucket_names_key(src_slice_rank: int, dst_slice_rank: int, tag: int) -> str:
    # the engine reads and deletes it (sglang_tpu.shared_slice.bucket_names_key)
    return f"bucket_names:{src_slice_rank}:{dst_slice_rank}:{tag}"


def pack_bucket(tensors: list[torch.Tensor]) -> torch.Tensor:
    """One int16 buffer holding each tensor's bits in its own zero-padded 4 KiB-aligned region."""
    pieces = []
    for tensor in tensors:
        assert tensor.dtype.itemsize % _BUFFER_DTYPE.itemsize == 0, f"cannot pack {tensor.dtype} into int16 words"
        words = tensor.detach().reshape(-1).view(_BUFFER_DTYPE)
        # an empty tensor still takes a region, as the engine lays it out
        num_regions = max(1, -(-words.numel() // _REGION_WORDS))
        pieces += [words, words.new_zeros(num_regions * _REGION_WORDS - words.numel())]
    # one concatenation: on TPU, copying each tensor into a slice of a preallocated buffer rewrites the whole buffer
    return torch.cat(pieces)
