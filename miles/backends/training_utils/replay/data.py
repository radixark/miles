import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from miles.backends.training_utils.data.context_parallel import CPLayout, take_ranges
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.output_store.plan import ForwardSegments, RowRead, StorePlanError, is_segments_column
from miles.utils.replay_base import BaseReplayManager

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore, ReadBuffer


def register_replay_list_sequential(replay_list, replay_data, *, blocks, **_kwargs):
    """Map replay streams to registered modules.

    `replay_data[b]` holds streams `blocks[b]`. Each replay records stream `replay.stream_idx` if
    set (used for sparse layer layouts under PP/VPP, where the global replay tensor contains more
    streams than this rank registered). Otherwise falls back to 1:1 enumeration order.
    """
    for replay_idx, replay in enumerate(replay_list):
        stream_idx = replay.stream_idx if replay.stream_idx is not None else replay_idx
        block, column = locate_stream(blocks, stream_idx)
        replay.record(replay_data[block][:, column])


def locate_stream(blocks: Sequence[tuple[int, int]], stream: int) -> tuple[int, int]:
    """The block holding `stream`, and its column there."""
    for i, (start, end) in enumerate(blocks):
        if start <= stream < end:
            return i, stream - start
    raise AssertionError(f"replay stream {stream} out of range (this rank's replay rows hold streams {blocks})")


def stream_blocks(replays) -> list[tuple[int, int]]:
    """The streams `register_replay_list_sequential` records for `replays`, as ascending runs `(start, end)`."""
    blocks = []
    for stream in sorted({r.stream_idx if r.stream_idx is not None else i for i, r in enumerate(replays)}):
        if blocks and blocks[-1][1] == stream:
            blocks[-1] = (blocks[-1][0], stream + 1)
        else:
            blocks.append((stream, stream + 1))
    return blocks


class ReadSlots:
    """The ReadBuffers a replay manager reads output store rows into.

    Kept across steps, so each is registered once; a slot too small for a micro batch is replaced
    by one that fits. Registering per read would take about half of each read, and over 80% below
    4 GiB; measurements: https://github.com/radixark/miles/pull/4573#issuecomment-6593240538
    """

    def __init__(self, store: "OutputStore", *, count: int):
        self.store = store
        self._buffers: list[ReadBuffer | None] = [None] * count

    def get(self, index: int, nbytes: int) -> "ReadBuffer":
        buffer = self._buffers[index]
        if buffer is None or buffer.nbytes < nbytes:
            if buffer is not None:
                buffer.close()
            buffer = self._buffers[index] = self.store.read_buffer(nbytes)
        return buffer

    def close(self) -> None:
        for buffer in self._buffers:
            if buffer is not None:
                buffer.close()


def attach_replay_data(
    manager: BaseReplayManager,
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    output_store=None,
    sequence_first=False,
    align=None,
    blocks=None,
):
    """Load rollout replay tensors into `manager`'s replay queues, one micro batch per `fill(index)`.

    `rollout_data[manager.data_key]` contains one tensor per sample with shape
    `[num_tokens - 1, num_streams, topk]`, or, from the output store, each sample's
    `ForwardSegments`, whose rows this rank reads per micro batch from `output_store`.
    Micro batch `index` holds the samples the log-prob and train forwards see as their
    `index`-th micro batch; their rows are padded/sliced to match the local CP/SP token
    layout, and stream-to-module mapping is delegated to `manager.register_replay_list_func`.
    The manager fills the micro batches in order on its worker thread, ahead of the forwards
    that pop them; the column is dropped once the last one has been waited for.

    `blocks` are the `(start, end)` stream ranges this rank registers (Megatron: `replay_blocks`),
    each a stored part from the output store; None means every stream, the whole stored field.
    Only their streams are read and padded, one tensor per block, and registered with `blocks=`.

    For `bshd`, `sequence_first` selects Megatron MoE's `[S, B]` token order;
    indexer replay consumes the default `[B, S]` order. `align`, if given, is
    applied to each micro batch's rows before they are registered.
    """
    data_key = manager.data_key
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")

    order = data_iterator[0].microbatch_indices(sum(num_microbatches))
    column = rollout_data[data_key]
    from_store = is_segments_column(column)
    if from_store:
        if output_store is None:
            raise StorePlanError(f"{data_key} is in the output store, but this trainer has no output store client")
        if manager.read_slots is None or manager.read_slots.store is not output_store:
            if manager.read_slots is not None:
                manager.read_slots.close()
            manager.read_slots = ReadSlots(output_store, count=1)
    if blocks is None:
        blocks = (column[0].parts() if from_store else [(0, column[0].shape[1])]) if column else []
    # A train-data dump, written after training, replaces the column with each sample's local rows
    # (after CP padding, before packing) of every block in turn, what replay consumed on this rank;
    # None for a sample no micro batch holds, which a fixed schedule leaves when the batch does not
    # divide evenly.
    keep_column = manager.kept_in_train_dump and args.save_debug_train_data is not None
    kept_rows = {}

    parallel_state = get_parallel_state()
    tp_rank = parallel_state.tp.rank
    tp_size = parallel_state.tp.size
    qkv_format = args.qkv_format

    def pad_func(data, pad):
        _, num_layers, topk = data.shape
        pad_tensor = torch.full(
            (pad, num_layers, topk),
            fill_value=-1,
            device=data.device,
            dtype=data.dtype,
        )
        return torch.cat([data, pad_tensor], dim=0)

    def fill(index):
        samples = [column[j] for j in order[index]]
        tokens = [rollout_data["tokens"][j] for j in order[index]]
        for a, b in zip(samples, tokens, strict=False):
            rows = a.num_rows() if from_store else a.shape[0]
            assert rows == b.shape[0] - 1, f"{rows}, {b.shape}"

        # Row p replays token p. The final token has no row, so it is padded like the CP padding; the
        # padded positions are masked from loss.
        max_seq_len = rollout_data["max_seq_lens"][order[index][0]] if qkv_format == "bshd" else None
        pad_size = None if qkv_format == "bshd" else parallel_state.tp.size * args.data_pad_size_multiplier
        layout = CPLayout.build(
            qkv_format=qkv_format,
            allgather_cp=args.allgather_cp,
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
            max_seq_len=max_seq_len,
            pad_size=pad_size,
        )
        local_ranges = layout.local_ranges([t.size(0) for t in tokens])
        if from_store:
            # A rank that replays nothing, e.g. a pipeline stage of dense layers only, may read all-zero rows.
            rows = _read_local_rows(
                manager.read_slots, samples, local_ranges, blocks=blocks, check=bool(manager.replays)
            )
        else:
            rows = [
                [r[:, start:end] for start, end in blocks]
                for r in (take_ranges(s, ranges, pad_func) for s, ranges in zip(samples, local_ranges, strict=True))
            ]
        if keep_column:
            # A copy: store rows are views of a slot the next micro batch overwrites.
            kept_rows.update((j, torch.cat(r, dim=1)) for j, r in zip(order[index], rows, strict=True))

        replay_data = [pack([r[i] for r in rows], layout, pad_size) for i in range(len(blocks))]
        manager.register_replay_list_func(manager.replays, replay_data, models=models, blocks=blocks)

    def pack(replay_data, layout, pad_size):
        """One block's rows of the micro batch's samples, in the token layout its forward consumes."""
        if qkv_format == "bshd":
            replay_data = torch.stack(replay_data, dim=0)
            batch_size, seqlen, num_layers, topk = replay_data.shape
            if sequence_first:
                replay_data = replay_data.transpose(0, 1)
            replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
        elif layout.per_pack:
            # The ranges already cover this rank's piece of the padded concatenation.
            replay_data = torch.cat(replay_data, dim=0)
        else:
            if manager.replay_indices_are_token_positions:
                # map indices to thd format
                offset = 0
                for i, r in enumerate(replay_data):
                    replay_data[i] = torch.where(r != -1, r + offset, r)
                    offset += r.shape[0]
            replay_data = torch.cat(replay_data, dim=0)
            pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
            if pad != 0:
                replay_data = pad_func(replay_data, pad)

        # sequence_parallel is Megatron-only; FSDP has tp_size == 1 so the slice is a no-op there.
        if args.train_backend == "megatron" and args.backend.sequence_parallel and manager.if_sp_region:
            seqlen = replay_data.size(0)
            assert seqlen % tp_size == 0
            start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
            replay_data = replay_data[start:end]

        if align is not None:
            replay_data = align(replay_data, -1)
        return replay_data

    def finish():
        # On the main thread: it may be iterating rollout_data while the worker fills.
        if keep_column:
            rollout_data[data_key] = [kept_rows.get(j) for j in range(len(column))]
        else:
            del rollout_data[data_key]

    manager.attach(fill, len(order), finish)


def _read_local_rows(
    slots: ReadSlots,
    samples: Sequence[ForwardSegments],
    local_ranges: Sequence[Sequence[tuple[int, int]]],
    *,
    blocks: Sequence[tuple[int, int]],
    check: bool,
) -> list[list[torch.Tensor]]:
    """Each sample's rows at this rank's local positions, one tensor per block, read in one call into a reused slot.

    The slot holds each block's part in turn, every sample's local rows of it. Positions past a sample's
    last row are padded with -1, as `take_ranges` pads inline rows. With `check`, the rows read from each
    object are checked, those of every block together.
    """
    if not blocks:
        # This rank registers no stream.
        return [[] for _ in samples]
    # radix_transfer_plugins ships with SGLang and is not a Miles dependency.
    from radix_transfer_plugins import RangeRead

    counts = [sum(end - start for start, end in ranges) for ranges in local_ranges]
    regions, reads, offset = [], [], 0
    for block in blocks:
        region = _BlockRegion.plan(samples, local_ranges, counts, block=block, offset=offset)
        regions.append(region)
        reads += [(block, r) for r in region.reads]
        offset += sum(counts) * region.row_bytes
    buffer = slots.get(0, offset)
    slots.store.read_ranges(
        [
            RangeRead(
                ref=r.ref,
                field=samples[0].part_name(block),
                src_offset=r.src_offset,
                nbytes=r.nbytes,
                dst_offset=r.dst_offset,
            )
            for block, r in reads
        ],
        into=buffer,
    )
    rows = [region.rows(buffer, num_rows=sum(counts)) for region in regions]
    if check:
        # Every region holds the same local rows: check those of each object read of the first block in all.
        first = regions[0]
        for r in first.reads:
            start = (r.dst_offset - first.offset) // first.row_bytes
            end = start + r.nbytes // first.row_bytes
            samples[0].check_rows([block_rows[start:end] for block_rows in rows])
    by_block = [torch.from_numpy(block_rows).split(counts) for block_rows in rows]
    return [list(sample_rows) for sample_rows in zip(*by_block, strict=True)]


class _BlockRegion(NamedTuple):
    """Where one block's rows of a micro batch's samples land in a slot, and the reads that put them there."""

    offset: int
    dtype: np.dtype
    row_shape: tuple[int, ...]
    row_bytes: int
    reads: list[RowRead]
    pads: list[tuple[int, int]]

    @classmethod
    def plan(cls, samples, local_ranges, counts, *, block, offset) -> "_BlockRegion":
        formats = {segments.row_format(block) for segments in samples}
        if len(formats) != 1:
            raise StorePlanError(f"a micro batch's {samples[0].name} rows come in {len(formats)} formats: {formats}")
        ((dtype, row_shape),) = formats
        row_bytes = dtype.itemsize * math.prod(row_shape)
        reads, pads, first = [], [], 0
        for segments, ranges, count in zip(samples, local_ranges, counts, strict=True):
            sample_reads, sample_pads = segments.range_reads(
                ranges, dst_offset=offset + first * row_bytes, block=block
            )
            reads += sample_reads
            pads += [(first + start, first + end) for start, end in sample_pads]
            first += count
        return cls(offset, dtype, row_shape, row_bytes, reads, pads)

    def rows(self, buffer: "ReadBuffer", *, num_rows: int) -> np.ndarray:
        """The region's rows in `buffer` once read, padding filled."""
        count = num_rows * math.prod(self.row_shape)
        rows = np.frombuffer(buffer.view, dtype=self.dtype, count=count, offset=self.offset)
        rows = rows.reshape(num_rows, *self.row_shape)
        for start, end in self.pads:
            rows[start:end] = -1
        return rows
