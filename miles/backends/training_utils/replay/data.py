import torch

from miles.backends.training_utils.data.context_parallel import CPLayout, take_ranges
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.replay_base import BaseReplayManager


def register_replay_list_sequential(replay_list, replay_data, **_kwargs):
    """Map replay streams to registered modules.

    Each replay records `replay_data[:, replay.stream_idx]` if `stream_idx` is
    set (used for sparse layer layouts under PP/VPP, where the global replay
    tensor contains more streams than this rank registered). Otherwise falls
    back to 1:1 enumeration order.
    """
    for replay_idx, replay in enumerate(replay_list):
        stream_idx = replay.stream_idx if replay.stream_idx is not None else replay_idx
        if not 0 <= stream_idx < replay_data.shape[1]:
            raise AssertionError(
                f"replay stream_idx {stream_idx} out of range " f"(replay_data has {replay_data.shape[1]} streams)"
            )
        replay.record(replay_data[:, stream_idx])


def attach_replay_data(
    manager: BaseReplayManager,
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    sequence_first=False,
    align=None,
):
    """Load rollout replay tensors into `manager`'s replay queues, one micro batch per `fill(index)`.

    `rollout_data[manager.data_key]` contains one tensor per sample with shape
    `[num_tokens - 1, num_streams, topk]`. Micro batch `index` holds the samples
    the log-prob and train forwards see as their `index`-th micro batch; their
    tensors are padded/sliced to match the local CP/SP token layout, and
    stream-to-module mapping is delegated to `manager.register_replay_list_func`.
    Every micro batch is filled before this returns.

    For `bshd`, `sequence_first` selects Megatron MoE's `[S, B]` token order;
    indexer replay consumes the default `[B, S]` order. `align`, if given, is
    applied to each recorded entry.
    """
    data_key = manager.data_key
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")

    order = data_iterator[0].microbatch_indices(sum(num_microbatches))
    # A train-data dump, written after training, keeps the column so it holds what replay consumed.
    keep_column = manager.kept_in_train_dump and args.save_debug_train_data is not None

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
        replay_data = [rollout_data[data_key][j] for j in order[index]]
        tokens = [rollout_data["tokens"][j] for j in order[index]]
        for a, b in zip(replay_data, tokens, strict=False):
            assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

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
        replay_data = [take_ranges(r, ranges, pad_func) for r, ranges in zip(replay_data, local_ranges, strict=True)]

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

        manager.register_replay_list_func(manager.replays, replay_data, models=models)
        if align is not None:
            for replay in manager.replays:
                replay.top_indices_list[-1] = align(replay.top_indices_list[-1], -1)
        if index == len(order) - 1 and not keep_column:
            del rollout_data[data_key]

    manager.attach(fill)
    manager.fill_through(len(order) - 1)
