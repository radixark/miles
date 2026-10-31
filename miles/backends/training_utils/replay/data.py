from typing import Protocol

import torch

from miles.backends.training_utils.data.context_parallel import CPLayout, take_ranges
from miles.backends.training_utils.parallel import get_parallel_state


class RegisterReplayListFunc(Protocol):
    def __call__(self, replay_list: list, replay_data: torch.Tensor, **kwargs) -> None: ...


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


def fill_replay_data(
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    data_key: str,
    replay_list: list,
    register_replay_list_func: RegisterReplayListFunc,
    if_sp_region=True,
    indices_are_token_positions=False,
    sequence_first=False,
    kept_in_train_dump=False,
):
    """Load rollout replay tensors into module replay queues.

    `rollout_data[data_key]` contains one tensor per sample with shape
    `[num_tokens - 1, num_streams, topk]`. This function replays the training
    data iterator to process those tensors in the same microbatch order as
    log-prob and train forwards, pads/slices them to match the local CP/SP
    token layout, and then delegates stream-to-module mapping to
    `register_replay_list_func`.

    For `bshd`, `sequence_first` selects Megatron MoE's `[S, B]` token order;
    indexer replay consumes the default `[B, S]` order.
    """
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")

    for iterator in data_iterator:
        iterator.reset()

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

    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
        replay_data = batch[data_key]
        tokens = batch["tokens"]
        assert len(replay_data) == len(tokens)
        for a, b in zip(replay_data, tokens, strict=False):
            assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

        # Row p replays token p. The final token has no row, so it is padded like the CP padding; the
        # padded positions are masked from loss.
        layout = CPLayout.build(
            qkv_format=qkv_format,
            allgather_cp=args.allgather_cp,
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
            max_seq_len=batch["max_seq_lens"][0] if qkv_format == "bshd" else None,
            pad_size=None if qkv_format == "bshd" else parallel_state.tp.size * args.data_pad_size_multiplier,
        )
        local_ranges = layout.local_ranges([t.size(0) for t in tokens])
        replay_data = [take_ranges(r, ranges, pad_func) for r, ranges in zip(replay_data, local_ranges, strict=True)]
        if indices_are_token_positions and qkv_format == "thd" and not layout.per_pack:
            # map indices to thd format
            offset = 0
            for i, r in enumerate(replay_data):
                replay_data[i] = torch.where(r != -1, r + offset, r)
                offset += r.shape[0]
        replay_data = layout.pack(replay_data, pad_func)

        if qkv_format == "bshd":
            batch_size, seqlen, num_layers, topk = replay_data.shape
            if sequence_first:
                replay_data = replay_data.transpose(0, 1)
            replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)

        # sequence_parallel is Megatron-only; FSDP has tp_size == 1 so the slice is a no-op there.
        if args.train_backend == "megatron" and args.backend.sequence_parallel and if_sp_region:
            seqlen = replay_data.size(0)
            assert seqlen % tp_size == 0
            start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
            replay_data = replay_data[start:end]

        register_replay_list_func(replay_list, replay_data, models=models)

    # A train-data dump, written after training, keeps the column so it holds what replay consumed.
    if not (kept_in_train_dump and args.save_debug_train_data is not None):
        del rollout_data[data_key]

    for iterator in data_iterator:
        iterator.reset()
