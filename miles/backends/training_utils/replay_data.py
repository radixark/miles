import torch

from .cp_utils import slice_with_cp
from .parallel import get_parallel_state


def fill_replay_data(
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    data_key: str,
    replay_list: list,
    register_replay_list_func,
    if_sp_region=True,
):
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

        # Pad replay data to align with the token batch's final token. The padded token is masked from loss.
        # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
        replay_data = [pad_func(r, 1) for r in replay_data]
        # TODO: maybe extract a common process function for here and get_batch?

        cp_size = parallel_state.cp.size
        cp_rank = parallel_state.cp.rank
        if qkv_format == "bshd":
            max_seqlen = batch["max_seq_lens"][0]
            if args.allgather_cp and cp_size > 1:
                assert max_seqlen % cp_size == 0, f"max_seqlen {max_seqlen} must be divisible by cp_size {cp_size}"
                local_len = max_seqlen // cp_size
                start = cp_rank * local_len
                replay_data = [pad_func(r, max_seqlen - r.size(0))[start : start + local_len] for r in replay_data]
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
            replay_data = torch.stack(replay_data, dim=0)
            batch_size, seqlen, num_layers, topk = replay_data.shape
            replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
        else:
            pad_size = parallel_state.tp.size * args.data_pad_size_multiplier
            if args.allgather_cp and cp_size > 1:
                replay_data = torch.cat(replay_data, dim=0)
                global_pad_size = cp_size * pad_size
                pad = (global_pad_size - replay_data.size(0) % global_pad_size) % global_pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)
                replay_data = replay_data.chunk(cp_size, dim=0)[cp_rank]
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
                replay_data = torch.cat(replay_data, dim=0)
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)

        if args.sequence_parallel and if_sp_region:
            seqlen = replay_data.size(0)
            assert seqlen % tp_size == 0
            start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
            replay_data = replay_data[start:end]

        register_replay_list_func(replay_list, replay_data, models)

    del rollout_data[data_key]

    for iterator in data_iterator:
        iterator.reset()
