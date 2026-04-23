"""Reference (single-device, no-CP) builders for the DSA indexer's top-k
candidate index tensors. Used by :mod:`.cp_utils` as an oracle for its
CP-aware builders when running with ``cp_size == 1``.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    def _get_window_topk_idxs():
        if start_pos >= window_size - 1:
            return torch.arange(window_size)
        elif start_pos > 0:
            return F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
        else:
            base = torch.arange(seqlen).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
            matrix = torch.where(matrix > base, -1, matrix)
            return matrix

    return _get_window_topk_idxs().unsqueeze(0).expand(bsz, -1, -1).cuda()


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    def _get_compress_topk_idxs():
        if start_pos > 0:
            return torch.arange(0, (start_pos + 1) // ratio) + offset
        else:
            matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
            mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)
            return matrix

    return _get_compress_topk_idxs().unsqueeze(0).expand(bsz, -1, -1).cuda()
