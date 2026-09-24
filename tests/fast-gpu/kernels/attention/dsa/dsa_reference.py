"""fp32 torch references for the DSA kernels. Inputs follow miles.kernels.attention.dsa layouts."""

import torch


def indexer_logits_ref(q, k, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Packed: q [T, H, D], k [T_kv, D], weights [T, H] -> [T, T_kv], -inf outside [ks, ke)."""
    scores = torch.einsum("thd,kd->thk", q.float(), k.float())
    scores = torch.relu(scores) * weights.float().unsqueeze(-1)
    logits = scores.sum(dim=1)
    positions = torch.arange(k.shape[0], device=q.device)
    valid = (positions >= cu_seqlen_ks.unsqueeze(1)) & (positions < cu_seqlen_ke.unsqueeze(1))
    return logits.masked_fill(~valid, float("-inf"))


def sparse_attention_ref(q, kv, indices, sm_scale, d_v, attn_sink=None):
    """q [B, S, H, Dq], kv [B, S_kv, G, Dq], indices [B, S, G, topk] (-1 = padding), attn_sink [H] or None.
    Returns out [B, S, H, d_v]."""
    q, kv = q.float(), kv.float()
    batch, seq_len, heads, _ = q.shape
    seq_len_kv, groups = kv.shape[1], kv.shape[2]
    heads_per_group = heads // groups
    outs = []
    for g in range(groups):
        q_g = q[:, :, g * heads_per_group : (g + 1) * heads_per_group]
        kv_g = kv[:, :, g]
        idx = indices[:, :, g].long()
        valid = idx != -1
        # Padded slots clamp to key 0, so accumulate hits instead of scattering booleans: a plain
        # scatter of `valid` would let a padded slot overwrite a real selection of key 0.
        hits = torch.zeros(batch, seq_len, seq_len_kv, dtype=torch.int32, device=q.device)
        selected = hits.scatter_add_(-1, idx.clamp(min=0), valid.int()) > 0
        scores = torch.einsum("bshd,bkd->bshk", q_g, kv_g) * sm_scale
        scores = scores.masked_fill(~selected.unsqueeze(2), float("-inf"))
        row_max = scores.amax(dim=-1, keepdim=True).clamp(min=-1e30)
        weights = torch.exp(scores - row_max)
        denom = weights.sum(dim=-1)
        if attn_sink is not None:
            sink_g = attn_sink.float()[g * heads_per_group : (g + 1) * heads_per_group]
            denom = denom + torch.exp(sink_g.view(1, 1, -1) - row_max.squeeze(-1))
        numer = torch.einsum("bshk,bkd->bshd", weights, kv_g[..., :d_v])
        outs.append(numer / denom.unsqueeze(-1))
    return torch.cat(outs, dim=2)


def rel_diff(a, b):
    """Cosine-distance-like metric used by the miles dumper comparator."""
    x, y = a.flatten().float(), b.flatten().float()
    denom = (x * x).sum() + (y * y).sum()
    if denom == 0:
        return 0.0
    return (1.0 - 2.0 * (x * y).sum() / denom).item()
