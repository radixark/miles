import torch

from miles_plugins.models.deepseek_v41.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v41.ops.rope import apply_rotary_emb

_PAGE_SIZE = 64
_BYTES_PER_TOKEN = 584
_ROW_BYTES = 576


def kv_norm_rope(
    kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    x = kv.float()
    data = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).contiguous()
    apply_rotary_emb(data[..., -rope_dim:], freqs_cis)
    return data.to(kv.dtype)


@torch.no_grad()
def kv_cache_roundtrip(kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla

    bsz, seqlen, dim = kv.shape
    flat = kv.reshape(bsz * seqlen, dim).contiguous()
    positions = torch.arange(seqlen, device=kv.device, dtype=torch.int32).repeat(bsz)
    loc = torch.arange(bsz * seqlen, device=kv.device, dtype=torch.int32)
    pages = (bsz * seqlen + _PAGE_SIZE) // _PAGE_SIZE + 1
    page_bytes = -(-(_BYTES_PER_TOKEN * _PAGE_SIZE) // _ROW_BYTES) * _ROW_BYTES
    cache = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=kv.device)
    fused_k_norm_rope_flashmla(
        kv=flat,
        kv_weight=weight.to(kv.dtype).contiguous(),
        eps=eps,
        freqs_cis=freqs_cis.contiguous(),
        positions=positions,
        out_loc=loc,
        kvcache=cache,
        page_size=_PAGE_SIZE,
    )
    return dequantize_k_cache_paged(cache, loc, _PAGE_SIZE).reshape(bsz, seqlen, dim).to(kv.dtype)


def kv_norm_rope_fp8(
    kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    soft = kv_norm_rope(kv, weight, eps, freqs_cis, rope_dim).clone()
    soft[..., :-rope_dim] = fp8_simulate_qat(soft[..., :-rope_dim], 64)
    hard = kv_cache_roundtrip(kv, weight, eps, freqs_cis)
    return soft + (hard - soft).detach()


@torch.no_grad()
def compressed_kv_cache_roundtrip(latent: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged

    bsz, seqlen, dim = latent.shape
    flat = latent.reshape(bsz * seqlen, dim).contiguous()
    loc = torch.arange(bsz * seqlen, device=latent.device, dtype=torch.int64)
    pages = (bsz * seqlen + _PAGE_SIZE) // _PAGE_SIZE + 1
    page_bytes = -(-(_BYTES_PER_TOKEN * _PAGE_SIZE) // _ROW_BYTES) * _ROW_BYTES
    cache = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=latent.device)
    fused_store_cache(input=flat, cache=cache, indices=loc, page_size=_PAGE_SIZE, type="flashmla")
    return dequantize_k_cache_paged(cache, loc, _PAGE_SIZE).reshape(bsz, seqlen, dim).to(latent.dtype)


def compressed_kv_stored(latent: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The compressed latent as the engine's fp8 pool returns it: the fp4/E4M3 values are requantized to fp8 per 64."""
    soft = latent.clone()
    soft[..., :-rope_dim] = fp8_simulate_qat(soft[..., :-rope_dim], 64)
    hard = compressed_kv_cache_roundtrip(latent)
    return soft + (hard - soft).detach()
