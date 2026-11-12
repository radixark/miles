"""Transformer Engine's FlashAttention-4 path against fp32 torch SDPA on packed (varlen) sequences.

TE hands FA4 its causal window (-1, 0) together with causal=True. flash-attn-4 >= 4.0.0b30 reads that as
a local window with left=-1 unless miles/backends/megatron_utils/te_fa4_window.py translates it.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600, suite="stage-c-4-gpu-b200", labels=["precision", "megatron"], hardware=["blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="TE selects FlashAttention-4 on SM100/SM103",
)

from transformer_engine.pytorch import DotProductAttention  # noqa: E402
from transformer_engine.pytorch.attention.dot_product_attention import backends  # noqa: E402

from miles.backends.megatron_utils.te_fa4_window import patch_te_flash_attn_4_window  # noqa: E402

MAX_ABS_ERROR = 0.05
SEQLENS = [1, 37, 128, 300]
NUM_HEADS = 8
NUM_KV_HEADS = 2


@pytest.fixture(scope="module", autouse=True)
def _te_fa4_window_patch() -> None:
    patch_te_flash_attn_4_window()
    assert backends.flash_attn_varlen_func_v4 is not None, "Transformer Engine did not import flash-attn-4"


def _packed_qkv(head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(head_dim)
    total = sum(SEQLENS)

    def randn(num_heads: int) -> torch.Tensor:
        return torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)

    cu_seqlens = torch.tensor([0, *itertools.accumulate(SEQLENS)], device="cuda", dtype=torch.int32)
    return randn(NUM_HEADS), randn(NUM_KV_HEADS), randn(NUM_KV_HEADS), cu_seqlens


def _sdpa_reference(q, k, v, *, window_left: int | None = None) -> torch.Tensor:
    """Per-sequence fp32 causal SDPA; window_left, when set, also bounds how far back a query attends."""
    outputs = []
    for start, seqlen in zip(itertools.accumulate([0, *SEQLENS]), SEQLENS, strict=False):
        q_seq, k_seq, v_seq = (t[start : start + seqlen].float().transpose(0, 1) for t in (q, k, v))
        k_seq, v_seq = (t.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=0) for t in (k_seq, v_seq))
        position = torch.arange(seqlen, device="cuda")
        distance = position[:, None] - position[None, :]
        allowed = distance >= 0
        if window_left is not None:
            allowed &= distance <= window_left
        outputs.append(F.scaled_dot_product_attention(q_seq, k_seq, v_seq, attn_mask=allowed).transpose(0, 1))
    return torch.cat(outputs)


def _fa4_varlen(q, k, v, cu_seqlens, *, window_size: tuple[int, int]) -> torch.Tensor:
    """The call TE's FlashAttention.forward makes for thd input."""
    output = backends.flash_attn_varlen_func_v4(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(SEQLENS),
        max_seqlen_k=max(SEQLENS),
        causal=True,
        window_size=window_size,
    )
    return output[0] if isinstance(output, tuple) else output


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    error = (actual.float() - expected).abs().max().item()
    assert error < MAX_ABS_ERROR, f"{name}: max abs error {error:.4f} >= {MAX_ABS_ERROR}"


def _assert_forward_and_backward_match(run_attention, head_dim: int) -> None:
    q, k, v, cu_seqlens = _packed_qkv(head_dim)
    inputs = [t.requires_grad_() for t in (q, k, v)]
    reference_inputs = [t.detach().float().requires_grad_() for t in (q, k, v)]

    output = run_attention(*inputs, cu_seqlens).reshape(q.shape)
    reference = _sdpa_reference(*reference_inputs)
    _assert_close(output, reference, "out")

    # GQA sums each K/V gradient over the query heads it serves; this scale keeps the gradients at the
    # output's magnitude, so one absolute bound covers both.
    grad_output = 0.25 * torch.randn_like(output)
    output.backward(grad_output)
    reference.backward(grad_output.float())
    for name, tensor, reference_tensor in zip(("dq", "dk", "dv"), inputs, reference_inputs, strict=True):
        _assert_close(tensor.grad, reference_tensor.grad, name)


@pytest.mark.parametrize("head_dim", [128, 256])
def test_fa4_with_te_causal_window_matches_sdpa(head_dim: int) -> None:
    """FA4 varlen with causal=True and TE's (-1, 0) window is causal attention, forward and backward."""
    _assert_forward_and_backward_match(
        lambda q, k, v, cu_seqlens: _fa4_varlen(q, k, v, cu_seqlens, window_size=(-1, 0)), head_dim
    )


@pytest.mark.parametrize("head_dim", [128, 256])
def test_te_flash_backend_matches_sdpa(monkeypatch: pytest.MonkeyPatch, head_dim: int) -> None:
    """TE DotProductAttention on thd input with the flash backend forced, as --attention-backend flash runs it."""
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    windows_seen = []
    fa4_varlen = backends.flash_attn_varlen_func_v4

    def record_window(*args, **kwargs):
        windows_seen.append(kwargs["window_size"])
        return fa4_varlen(*args, **kwargs)

    monkeypatch.setattr(backends, "flash_attn_varlen_func_v4", record_window)
    attention = DotProductAttention(
        num_attention_heads=NUM_HEADS,
        kv_channels=head_dim,
        num_gqa_groups=NUM_KV_HEADS,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    )

    def run_attention(q, k, v, cu_seqlens):
        return attention(
            q,
            k,
            v,
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(SEQLENS),
            max_seqlen_kv=max(SEQLENS),
            attn_mask_type="padding_causal",
        )

    _assert_forward_and_backward_match(run_attention, head_dim)
    assert [tuple(window) for window in windows_seen] == [(-1, 0)], f"expected one FA4 call, got {windows_seen}"


def test_fa4_sliding_window_matches_sdpa() -> None:
    """A genuine causal sliding window still reaches FA4 as a local window."""
    window_left = 64
    q, k, v, cu_seqlens = _packed_qkv(head_dim=128)

    output = _fa4_varlen(q, k, v, cu_seqlens, window_size=(window_left, 0))

    _assert_close(output, _sdpa_reference(q, k, v, window_left=window_left), "out")
