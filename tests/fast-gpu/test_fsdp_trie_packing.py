"""Trie attention on the real flash-attn kernels, each installed one of FA2, FA3 and FA4: every segment of a trie
row must attend over its gathered keys as the plain-PyTorch reference does, with full and sliding-window attention,
and a row that many segments gather must get one fp32 sum of its gradient copies.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper"], num_gpus=1)

import random

import pytest
import torch
from tests.fast.backends.training_utils.reference_trie_attention import reference_trie_attention

from miles.backends.fsdp_utils.adaptations.trie_attention import (
    FLASH_ATTN_MODULES,
    TrieAttentionArgs,
    trie_attention_forward,
)
from miles.backends.fsdp_utils.adaptations.trie_gather import gather_rows
from miles.backends.training_utils.data.trie_packing import WINDOW_START_MULTIPLE, build_trie_layout, pack_trie_bins

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 512
_WINDOW = 24


def _samples(seed: int) -> dict:
    """Two prompts of three samples, a shared system prompt; prompts longer than the window plus flash-attn's
    128-key tile, so window gathers drop ancestors."""
    rng = random.Random(seed)
    system = [rng.randrange(1, _VOCAB) for _ in range(180)]
    tokens, total_lengths, response_lengths = [], [], []
    for _ in range(2):
        prompt = system + [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(10, 30))]
        for _ in range(3):
            response = [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(5, 20))]
            tokens.append(torch.tensor(prompt + response))
            total_lengths.append(len(prompt) + len(response))
            response_lengths.append(len(response))
    return {
        "tokens": tokens,
        # on the GPU, as the actor's rollout data is
        "loss_masks": [torch.ones(r, dtype=torch.int, device="cuda") for r in response_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


def test_gather_rows_sums_a_shared_rows_copies_in_fp32():
    """A row gathered by many segments gets one fp32 sum of its copies' gradients, rounded to bf16 once, where
    index_select's own backward rounds after every copy (CUDA's atomic bf16 adds) and lands far from the sum."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    source = torch.randn(4, 2, 64, device="cuda", generator=generator).bfloat16()
    rows = torch.tensor([0] * 2000 + [1, 2, 3], device="cuda")
    grad = torch.randn(len(rows), 2, 64, device="cuda", generator=generator).bfloat16()
    exact = torch.zeros(source.shape, dtype=torch.float64, device="cuda").index_add_(0, rows, grad.double())

    assert torch.equal(gather_rows(source, rows), source.index_select(0, rows))
    errors = {}
    for name, gather in (
        ("gather_rows", gather_rows),
        ("index_select", lambda leaf, rows: leaf.index_select(0, rows)),
    ):
        leaf = source.clone().requires_grad_()
        gather(leaf, rows).backward(grad)
        assert leaf.grad.dtype == torch.bfloat16
        errors[name] = ((leaf.grad.double() - exact).norm() / exact.norm()).item()
    # measured: gather_rows about 1.7e-3 (one bf16 rounding), index_select 28-40 times that
    assert errors["gather_rows"] < 2**-8, errors
    assert errors["index_select"] > 10 * errors["gather_rows"], errors


@pytest.mark.parametrize("attn_implementation", sorted(FLASH_ATTN_MODULES))
@pytest.mark.parametrize("window", [None, _WINDOW], ids=["full", "sliding"])
def test_flash_attn_trie_attention_matches_the_reference(window, attn_implementation):
    flash_attn_module = FLASH_ATTN_MODULES[attn_implementation]
    pytest.importorskip(flash_attn_module)
    tokens = [t.tolist() for t in _samples(2)["tokens"]]
    (bin_trie,) = pack_trie_bins(tokens, max_bin_width=sum(len(t) for t in tokens))
    layout = build_trie_layout(
        bin_trie,
        window_left=None if window is None else window - 1,
        window_start_multiple=WINDOW_START_MULTIPLE,
        pad_multiple=128,
    )
    if window is not None:
        assert layout.window_kv.index.numel() < layout.kv.index.numel(), "no window gather dropped an ancestor"
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, layout.num_tokens, 64, device="cuda") for heads in (4, 2, 2))
    trie = TrieAttentionArgs(layout, torch.device("cuda"))

    got, _ = trie_attention_forward(
        None,
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        None,
        sliding_window=window,
        flash_attn_module=flash_attn_module,
        miles_trie_attn_args=trie,
    )
    kv = layout.kv if window is None else layout.window_kv
    want = reference_trie_attention(
        *(x[0].transpose(0, 1).double().cpu() for x in (q, k, v)),
        kv,
        layout.cu_seqlens_q,
        window_left=None if window is None else window - 1,
    )
    torch.testing.assert_close(got[0].double().cpu(), want, rtol=0, atol=2e-2)
