"""The MSA block scores must equal each query's max causal logit per key block, sequence by sequence."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from miles.kernels.attention.msa import msa_block_max_scores  # noqa: E402

BLOCK = 128
D = 128


def _reference(q, k, lens, scale):
    T, H, _ = q.shape
    num_blocks = max(-(-n // BLOCK) for n in lens)
    out = torch.full((H, T, num_blocks), float("-inf"), device=q.device)
    start = 0
    for n in lens:
        logits = torch.einsum("qhd,kd->hqk", q[start : start + n].float(), k[start : start + n].float()) * scale
        future = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        logits = logits.masked_fill(future, float("-inf"))
        nb = -(-n // BLOCK)
        padded = torch.nn.functional.pad(logits, (0, nb * BLOCK - n), value=float("-inf"))
        out[:, start : start + n, :nb] = padded.view(H, n, nb, BLOCK).amax(-1)
        start += n
    return out


@pytest.mark.parametrize("lens", [[300], [128, 128], [61, 513, 3], [1, 1, 200], [1000, 129]], ids=str)
def test_block_scores_match_dense_reference(lens):
    g = torch.Generator(device="cuda").manual_seed(0)
    T, H = sum(lens), 2
    q = torch.randn(T, H, D, device="cuda", generator=g).to(torch.bfloat16)
    k = torch.randn(T, D, device="cuda", generator=g).to(torch.bfloat16)
    cu = torch.tensor([0, *lens], device="cuda").cumsum(0).to(torch.int32)
    got = msa_block_max_scores(q, k, cu, BLOCK, D**-0.5)
    want = _reference(q, k, lens, D**-0.5)
    assert torch.equal(torch.isinf(got), torch.isinf(want)), "the causal block pattern differs"
    finite = torch.isfinite(want)
    torch.testing.assert_close(got[finite], want[finite], rtol=1e-4, atol=1e-4)
