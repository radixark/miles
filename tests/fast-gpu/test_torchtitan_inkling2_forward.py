from tests.ci.ci_register import register_cuda_ci

# flex attention returns the log-sum-exp the attention residuals read only on GPU
register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["torchtitan"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("torchtitan")

from tests.fast.backends.inkling2_tiny import TEXT_CONFIG, as_trained, random_model  # noqa: E402


def test_a_packed_document_does_not_see_the_one_before_it(tmp_path):
    _, model = random_model(tmp_path)
    model = as_trained(model).cuda()
    first, second = 12, 10
    positions = torch.cat([torch.arange(first), torch.arange(second)]).cuda()
    tokens = torch.randint(0, TEXT_CONFIG["unpadded_vocab_size"], (first + second,), device="cuda")
    other_first = tokens.clone()
    other_first[:first] = torch.randint(0, TEXT_CONFIG["unpadded_vocab_size"], (first,), device="cuda")
    with torch.no_grad():
        logits = model(tokens, positions)
        logits_other_first = model(other_first, positions)
    assert logits.shape[-1] == TEXT_CONFIG["unpadded_vocab_size"]
    assert logits.abs().max() <= TEXT_CONFIG["final_logit_softcapping"]
    torch.testing.assert_close(logits[first:], logits_other_first[first:])
    assert not torch.allclose(logits[:first], logits_other_first[:first])
