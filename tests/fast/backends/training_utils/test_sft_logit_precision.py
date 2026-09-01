import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss_hub.losses import sft_loss_function


def test_sft_graph_placeholder_survives_fp16_logits():
    # With every response token on another CP rank, the loss reduces to the
    # `0 * logits.sum()` graph placeholder; an fp16 sum overflows to inf.
    make_parallel_state()
    args = make_args(loss_type="sft_loss", true_on_policy_mode=False)
    batch = {
        "unconcat_tokens": [torch.zeros(5, dtype=torch.long)],
        "total_lengths": [5],
        "response_lengths": [0],
    }
    logits = torch.full((1, 5, 70_000), 2.0, dtype=torch.float16)
    assert torch.isinf(logits.sum())

    loss, _ = sft_loss_function(args, batch, logits, sum_of_sample_mean=torch.sum)

    assert torch.isfinite(loss)
    assert loss == 0
