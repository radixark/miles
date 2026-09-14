import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from examples.on_policy_distillation.qwen3_1_7b_opsd import rm
from examples.on_policy_distillation.qwen3_1_7b_opsd.loss import loss_function
from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss_hub.opd import forward_kl_loss
from miles.utils.types import Sample


@pytest.mark.parametrize("clip", [0.05, 10.0])
def test_clipped_loss_matches_per_entry_reference_and_gradients(monkeypatch, clip):
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(cp=SimpleNamespace(size=1)))
    args = Namespace(
        qkv_format="thd", true_on_policy_mode=False, opd_log_prob_top_k=3, opd_kl_coef=0.3, opsd_kl_clip=clip
    )
    batch = {
        "metadata": [{"opd": {"ids": [[0, 1, 0]], "logprobs": [[math.log(0.7), math.log(0.2), -math.inf]]}}],
        "unconcat_tokens": [torch.tensor([0, 1])],
        "total_lengths": [2],
        "response_lengths": [1],
    }
    logits = torch.tensor([[[-5.0, 1.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    loss, metrics = loss_function(args, batch, logits, torch.sum)
    reference = logits.detach().clone().requires_grad_()
    teacher = torch.tensor([0.7, 0.2])
    terms = teacher * (teacher.log() - reference[0, 0].log_softmax(-1)[:2])
    expected = args.opd_kl_coef * terms.clamp(max=clip).sum()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, reference)[0])
    assert metrics["opd_kl_clipfrac"].item() == pytest.approx((terms > clip).sum().item() / 3)
    assert metrics["opd_teacher_coverage"].item() == pytest.approx(0.9)
    assert all(not value.requires_grad for value in metrics.values())
    core_loss, core_metrics = forward_kl_loss(args, batch, logits, torch.sum)
    torch.testing.assert_close(core_loss, terms.sum())
    assert "opd_kl_clipfrac" not in core_metrics


@pytest.mark.parametrize("clip", [0, -1, math.nan, math.inf])
def test_invalid_example_clip_rejected(clip):
    with pytest.raises(ValueError, match="opsd_kl_clip"):
        loss_function(Namespace(opsd_kl_clip=clip), {}, None, torch.sum)


async def test_privileged_teacher_scores_student_response(monkeypatch):
    args = Namespace(hf_checkpoint="model", chat_template_path=None, opd_log_prob_top_k=2, rm_url="teacher")
    sample = Sample(tokens=[7, 1, 2, 3], response_length=2, metadata={"teacher_prompt": "solution"})
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [8, 9])
    monkeypatch.setattr(rm, "load_tokenizer", lambda *a, **kw: tokenizer)

    async def post(url, payload):
        assert url == "teacher"
        assert payload["input_ids"] == [8, 9, 2, 3]
        assert payload["logprob_start_len"] == 1
        return {
            "meta_info": {
                "input_token_logprobs": [None, [-0.1, 2], [-0.2, 3]],
                "input_top_logprobs": [None, [[-0.1, 2]], [[-0.2, 3]]],
            }
        }

    monkeypatch.setattr(rm, "post", post)
    assert await rm.reward_func(args, sample) == 0
    assert sample.train_metadata["opd"]["ids"] == [[2, 0], [3, 0]]


def test_evaluation_accepts_zero_padded_answers():
    assert rm._is_correct("The answer is \\boxed{25}.", "025") == 1
