from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.rollout import sft_rollout
from miles.utils.types import Sample


class _FakeMaskGenerator:
    def __init__(self, loss_mask: list[int]):
        self.loss_mask = loss_mask

    def get_loss_mask(self, messages, tools=None):
        return list(range(len(self.loss_mask))), list(self.loss_mask)

    def get_response_lengths(self, loss_masks):
        return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class _FakeDataBuffer:
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def get_samples(self, n):
        return [[sample] for sample in self.samples]


@pytest.mark.parametrize(
    "loss_mask,response_length,response_mask",
    [
        ([0, 0, 1, 1, 0, 1], 4, [1, 1, 0, 1]),
        # Fully masked sample: one zero-weight response token, never the whole mask.
        ([0, 0, 0, 0], 1, [0]),
    ],
    ids=["trained", "fully_masked"],
)
def test_response_mask_matches_response_length(monkeypatch, loss_mask, response_length, response_mask):
    monkeypatch.setattr(sft_rollout, "TOKENIZER", object())
    monkeypatch.setattr(sft_rollout, "PROCESSOR", object())
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", _FakeMaskGenerator(loss_mask))
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", True)
    args = Namespace(rollout_global_dataset=True, rollout_batch_size=1)
    sample = Sample(prompt=[{"role": "user", "content": "Q"}], metadata={})

    (group,) = sft_rollout.generate_rollout(args, 0, _FakeDataBuffer([sample]))

    (out,) = group
    assert out.response_length == response_length
    assert out.loss_mask == response_mask
    out.validate()
