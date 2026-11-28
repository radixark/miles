"""Check experiment alignment and the custom GSPO gradient independently of a server."""

import math
import shlex

import pytest
import torch
from examples.multi_lora.custom_gspo import make_gspo_loss
from scripts.run_qwen3_5_4b_pissa import ScriptArgs, _algorithm_args, _train_args


def _flags(command):
    result = {}
    current = None
    for token in shlex.split(command):
        if token.startswith("--"):
            current = token
            assert current not in result, current
            result[current] = []
        else:
            result[current].append(token)
    return result


def test_lora_and_pissa_share_the_experiment_and_original_reference():
    args = ScriptArgs()
    lora, pissa = (_flags(_train_args(args, arm)) for arm in ("lora", "pissa"))
    differing = {key for key in lora.keys() | pissa.keys() if lora.get(key) != pissa.get(key)}
    assert differing == {
        "--hf-checkpoint",
        "--load",
        "--lora-adapter-path",
        "--save",
        "--dump-details",
        "--tb-experiment-name",
    }
    assert pissa["--ref-load"] == lora["--load"] == ["/root/models/Qwen3.5-4B_torch_dist"]
    assert pissa["--hf-checkpoint"] == ["/root/models/Qwen3.5-4B-pissa-r32/residual_base"]
    assert pissa["--lora-adapter-path"] == ["/root/models/Qwen3.5-4B-pissa-r32/adapter"]
    assert pissa["--lora-type"] == ["canonical_lora"]
    assert "--no-load-optim" in pissa and "--no-load-rng" in pissa
    assert pissa["--num-gpus-per-node"] == pissa["--actor-num-gpus-per-node"] == ["8"]


@pytest.mark.parametrize("objective", ["grpo", "tis", "gspo", "score_centering", "score_centering_tis"])
def test_objectives_use_their_existing_native_paths(objective):
    flags = _flags(_algorithm_args(objective))
    assert flags["--advantage-estimator"] == ["gspo" if objective == "gspo" else "grpo"]
    if objective.startswith("score_centering"):
        assert "--use-tis" not in flags
        assert flags["--loss-type"] == ["score_centering"]
        assert flags["--score-centering-is"] == ["tis" if objective.endswith("_tis") else "none"]
        assert "--disable-grpo-std-normalization" in flags
        assert "--calculate-per-token-loss" in flags
    else:
        assert ("--use-tis" in flags) == (objective == "tis")


def test_gspo_uses_sequence_ratios_and_masks_with_the_correct_gradient():
    lp = torch.tensor([float("nan"), math.log(0.525), math.log(0.525)], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([float("nan"), math.log(0.5), math.log(0.5)], dtype=torch.float64, requires_grad=True)
    callback = make_gspo_loss([old], [torch.tensor([0, 1, 1])], torch.tensor([2.0]), clip_low=0.1, clip_high=0.1)
    loss, metrics = callback([object()], [lp])
    assert loss.item() == pytest.approx(-2.1)
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.tensor([0.0, -1.05, -1.05], dtype=torch.float64))
    assert old.grad is None
    assert metrics["sequence_ratio_outside_clip_fraction"] == 0


def test_negative_advantage_clipping_has_no_gradient():
    lp = torch.tensor([math.log(0.4)], requires_grad=True)
    callback = make_gspo_loss(
        [torch.tensor([math.log(0.5)])], [torch.ones(1)], torch.tensor([-1.0]), clip_low=0.1, clip_high=0.1
    )
    loss, metrics = callback([object()], [lp])
    assert loss.item() == pytest.approx(0.9)
    loss.backward()
    assert lp.grad.item() == 0
    assert metrics["sequence_ratio_outside_clip_fraction"] == 1


def test_gspo_weights_sequences_equally_despite_different_response_lengths():
    short = torch.zeros(1, requires_grad=True)
    long = torch.zeros(3, requires_grad=True)
    callback = make_gspo_loss([short, long], [torch.ones(1), torch.ones(3)], torch.ones(2))
    loss, _ = callback([object(), object()], [short, long])
    loss.backward()
    torch.testing.assert_close(short.grad, torch.tensor([-0.5]))
    torch.testing.assert_close(long.grad, torch.full((3,), -1 / 6))


def test_no_response_tokens_fail_before_submitting_a_custom_loss():
    with pytest.raises(ValueError, match="response mask"):
        make_gspo_loss([torch.zeros(2)], [torch.zeros(2)], torch.ones(1))
