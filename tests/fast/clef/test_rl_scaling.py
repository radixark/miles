import torch

from examples.clef.rl_objective import group_loss, sample_group
from examples.clef.rl_pilot.scaled import check_case, make_case


def test_control_gradient_matches_direct_brier_kl() -> None:
    scores = torch.tensor([0.2, -0.7, 0.8], requires_grad=True)
    target = [[0.0, 1.0, 0.0]]
    ref = [[0.3, 0.4, 0.3]]
    group = sample_group([scores], target, 32, torch.Generator().manual_seed(3))
    control, _ = group_loss([scores], target, group, ref, policy_weight=0)
    p = scores.softmax(-1)
    expected = (p - torch.tensor(target[0])).square().sum() + 0.1 * (p * (p.log() - torch.tensor(ref[0]).log())).sum()
    actual_grad = torch.autograd.grad(control, scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, scores, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    hybrid, _ = group_loss([scores], target, group, ref)
    assert not torch.allclose(torch.autograd.grad(hybrid, scores)[0], actual_grad)


def test_scaled_labels_and_disjoint_templates() -> None:
    templates = {}
    for split in ("train", "validation"):
        templates[split] = set()
        for index in range(192):
            case = make_case(index, split, 261012)
            check_case(case)
            templates[split].add(case["template_group"])
            assert case["split"] == split
            assert len(case["questions"]) >= 3
    assert not templates["train"] & templates["validation"]
