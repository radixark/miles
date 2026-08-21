import torch

from miles.backends.training_utils.loss_hub.advantages import compute_advantages

from .loss_test_utils import make_args, make_parallel_state


def test_ppo_advantages_preserve_raw_kl_metric():
    make_parallel_state()
    raw_kl = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
    kl = [raw_kl.clone()]

    compute_advantages(
        args=make_args(advantage_estimator="ppo", kl_coef=0.05),
        kl=kl,
        rewards=[1.5],
        log_probs=[torch.zeros_like(raw_kl)],
        loss_masks=[torch.ones_like(raw_kl)],
        total_lengths=[5],
        response_lengths=[3],
        values=[torch.zeros_like(raw_kl)],
    )

    assert torch.equal(kl[0], raw_kl)
