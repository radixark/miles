from argparse import Namespace

import pytest

from miles.backends.training_utils.ci_utils import check_kl


class TestCheckKl:
    def test_namespaced_policy_metrics_still_trigger_the_kl_checker(self) -> None:
        """A policy namespace must not hide an out-of-tolerance PPO KL value."""
        args = Namespace(
            multi_latent_attention=False,
            trainer_model_id="alpha",
            use_rollout_routing_replay=False,
        )

        with pytest.raises(AssertionError):
            check_kl(
                args=Namespace(**{**vars(args), "trainer_model_id": None}),
                log_dict={"train/ppo_kl": 0.1, "train/pg_clipfrac": 0.2},
                step_id=0,
                accumulated_step_id=1,
            )
        with pytest.raises(AssertionError):
            check_kl(
                args=args,
                log_dict={"alpha/train/ppo_kl": 0.1, "alpha/train/pg_clipfrac": 0.2},
                step_id=0,
                accumulated_step_id=1,
            )
