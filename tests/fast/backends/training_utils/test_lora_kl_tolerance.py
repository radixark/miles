from argparse import Namespace

import pytest

from miles.backends.training_utils.metrics.checks import check_kl


class TestLoraKlTolerance:
    @pytest.mark.parametrize("ppo_kl", [-5e-9, 5e-9])
    def test_lora_allows_conversion_noise_that_full_weights_reject(self, ppo_kl: float) -> None:
        """LoRA conversion noise receives the relaxed KL threshold only for LoRA."""
        args = Namespace(
            backend=Namespace(multi_latent_attention=False),
            trainer_model_id=None,
            lora_rank=8,
            use_rollout_routing_replay=False,
        )
        metrics = {"train/ppo_kl": ppo_kl, "train/pg_clipfrac": 0.0}

        check_kl(args=args, log_dict=metrics, step_id=0, accumulated_step_id=1)

        args.lora_rank = 0
        with pytest.raises(AssertionError):
            check_kl(args=args, log_dict=metrics, step_id=0, accumulated_step_id=1)

    @pytest.mark.parametrize(
        "metrics",
        [
            {"train/ppo_kl": -1e-8, "train/pg_clipfrac": 0.0},
            {"train/ppo_kl": 1e-8, "train/pg_clipfrac": 0.0},
            {"train/ppo_kl": 0.0, "train/pg_clipfrac": 1e-10},
        ],
    )
    def test_lora_still_rejects_kl_or_clipping_at_the_limit(self, metrics: dict[str, float]) -> None:
        """The relaxed LoRA threshold preserves strict KL and clipping limits."""
        args = Namespace(
            backend=Namespace(multi_latent_attention=False),
            trainer_model_id=None,
            lora_rank=8,
            use_rollout_routing_replay=False,
        )

        with pytest.raises(AssertionError):
            check_kl(args=args, log_dict=metrics, step_id=0, accumulated_step_id=1)
