from argparse import Namespace
from pathlib import Path

import torch

from miles.backends.training_utils.metrics.debug_dump import maybe_dump_policy_loss_debug


class TestDebugDumpConfiguration:
    def test_disabled_dump_returns_before_reading_payload_or_parallel_state(self) -> None:
        """A disabled debug dump accepts empty payloads without distributed setup."""
        maybe_dump_policy_loss_debug(
            args=Namespace(dump_details=None),
            batch={},
            train_log_probs=[],
            old_log_probs=[],
            rollout_log_probs=None,
            advantages=[],
            local_loss_masks=[],
            ppo_kl=torch.tensor(0.0),
            pg_loss=torch.tensor(0.0),
        )

    def test_enabled_dump_preserves_detached_samples_under_the_configured_root(
        self, tmp_path: Path, single_process_parallel_state: None
    ) -> None:
        """An enabled dump writes detached sample diagnostics under its configured root."""
        train = torch.tensor([-0.25, -0.5], requires_grad=True)

        maybe_dump_policy_loss_debug(
            args=Namespace(dump_details=tmp_path),
            batch={"total_lengths": [3], "response_lengths": [2]},
            train_log_probs=[train],
            old_log_probs=[torch.tensor([-0.5, -0.75])],
            rollout_log_probs=[torch.tensor([-0.375, -0.75])],
            advantages=[torch.tensor([1.0, -1.0])],
            local_loss_masks=[torch.tensor([1, 0])],
            ppo_kl=torch.tensor([0.0, float("nan")]),
            pg_loss=torch.tensor([1.0, 2.0], requires_grad=True),
        )

        [path] = (tmp_path / "policy_loss_debug").glob("rank_0_call_*.pt")
        payload = torch.load(path, weights_only=False)
        [sample] = payload["samples"]
        assert sample["total_length"] == 3
        assert sample["response_length"] == 2
        assert torch.equal(sample["train_log_probs"], train.detach())
        assert not sample["train_log_probs"].requires_grad
        assert torch.equal(sample["train_rollout_abs_diff"], torch.tensor([0.125, 0.25]))
        assert payload["finite"]["ppo_kl"] is False
        assert payload["finite"]["pg_loss"] is True
