import math
from types import SimpleNamespace

import torch

from examples.infra_features.train_infer_mismatch_helper.mis import MisConfig, calculate_veto_mask, compute_mis_weights


class TestOptionalVetoThreshold:
    def test_null_threshold_survives_transport_and_disables_veto(self) -> None:
        """An explicitly disabled rejection veto remains disabled after typed configuration transport."""
        config = MisConfig.model_validate_json('{"rs_veto_threshold": null}')
        restored = MisConfig.model_validate_json(config.model_dump_json())
        ratios = torch.tensor([-100.0, 0.0, 100.0])
        mask = torch.tensor([1.0, 1.0, 0.0])

        result = calculate_veto_mask(ratios, mask, restored.rs_veto_threshold, {})

        assert restored.rs_veto_threshold is None
        torch.testing.assert_close(result, torch.ones_like(ratios))


class TestTisBatchNormalize:
    def test_geometric_level_normalizes_over_sequences(self) -> None:
        """Geometric weights are constant per sequence, so the batch mean is taken over sequences."""
        config = MisConfig(tis_level="geometric", tis_upper_bound=100.0, tis_batch_normalize=True)
        args = SimpleNamespace(**config.model_dump(), use_tis=True)
        rollout_log_probs = [torch.zeros(2), torch.zeros(4)]
        train_log_probs = [torch.tensor([0.2, 0.4]), torch.tensor([-0.1, -0.1, -0.1, 0.0])]
        loss_masks = [torch.ones(2), torch.tensor([1.0, 1.0, 1.0, 0.0])]

        weights, _, _ = compute_mis_weights(
            args,
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
        )

        sequence_weights = [math.exp(0.3), math.exp(-0.1)]
        batch_mean = sum(sequence_weights) / len(sequence_weights)
        torch.testing.assert_close(weights[0], torch.full((2,), sequence_weights[0] / batch_mean))
        torch.testing.assert_close(weights[1], torch.tensor([sequence_weights[1] / batch_mean] * 3 + [0.0]))
