import torch

from examples.infra_features.train_infer_mismatch_helper.mis import MisConfig, calculate_veto_mask


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
