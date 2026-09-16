"""Coverage plan for running tinker-cookbook's rl/train.py loop unchanged against the gateway; skeletons until the fixes land."""

import pytest

SKELETON = pytest.mark.skip(reason="skeleton: lands with the cookbook-compatibility work package")


@SKELETON
def test_ttl_seconds_is_accepted_and_ignored():
    """save_weights_for_sampler / save_state with ttl_seconds (the cookbook's 7-day default) decode without a validation_error."""


@SKELETON
def test_create_model_accepts_cookbook_defaults():
    """create_lora_training_client(model_name, rank, user_metadata) with seed=None and train_*=True is accepted when the server layout trains attn, mlp and unembed."""


@SKELETON
def test_weights_info_carries_what_resume_needs():
    """weights_info returns base_model, is_lora, lora_rank and train_* so create_training_client_from_state_with_optimizer can rebuild the client."""
