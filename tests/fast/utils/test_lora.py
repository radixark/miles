"""Unit tests for the miles.utils.lora rollout gate."""

from argparse import Namespace

from miles.utils.lora import lora_rollout_enabled


class TestLoraRolloutEnabled:
    def test_enabled_when_lora_on_and_not_train_only(self):
        assert lora_rollout_enabled(Namespace(lora_rank=16, debug_lora_train_only=False))

    def test_disabled_under_train_only(self):
        assert not lora_rollout_enabled(Namespace(lora_rank=16, debug_lora_train_only=True))

    def test_disabled_without_lora(self):
        assert not lora_rollout_enabled(Namespace(lora_rank=0, debug_lora_train_only=False))

    def test_missing_train_only_attr_defaults_to_enabled(self):
        assert lora_rollout_enabled(Namespace(lora_rank=8))
