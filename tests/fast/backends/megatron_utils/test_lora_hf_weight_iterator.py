"""Unit tests for HfWeightIteratorBase factory routing with LoRA flag.

Validates that the is_lora flag is correctly propagated through the factory
and that the right iterator subclass is selected based on megatron_to_hf_mode.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase


class _ConcreteIterator(HfWeightIteratorBase):
    def get_hf_weight_chunks(self, megatron_local_weights):
        return []


_BASE_MODULE = "miles.backends.megatron_utils.update_weight.hf_weight_iterator_base"


class TestHfWeightIteratorFactory:
    def _make_args(self, mode="bridge"):
        return Namespace(
            megatron_to_hf_mode=mode,
            hf_checkpoint="/fake/path",
            update_weight_buffer_size=1,
        )

    @patch(f"{_BASE_MODULE}.HfWeightIteratorBase.__init__", return_value=None)
    def test_bridge_mode_creates_bridge_iterator(self, mock_init):
        """Factory should select HfWeightIteratorBridge for 'bridge' mode."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import (
            HfWeightIteratorBridge,
        )

        with patch.object(HfWeightIteratorBridge, "__init__", return_value=None) as mock_bridge_init:
            args = self._make_args("bridge")
            iterator = HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=True, model_name="qwen", quantization_config=None
            )
            assert isinstance(iterator, HfWeightIteratorBridge)

    @patch(f"{_BASE_MODULE}.HfWeightIteratorBase.__init__", return_value=None)
    def test_raw_mode_creates_direct_iterator(self, mock_init):
        """Factory should select HfWeightIteratorDirect for 'raw' mode."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
            HfWeightIteratorDirect,
        )

        with patch.object(HfWeightIteratorDirect, "__init__", return_value=None) as mock_direct_init:
            args = self._make_args("raw")
            iterator = HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=False, model_name="qwen", quantization_config=None
            )
            assert isinstance(iterator, HfWeightIteratorDirect)

    def test_invalid_mode_raises(self):
        args = self._make_args("invalid_mode")
        with pytest.raises(KeyError):
            HfWeightIteratorBase.create(
                args=args, model=[MagicMock()], is_lora=False, model_name="qwen", quantization_config=None
            )

    def test_is_lora_stored_on_instance(self):
        """Verify is_lora attribute is stored on the base class."""
        instance = _ConcreteIterator(
            args=self._make_args(),
            model=[MagicMock()],
            model_name="qwen",
            quantization_config=None,
            is_lora=True,
        )
        assert instance.is_lora is True

    def test_is_lora_default_false(self):
        instance = _ConcreteIterator(
            args=self._make_args(),
            model=[MagicMock()],
            model_name="qwen",
            quantization_config=None,
        )
        assert instance.is_lora is False
