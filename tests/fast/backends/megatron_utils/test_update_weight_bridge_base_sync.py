"""Base weight sync on the disaggregated path must follow --megatron-to-hf-mode.

In bridge mode the model is built by AutoBridge, so its parameter names are the
bridge's; the hand-written converters in megatron_to_hf/ only know the names
``--spec`` would have produced. Converting bridge-built weights with them raises
"Unknown parameter name" on architectures where the two disagree (Qwen3.5 GDN,
Qwen3-VL), so bridge mode has to convert base weights with the bridge too.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import (
    DistBucketedWeightUpdateMixin,
)

_MIXIN_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin"


class _Harness(DistBucketedWeightUpdateMixin):
    """Minimal stand-in for the transports, which build themselves differently."""

    def __init__(self, megatron_to_hf_mode, *, is_source=True, chunks=None):
        self.args = Namespace(
            megatron_to_hf_mode=megatron_to_hf_mode,
            update_weight_buffer_size=1 << 30,
            pipeline_model_parallel_size=1,
        )
        self.model = [MagicMock()]
        self.model_name = "qwen3_5"
        self.quantization_config = None
        self.weight_version = 0
        self.is_lora = False
        self.multi_lora_adapters = None
        self._group_name = "test"
        self.__is_source = is_source
        self.transmitted: list[list[tuple[str, torch.Tensor]]] = []

        if chunks is not None:
            iterator = MagicMock()
            iterator.get_hf_weight_chunks.return_value = iter(chunks)
            self._hf_weight_iterator = iterator

    @property
    def _is_source(self):
        return self.__is_source

    def _update_weight_implementation(self, named_tensors, pbar=None):
        self.transmitted.append(named_tensors)


_CHUNKS = [
    [("model.layers.0.linear_attn.in_proj_qkv.weight", torch.zeros(2, 2))],
    [("model.layers.0.linear_attn.out_proj.weight", torch.zeros(2, 2))],
]


class TestConversionRouting:
    def test_bridge_mode_converts_base_via_bridge(self):
        assert _Harness("bridge")._convert_base_via_bridge is True

    def test_raw_mode_keeps_hand_written_converters(self):
        assert _Harness("raw")._convert_base_via_bridge is False

    def test_bridge_mode_skips_the_raw_gather_paths(self):
        """The bug: bridge-built weights were fed to the megatron_to_hf/ converters."""
        harness = _Harness("bridge", chunks=list(_CHUNKS))

        with (
            patch.object(harness, "_gather_and_update_non_expert_weights") as gather_non_expert,
            patch.object(harness, "_gather_and_update_expert_weights") as gather_expert,
            patch.object(harness, "_pause_and_prepare_engines"),
            patch.object(harness, "_finalize_and_resume_engines"),
            patch(f"{_MIXIN_MODULE}.dist.barrier"),
            patch(f"{_MIXIN_MODULE}.get_gloo_group"),
        ):
            harness.update_weights()

        gather_non_expert.assert_not_called()
        gather_expert.assert_not_called()
        assert harness.transmitted == _CHUNKS

    def test_raw_mode_still_uses_the_gather_paths(self):
        harness = _Harness("raw")

        with (
            patch.object(harness, "_gather_and_update_non_expert_weights") as gather_non_expert,
            patch.object(harness, "_gather_and_update_expert_weights") as gather_expert,
            patch.object(harness, "_update_base_weights_via_bridge") as via_bridge,
            patch.object(harness, "_pause_and_prepare_engines"),
            patch.object(harness, "_finalize_and_resume_engines"),
            patch(f"{_MIXIN_MODULE}.dist.barrier"),
            patch(f"{_MIXIN_MODULE}.get_gloo_group"),
        ):
            harness.update_weights()

        via_bridge.assert_not_called()
        gather_non_expert.assert_called_once()
        gather_expert.assert_called_once()


class TestBridgeBaseSync:
    def test_non_source_ranks_iterate_but_do_not_transmit(self):
        """The bridge runs collectives internally, so every rank has to iterate it."""
        harness = _Harness("bridge", is_source=False, chunks=list(_CHUNKS))

        harness._update_base_weights_via_bridge(harness._update_weight_implementation)

        harness._hf_weight_iterator.get_hf_weight_chunks.assert_called_once_with({}, weight_type="base")
        assert harness.transmitted == []

    def test_empty_export_is_an_error_not_a_silent_no_op(self):
        harness = _Harness("bridge", chunks=[])

        with pytest.raises(RuntimeError, match="zero chunks"):
            harness._update_base_weights_via_bridge(harness._update_weight_implementation)

    def test_base_iterator_is_created_once_and_reused(self):
        harness = _Harness("bridge")
        sentinel = object()

        with patch(f"{_MIXIN_MODULE}.HfWeightIteratorBase.create", return_value=sentinel) as create:
            assert harness._base_hf_weight_iterator() is sentinel
            assert harness._base_hf_weight_iterator() is sentinel

        create.assert_called_once()
        assert create.call_args.kwargs["is_lora"] is False

    def test_an_existing_lora_iterator_is_not_replaced(self):
        harness = _Harness("bridge", chunks=list(_CHUNKS))
        existing = harness._hf_weight_iterator

        with patch(f"{_MIXIN_MODULE}.HfWeightIteratorBase.create") as create:
            assert harness._base_hf_weight_iterator() is existing

        create.assert_not_called()
