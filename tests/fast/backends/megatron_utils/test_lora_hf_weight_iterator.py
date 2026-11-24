"""Unit tests for the megatron hf-weight-iterator factory: mode routing and
placement resolution against each implementation's forced placement."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["lora", "weight-update"])


from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


class TestHfWeightIteratorFactory:
    def _make_args(self, mode="bridge"):
        return Namespace(
            megatron_to_hf_mode=mode,
            hf_checkpoint="/fake/path",
            update_weight_buffer_size=1,
        )

    def _create(self, mode, required_placement=None):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator

        if required_placement is None:
            required_placement = WeightUpdatePlacement(gather_pp=True)
        return get_hf_weight_iterator(
            self._make_args(mode),
            [MagicMock()],
            required_placement=required_placement,
            model_name="qwen",
            quantization_config=None,
        )

    def test_bridge_mode_creates_bridge_iterator(self):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge

        with patch.object(HfWeightIteratorBridge, "__init__", return_value=None):
            iterator = self._create("bridge")
            assert isinstance(iterator, HfWeightIteratorBridge)

    def test_raw_mode_creates_direct_iterator(self):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        with patch.object(HfWeightIteratorDirect, "__init__", return_value=None):
            iterator = self._create("raw")
            assert isinstance(iterator, HfWeightIteratorDirect)

    def test_dsv4_bridge_uses_canonical_atomic_groups(self):
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge

        iterator = object.__new__(HfWeightIteratorBridge)
        iterator.model_name = "deepseekv4"
        iterator.args = Namespace(backend=Namespace(q_lora_rank=None))

        groups = {group.key: group.suffixes for group in iterator._hf_atomic_update_groups()}
        assert groups["wqkv_a"] == (".self_attn.wq_a.weight", ".self_attn.wkv.weight")
        assert groups["compressor_wkv_gate"] == (
            ".self_attn.compressor.wkv.weight",
            ".self_attn.compressor.wgate.weight",
        )

    def test_invalid_mode_raises(self):
        with pytest.raises(KeyError):
            self._create("invalid_mode")

    def test_forced_placement_resolves_by_implementation(self):
        """Bridge forces a full gather; the direct iterator can keep PP local,
        so resolution joins the requirement with each implementation's floor."""
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
        from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        full = WeightUpdatePlacement(gather_pp=True)
        keep_pp = WeightUpdatePlacement(gather_pp=False)
        assert HfWeightIteratorBridge.forced_placement == full
        assert HfWeightIteratorDirect.forced_placement == keep_pp

        captured = {}

        def _capture_init(self, args, model, *, placement, model_name, quantization_config):
            captured["placement"] = placement

        with patch.object(HfWeightIteratorBridge, "__init__", _capture_init):
            self._create("bridge", required_placement=keep_pp)
        assert captured["placement"] == full

        with patch.object(HfWeightIteratorDirect, "__init__", _capture_init):
            self._create("raw", required_placement=keep_pp)
        assert captured["placement"] == keep_pp


class TestPpLocalAdapterUnits:
    def _iter_stage(self, stage_tensors, peer_counts):
        from miles.backends.megatron_utils.update_weight import hf_weight_iterator as module

        class _Stage(module.MegatronHfWeightIteratorBase):
            def _iter_hf_param_units(self, weights, *, materialize):
                yield from ()

            def _export_pp_local_lora(self, adapter):
                return list(stage_tensors)

        iterator = object.__new__(_Stage)
        iterator.placement = WeightUpdatePlacement(gather_pp=False)

        def _all_reduce(counts_tensor, group):
            counts_tensor += torch.tensor(peer_counts)

        pp = Namespace(size=2, group=None)
        with (
            patch.object(module, "get_parallel_state", return_value=Namespace(pp=pp)),
            patch.object(module.dist, "all_reduce", _all_reduce),
            patch.object(module.torch.cuda, "current_device", return_value="cpu"),
        ):
            return list(iterator._iter_hf_adapter_units(None, materialize=True))

    def test_a_stage_without_adapted_layers_sends_nothing(self):
        assert self._iter_stage([], peer_counts=[2, 2]) == []

    def test_an_adapter_empty_on_every_stage_raises(self):
        with pytest.raises(RuntimeError, match="zero tensors"):
            self._iter_stage([], peer_counts=[0, 0])
