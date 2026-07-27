"""Unit tests for the backend-neutral HF weight iterator base.

Covers WeightUpdatePlacement / resolve_placement and the iter_hf_weights
template method (adapter units in the bucketed stream), which every backend
shares.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)

SAMPLE_LORA_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
]

SAMPLE_BASE_ONLY_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
]


class TestWeightUpdatePlacement:
    def test_resolve_without_forced_returns_required(self):
        required = WeightUpdatePlacement(gather_pp=False)
        assert resolve_placement(required, None) == required

    def test_resolve_joins_gathered_dims(self):
        keep_pp = WeightUpdatePlacement(gather_pp=False)
        full = WeightUpdatePlacement(gather_pp=True)
        assert resolve_placement(keep_pp, full) == full
        assert resolve_placement(full, keep_pp) == full


class _StubIterator(HfWeightIteratorBase):
    """Concrete subclass with canned base and adapter unit streams."""

    def __init__(self, exported, base=()):
        super().__init__(
            Namespace(update_weight_buffer_size=1 << 30),
            [],
            placement=WeightUpdatePlacement(gather_pp=True),
            model_name="stub",
            quantization_config=None,
        )
        self._exported = exported
        self._base = base
        self.export_calls = []

    def _iter_hf_param_units(self, weights, *, materialize):
        for pair in self._base:
            yield [pair]

    def _iter_hf_adapter_units(self, lora_name, adapter, *, weights, materialize):
        self.export_calls.append(adapter)
        for name, tensor in self._exported:
            yield [(f"{lora_name}:{name}", tensor)]


class TestIterHfWeightsTemplate:
    @staticmethod
    def _names(buckets):
        return [name for bucket in buckets for name, _ in bucket]

    def test_adapter_units_join_the_stream_prefixed(self):
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS, base=SAMPLE_BASE_ONLY_WEIGHTS)
        names = self._names(iterator.iter_hf_weights(None, adapters=[("miles_lora", None)]))
        assert names == [SAMPLE_BASE_ONLY_WEIGHTS[0][0]] + [f"miles_lora:{n}" for n, _ in SAMPLE_LORA_WEIGHTS]

    def test_adapter_argument_reaches_the_hook(self):
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS)
        adapter = SimpleNamespace(slot=3)
        list(iterator.iter_hf_weights(None, adapters=[("__miles_slot_3", adapter)]))
        assert iterator.export_calls == [adapter]

    def test_include_base_false_streams_adapters_only(self):
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS, base=SAMPLE_BASE_ONLY_WEIGHTS)
        names = self._names(iterator.iter_hf_weights(None, include_base=False, adapters=[("miles_lora", None)]))
        assert names == [f"miles_lora:{n}" for n, _ in SAMPLE_LORA_WEIGHTS]

    def test_no_adapters_matches_base_stream(self):
        iterator = _StubIterator([], base=SAMPLE_BASE_ONLY_WEIGHTS)
        names = self._names(iterator.iter_hf_weights(None))
        assert names == [SAMPLE_BASE_ONLY_WEIGHTS[0][0]]
        assert iterator.export_calls == []
