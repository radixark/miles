"""The weight-update selector follows the trainer model, not the conversion mode."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import MegatronHfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


class _Iterator(MegatronHfWeightIteratorBase):
    def _iter_hf_param_units(self, weights, *, materialize):
        return iter(())

    def _export_pp_local_lora(self, adapter, weights):
        return []


def _make(speculative, mtp_num_layers):
    model = [SimpleNamespace(config=SimpleNamespace(mtp_num_layers=mtp_num_layers))]
    args = Namespace(sglang_speculative_algorithm=speculative, q_lora_rank=None)
    return _Iterator(
        args, model, placement=WeightUpdatePlacement(gather_pp=True), model_name="qwen", quantization_config=None
    )


@pytest.mark.parametrize(
    "speculative, mtp_num_layers, selector",
    [
        (None, None, "all"),
        (None, 1, "all"),
        ("EAGLE", 1, "all"),
        ("EAGLE", None, "target"),
        ("EAGLE", 0, "target"),
    ],
    ids=["no_draft", "no_draft_with_mtp", "trained_draft", "frozen_draft", "frozen_draft_zero_layers"],
)
def test_frozen_draft_is_excluded_from_the_sync(speculative, mtp_num_layers, selector):
    """A draft the trainer never feeds must not be re-postprocessed on every sync."""
    assert _make(speculative, mtp_num_layers).weight_update_selector == selector
