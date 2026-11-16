from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["replay"])

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

# Six layers routing in 1, 3 and 4, as the Nemotron-H bridge sets moe_layer_freq.
_MOE_LAYER_FREQ = [0, 1, 0, 1, 1, 0]


class _Replay:
    def __init__(self):
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


@pytest.fixture
def register_replay_list_moe(monkeypatch):
    """The function under test on the last of two pipeline stages, which builds layers 3 to 5."""
    transformer_block = ModuleType("megatron.core.transformer.transformer_block")
    transformer_block.get_num_layers_to_build = lambda config, vp_stage: 3
    transformer_layer = ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer.get_transformer_layer_offset = lambda config, vp_stage: 3
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_block", transformer_block)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_layer", transformer_layer)
    monkeypatch.delitem(sys.modules, "miles.backends.megatron_utils.replay_utils", raising=False)
    return importlib.import_module("miles.backends.megatron_utils.replay_utils").register_replay_list_moe


def _models():
    config = SimpleNamespace(num_layers=len(_MOE_LAYER_FREQ), moe_layer_freq=_MOE_LAYER_FREQ)
    return [SimpleNamespace(module=SimpleNamespace(config=config))]


@pytest.mark.parametrize(
    ("num_columns", "expected_columns"),
    [
        # A column per model layer: each MoE layer reads its own layer index.
        (6, [3, 4]),
        # A column per MoE layer: layers 1, 3, 4 are columns 0, 1, 2, so the
        # earlier stage's layer 1 shifts this stage's layers to columns 1, 2.
        (3, [1, 2]),
    ],
)
def test_last_stage_reads_the_columns_of_its_moe_layers(register_replay_list_moe, num_columns, expected_columns):
    replay_data = torch.arange(2 * num_columns * 2).reshape(2, num_columns, 2)
    replays = [_Replay(), _Replay()]

    register_replay_list_moe(replays, replay_data, models=_models())

    for replay, column in zip(replays, expected_columns, strict=True):
        torch.testing.assert_close(replay.recorded[0], replay_data[:, column])


def test_rejects_a_layer_count_matching_neither_layout(register_replay_list_moe):
    with pytest.raises(AssertionError, match="MoE layers only"):
        register_replay_list_moe([_Replay(), _Replay()], torch.zeros(2, 5, 2), models=_models())
