from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_config

from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct as direct_module
from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


@pytest.fixture
def expert_owner_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[int], direct_module.HfWeightIteratorDirect]:
    def build(edp_rank: int) -> direct_module.HfWeightIteratorDirect:
        trivial = SimpleNamespace(size=1, rank=0, group=None)
        combined = SimpleNamespace(size=2, rank=edp_rank, group=object())
        parallel = SimpleNamespace(tp=trivial, etp=trivial, ep=trivial, pp=trivial, edp=combined, tp_dp_cp=combined)
        model = torch.nn.Linear(1, 1, bias=False)
        model.config = SimpleNamespace(mtp_num_layers=0)
        weights = [(f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}", torch.ones(1)) for i in range(3)]
        args = make_trainer_config(
            num_experts=3,
            update_weight_buffer_size=1024,
            sglang=SglangConfig(models=[], base_args={"speculative_algorithm": None}),
        )
        monkeypatch.setattr(direct_module, "get_parallel_state", lambda: parallel)
        monkeypatch.setattr(direct_module, "named_params_and_buffers", lambda _args, _model: iter(weights))
        monkeypatch.setattr(direct_module.dist, "get_rank", lambda: edp_rank)
        monkeypatch.setattr(direct_module.dist, "get_process_group_ranks", lambda _group: [0, 1])

        return direct_module.HfWeightIteratorDirect(
            args=args,
            model=[model],
            placement=WeightUpdatePlacement(gather_pp=False),
            model_name="qwen3_moe",
            quantization_config=None,
        )

    return build
