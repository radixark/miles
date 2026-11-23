from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import named_weights
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.training_utils import parallel
from miles.backends.training_utils.parallel import GroupInfo


class TestBackendExpertNaming:
    @pytest.mark.parametrize("expert_rank,global_expert", [(0, 0), (1, 2)])
    def test_global_weight_names_use_backend_expert_count_and_local_replica_rank(
        self, monkeypatch: pytest.MonkeyPatch, expert_rank: int, global_expert: int
    ) -> None:
        """Global weight names combine the backend expert count with the current replica's expert rank."""
        parameter = torch.nn.Parameter(torch.tensor([7.0]))
        model = torch.nn.ModuleDict(
            {
                "module": torch.nn.ModuleDict(
                    {
                        "decoder": torch.nn.ModuleDict(
                            {
                                "layers": torch.nn.ModuleList(
                                    [
                                        torch.nn.ModuleDict(
                                            {
                                                "mlp": torch.nn.ModuleDict(
                                                    {
                                                        "experts": torch.nn.ModuleDict(
                                                            {
                                                                "linear_fc1": torch.nn.ParameterDict(
                                                                    {"weight0": parameter}
                                                                )
                                                            }
                                                        )
                                                    }
                                                )
                                            }
                                        )
                                    ]
                                )
                            }
                        )
                    }
                )
            }
        )
        model.config = object()
        args = SimpleNamespace(backend=MegatronArgsNamespace(num_experts=4))
        monkeypatch.setattr(
            parallel, "_parallel_state", SimpleNamespace(ep=GroupInfo(rank=expert_rank, size=2, group=None))
        )
        monkeypatch.setattr(named_weights, "get_transformer_layer_offset", lambda config: 3)

        result = dict(named_weights.named_params_and_buffers(args=args, model=[model]))

        assert list(result) == [f"module.module.decoder.layers.3.mlp.experts.linear_fc1.weight{global_expert}"]
        assert next(iter(result.values())) is parameter
