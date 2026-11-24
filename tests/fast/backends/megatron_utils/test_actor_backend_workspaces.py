import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_actor_reload import actor_module

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace

__all__ = ["actor_module"]


class TestBackendWorkspaceLifetime:
    @pytest.mark.parametrize(
        "enabled,transformer_impl,graph_impl,cleared",
        [
            (True, "transformer_engine", "none", True),
            (False, "transformer_engine", "none", False),
            (True, "local", "none", False),
            (True, "transformer_engine", "local", False),
        ],
    )
    def test_only_uncaptured_enabled_transformer_engine_workspaces_are_released(
        self,
        actor_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        enabled: bool,
        transformer_impl: str,
        graph_impl: str,
        cleared: bool,
    ) -> None:
        """Offload preserves captured workspace addresses and releases only eligible Transformer Engine caches."""
        monkeypatch.setitem(
            sys.modules,
            "transformer_engine.pytorch.module.base",
            SimpleNamespace(TransformerEngineBaseModule=torch.nn.Linear),
        )
        te = torch.nn.Linear(1, 1)
        unrelated = torch.nn.ReLU()
        cached = object()
        te._fp8_workspaces = {"cached": cached}
        unrelated._fp8_workspaces = {"cached": cached}
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = SimpleNamespace(
            clear_quantized_weight_workspaces_on_offload=enabled,
            backend=MegatronArgsNamespace(transformer_impl=transformer_impl, cuda_graph_impl=graph_impl),
        )
        actor.model = [torch.nn.Sequential(te, unrelated)]

        actor._clear_quantized_weight_workspaces()

        assert te._fp8_workspaces == ({} if cleared else {"cached": cached})
        assert unrelated._fp8_workspaces == {"cached": cached}
