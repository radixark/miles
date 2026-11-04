from types import ModuleType, SimpleNamespace

import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config


class TestP2PResolvedPipelineGate:
    @pytest.mark.parametrize("group_override", [False, True])
    def test_pipeline_parallelism_is_rejected_before_distributed_setup(
        self, p2p_protocol: ModuleType, group_override: bool
    ) -> None:
        """Resolved global and group-specific pipeline parallelism fail at the supported-protocol boundary."""
        config = make_sglang_config(pp_size=1 if group_override else 2)
        if group_override:
            [model] = config.models
            [group] = model.server_groups
            group = group.model_copy(update={"overrides": {"pp_size": 2}})
            config = config.model_copy(update={"models": [model.model_copy(update={"server_groups": [group]})]})

        with pytest.raises(NotImplementedError, match="pipeline parallelism"):
            p2p_protocol.UpdateWeightP2P(SimpleNamespace(sglang=config))
