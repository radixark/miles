import shlex
from dataclasses import replace

import pytest

from tests.e2e.ft.conftest_ft.execution import get_ft_args
from tests.e2e.ft.conftest_ft.modes import FTTestMode


class TestGetFtArgs:
    @pytest.mark.parametrize(
        "components,extra,expected",
        [
            (("train",), (), ["train"]),
            (("rollout",), (), ["rollout", "weight_update"]),
            (("train", "rollout"), (), ["train", "rollout", "weight_update"]),
            (("train",), ("rollout",), ["train", "rollout", "weight_update"]),
        ],
    )
    def test_rollout_faults_enable_weight_update_isolation(
        self, components: tuple[str, ...], extra: tuple[str, ...], expected: list[str]
    ) -> None:
        """Receiver fault scenarios must retain partial-target updates without changing injected fault kinds."""
        mode = FTTestMode(
            model_name="test",
            model_hf_repo="test",
            megatron_model_type="test",
            num_cells=2,
            parallel_args="",
            rollout_num_engines=2,
            ft_components=components,
        )
        original = replace(mode)

        tokens = shlex.split(get_ft_args(mode, extra_ft_components=extra))
        start = tokens.index("--ft-components") + 1
        end = tokens.index("--api-server-port")

        assert tokens[start:end] == expected
        assert mode == original
        assert {"--enable-event-analyzer", "--enable-witness"} <= set(tokens)
