from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.inkling_provider_fixtures import inkling_tower_env

from miles.utils.args.custom_function import CustomFunctionConfig

_ = inkling_tower_env


class TestInklingMultimodalTowers:
    @pytest.mark.parametrize(
        ("provider", "materialize", "expected_names"),
        [
            (
                CustomFunctionConfig(path="miles_plugins.models.inkling.model.inkling_mm_model_provider"),
                True,
                ["audio.weight", "visual.weight"],
            ),
            (
                CustomFunctionConfig(path="miles_plugins.models.inkling.model.inkling_mm_model_provider"),
                False,
                [],
            ),
            (CustomFunctionConfig(path="miles_plugins.models.inkling.model.inkling_model_provider"), True, []),
            (None, True, []),
        ],
        ids=["multimodal", "non-materializing-rank", "text", "no-provider"],
    )
    def test_only_materializing_multimodal_provider_emits_frozen_tower_weights(
        self,
        inkling_tower_env: SimpleNamespace,
        provider: CustomFunctionConfig | None,
        materialize: bool,
        expected_names: list[str],
    ) -> None:
        """A typed multimodal provider sends frozen towers while text and non-materializing paths remain empty."""
        env = inkling_tower_env
        args = SimpleNamespace(custom_model_provider_path=provider, hf_checkpoint=str(env.checkpoint))

        units = list(env.module._iter_mm_tower_units(args, materialize=materialize))

        assert [unit[0][0] for unit in units] == expected_names
        for [(name, tensor)] in units:
            torch.testing.assert_close(tensor, env.tensors[name])
