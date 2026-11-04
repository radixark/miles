from types import ModuleType

import pytest
from tests.fast.fixtures.args_fixtures import make_trainer_config


class TestInklingMultimodalProviderConfig:
    def test_multimodal_provider_applies_the_supplied_context_parallel_constraint_before_loading_assets(
        self, inkling_model: ModuleType
    ) -> None:
        """The multimodal wrapper preserves the text provider's explicit context-parallel safety check."""
        args = make_trainer_config(context_parallel_size=2, allgather_cp=False)
        original = args.model_dump()

        with pytest.raises(AssertionError, match="Inkling CP requires --allgather-cp"):
            inkling_model.inkling_mm_model_provider(pre_process=False, post_process=True, vp_stage=1, args=args)

        assert args.model_dump() == original
