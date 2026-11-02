from argparse import Namespace
from functools import partial

import pytest


class TestTrainerConfigurationBoundary:
    @pytest.mark.parametrize("entrypoint", ["setup_model_and_optimizer", "initialize_model_and_optimizer"])
    def test_model_entrypoints_reject_unscoped_namespaces_before_accessing_model_state(self, entrypoint: str) -> None:
        """A launcher namespace cannot enter trainer model initialization."""
        from miles.backends.megatron_utils.model import initialize_model_and_optimizer, setup_model_and_optimizer

        functions = {
            "setup_model_and_optimizer": setup_model_and_optimizer,
            "initialize_model_and_optimizer": partial(initialize_model_and_optimizer, resume_from_ckpt=False),
        }

        with pytest.raises(AssertionError):
            functions[entrypoint](args=Namespace())
