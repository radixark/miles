import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.trainer_utils import compute_trainer_config


class TestDsv4SelectionTransport:
    @pytest.mark.parametrize("implementation", ["miles", "megatron"])
    def test_explicit_selection_survives_shared_parser_and_trainer_slicing(self, implementation: str) -> None:
        """An explicit DeepSeek implementation survives trainer slicing as a Miles-owned setting."""
        config = parse_megatron_test_config("--dsv4-impl", implementation)

        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])

        assert config.dsv4_impl == implementation
        assert trainer.dsv4_impl == implementation
        assert "dsv4_impl" not in vars(trainer.backend)
