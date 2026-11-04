from argparse import ArgumentParser

import pytest

from miles.utils.args.configs.custom_megatron_plugins import DsaMegatronPluginsConfig


class TestDsaTopkBackend:
    @pytest.mark.parametrize("selection", [None, "canonical", "torch", "flashinfer"])
    def test_typed_parser_preserves_canonical_default_and_explicit_backend(self, selection: str | None) -> None:
        """Typed registration accepts canonical top-k and preserves explicit legacy selections."""
        parser = ArgumentParser()
        DsaMegatronPluginsConfig.add_arguments(parser=parser)

        argv = [] if selection is None else ["--miles-dsa-topk-backend", selection]
        parsed = parser.parse_args(argv)
        config = DsaMegatronPluginsConfig.model_validate(vars(parsed))

        assert config.miles_dsa_topk_backend == (selection or "canonical")
        assert DsaMegatronPluginsConfig().miles_dsa_topk_backend == "canonical"
