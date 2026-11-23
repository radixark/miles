import argparse

import pytest

from tools.convert_hf_to_torch_dist import _ConversionConfig


class TestConversionVocabularyRegistration:
    def test_actual_conversion_registration_resets_an_existing_integer_vocab_default(self) -> None:
        """The conversion schema clears the backend default while retaining explicit integer input."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--padded-vocab-size", type=int, default=64)

        result = _ConversionConfig.add_arguments(parser=parser)

        assert result is parser
        assert parser.parse_args(["--hf-checkpoint", "/unused"]).padded_vocab_size is None
        assert parser.parse_args(["--hf-checkpoint", "/unused", "--padded-vocab-size", "17"]).padded_vocab_size == 17
        assert sum(action.dest == "padded_vocab_size" for action in parser._actions) == 1

    def test_actual_conversion_registration_rejects_an_existing_string_vocab_option(self) -> None:
        """The conversion schema rejects a conflicting backend option instead of preserving its type."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--padded-vocab-size", type=str, default="64")

        with pytest.raises(AssertionError, match="Cannot reset --padded-vocab-size: type"):
            _ConversionConfig.add_arguments(parser=parser)

        assert parser.get_default("padded_vocab_size") == "64"
        assert parser.parse_args(["--hf-checkpoint", "/unused", "--padded-vocab-size", "17"]).padded_vocab_size == "17"
