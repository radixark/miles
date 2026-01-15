import argparse
from unittest.mock import MagicMock

import pytest

from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.misc import function_registry


class TestAddArguments:
    def test_calls_class_add_arguments(self):
        mock_add_arguments = MagicMock()

        class MyRolloutFn:
            @classmethod
            def add_arguments(cls, parser):
                mock_add_arguments(parser)

        with function_registry.temporary("test:rollout_class", MyRolloutFn):
            parser = argparse.ArgumentParser()
            add_miles_arguments = get_miles_extra_args_provider()
            parser.add_argument("--rollout-function-path", default="test:rollout_class")
            add_miles_arguments(parser)

            mock_add_arguments.assert_called_once()
            assert isinstance(mock_add_arguments.call_args[0][0], argparse.ArgumentParser)

    def test_calls_function_add_arguments(self):
        mock_add_arguments = MagicMock()

        def my_generate_fn():
            pass

        my_generate_fn.add_arguments = mock_add_arguments

        with function_registry.temporary("test:generate_fn", my_generate_fn):
            parser = argparse.ArgumentParser()
            add_miles_arguments = get_miles_extra_args_provider()
            parser.add_argument("--rollout-function-path", default="miles.rollout.sglang_rollout.generate_rollout")
            parser.add_argument("--custom-generate-function-path", default="test:generate_fn")
            add_miles_arguments(parser)

            mock_add_arguments.assert_called_once()
            assert isinstance(mock_add_arguments.call_args[0][0], argparse.ArgumentParser)

    def test_skips_function_without_add_arguments(self):
        def my_rollout_fn():
            pass

        with function_registry.temporary("test:rollout_fn", my_rollout_fn):
            parser = argparse.ArgumentParser()
            add_miles_arguments = get_miles_extra_args_provider()
            parser.add_argument("--rollout-function-path", default="test:rollout_fn")
            add_miles_arguments(parser)

    def test_skips_none_path(self):
        parser = argparse.ArgumentParser()
        add_miles_arguments = get_miles_extra_args_provider()
        parser.add_argument("--rollout-function-path", default="miles.rollout.sglang_rollout.generate_rollout")
        parser.add_argument("--custom-generate-function-path", default=None)
        add_miles_arguments(parser)

    def test_custom_arg_is_parsed(self):
        class MyRolloutFn:
            @classmethod
            def add_arguments(cls, parser):
                parser.add_argument("--my-custom-arg", type=int, default=42)

        with function_registry.temporary("test:rollout_with_arg", MyRolloutFn):
            parser = argparse.ArgumentParser()
            add_miles_arguments = get_miles_extra_args_provider()
            parser.add_argument("--rollout-function-path", default="test:rollout_with_arg")
            add_miles_arguments(parser)

            args, _ = parser.parse_known_args(["--my-custom-arg", "100"])
            assert args.my_custom_arg == 100
