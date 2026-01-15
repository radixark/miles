import sys
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.misc import function_registry


class TestAddArgumentsSupport:

    def test_calls_class_add_arguments(self):
        mock_add_arguments = MagicMock()

        class MyRolloutFn:
            @classmethod
            def add_arguments(cls, parser):
                mock_add_arguments(parser)

        with function_registry.temporary("test:rollout_class", MyRolloutFn):
            with patch.object(sys, "argv", ["test", "--rollout-function-path", "test:rollout_class"]):
                add_miles_arguments = get_miles_extra_args_provider()
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments(parser)

                mock_add_arguments.assert_called_once()

    def test_calls_function_add_arguments(self):
        mock_add_arguments = MagicMock()

        def my_generate_fn():
            pass

        my_generate_fn.add_arguments = mock_add_arguments

        with function_registry.temporary("test:generate_fn", my_generate_fn):
            with patch.object(sys, "argv", ["test", "--custom-generate-function-path", "test:generate_fn"]):
                add_miles_arguments = get_miles_extra_args_provider()
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments(parser)

                mock_add_arguments.assert_called_once()

    def test_skips_function_without_add_arguments(self):
        def my_rollout_fn():
            pass

        with function_registry.temporary("test:rollout_fn", my_rollout_fn):
            with patch.object(sys, "argv", ["test", "--rollout-function-path", "test:rollout_fn"]):
                add_miles_arguments = get_miles_extra_args_provider()
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments(parser)

    def test_skips_none_path(self):
        with patch.object(sys, "argv", ["test"]):
            add_miles_arguments = get_miles_extra_args_provider()
            import argparse
            parser = argparse.ArgumentParser()
            add_miles_arguments(parser)

    def test_custom_arg_is_parsed(self):
        class MyRolloutFn:
            @classmethod
            def add_arguments(cls, parser):
                parser.add_argument("--my-custom-arg", type=int, default=42)

        with function_registry.temporary("test:rollout_with_arg", MyRolloutFn):
            with patch.object(sys, "argv", ["test", "--rollout-function-path", "test:rollout_with_arg", "--my-custom-arg", "100"]):
                add_miles_arguments = get_miles_extra_args_provider()
                import argparse
                parser = argparse.ArgumentParser()
                add_miles_arguments(parser)

                args, _ = parser.parse_known_args()
                assert args.my_custom_arg == 100
