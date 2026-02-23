import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.arguments import _maybe_apply_dumper_overrides, get_miles_extra_args_provider
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save_interval: int | None = 5,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save_interval=save_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.rollout_health_check_interval == 30.0
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save_interval == 5

    def test_disables_all_heartbeats(self) -> None:
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True
        assert args.rollout_health_check_interval == 1e18

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6
