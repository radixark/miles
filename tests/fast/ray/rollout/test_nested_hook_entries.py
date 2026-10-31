from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import FakeInferenceTopologyProvider, make_rollout_config
from tests.fast.ray.rollout.nested_hook_fixtures import (
    _generate,
    _HookBoundaryReached,
    _RecordingRollout,
    _RecordingSource,
)

from miles.ray.rollout.eval_fleet import EvalFleetInfo
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.args.component_rollout import InferenceRuntimeImmutState
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.function_registry import function_registry
from miles.utils.workers.worker_spec import HostAndPort


def _hook(path: str, **values: Any) -> CustomFunctionConfig:
    return CustomFunctionConfig(path=path, config=ImmutableNamespace(**values))


class TestNestedHookEntries:
    @pytest.mark.parametrize("shared", [False, True])
    async def test_real_executor_construction_and_eval_fleet_keep_all_nested_hook_values(
        self, monkeypatch: pytest.MonkeyPatch, shared: bool
    ) -> None:
        """Executor constructors and eval-fleet state retain merged config while equal rollout hooks share state."""
        monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "0")
        monkeypatch.setattr(inference_rollout_common, "load_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(inference_rollout_common, "load_processor", lambda *args, **kwargs: None)
        train_hook = _hook("test-train-rollout", train_setting=1)
        eval_hook = train_hook if shared else _hook("test-eval-rollout", eval_setting=2)
        args = make_rollout_config(
            debug_train_only=True,
            enable_sample_ownership_checker=False,
            data_source_path=_hook("test-source", source_setting=6),
            rollout_function_path=train_hook,
            eval_function_path=eval_hook,
            custom_generate_function_path=_hook("test-generate", generate_setting=3),
            custom_agent_function_path=_hook("test-agent", agent_setting=4),
            custom_rm_path=_hook("test-reward", reward_setting=5),
        ).model_copy(
            update={
                "eval_datasets": [
                    EvalDatasetConfig(
                        name="eval",
                        path="unused.jsonl",
                        custom_generate_function_path=_hook("test-dataset", dataset_setting=7),
                    )
                ]
            }
        )
        provider = FakeInferenceTopologyProvider(InferenceRuntimeImmutState(engine_count=0, gpu_count=0))
        executor = RolloutExecutor(
            args=args, router_providers=[], session_server_provider=None, inference_controller_provider=provider
        )

        with (
            function_registry.temporary("test-source", _RecordingSource),
            function_registry.temporary("test-train-rollout", _RecordingRollout),
            function_registry.temporary("test-eval-rollout", _RecordingRollout),
            function_registry.temporary("test-generate", _generate),
        ):
            try:
                await executor.init()
                train_args = executor.generate_rollout.constructor_input.args
                eval_args = executor.eval_generate_rollout.constructor_input.args
                await executor.set_eval_fleet_info(
                    EvalFleetInfo(router=HostAndPort(host="eval", port=30001), engine_gpu_counts=[1])
                )
                fleet_args = executor._eval_fleet._state.args

                assert executor.data_source.args.source_setting == 6
                assert train_args.train_setting == 1
                assert (executor.eval_generate_rollout is executor.generate_rollout) is shared
                assert dict(eval_args)["train_setting" if shared else "eval_setting"] == (1 if shared else 2)
                for view in (train_args, eval_args, fleet_args):
                    assert (view.generate_setting, view.agent_setting, view.reward_setting, view.dataset_setting) == (
                        3,
                        4,
                        5,
                        7,
                    )
                assert dict(fleet_args)["train_setting" if shared else "eval_setting"] == (1 if shared else 2)
                assert (fleet_args.sglang_router_ip, fleet_args.sglang_router_port) == ("eval", 30001)
                assert args.sglang_router_ip != "eval"
            finally:
                if hasattr(executor, "use_legacy_rollout_v1"):
                    await executor.dispose()

    @pytest.mark.parametrize("evaluation", [False, True])
    async def test_legacy_train_and_eval_entries_receive_the_selected_merged_hook_view(self, evaluation: bool) -> None:
        """Both legacy entrypoints pass their selected hook and nested dataset settings to the user callback."""
        captured: list[ImmutableNamespace] = []

        def rollout(args: ImmutableNamespace, rollout_id: int, data_source: Any, *, evaluation: bool) -> None:
            captured.append(args)
            assert rollout_id == 9
            raise _HookBoundaryReached

        args = ImmutableNamespace(
            load_debug_rollout_data=None,
            debug_train_only=False,
            eval_uses_snapshots=False,
            starts_inference_engines=False,
            rollout_function_path=_hook("train", train_setting=1),
            eval_function_path=_hook("eval", eval_setting=2),
            custom_generate_function_path=None,
            custom_agent_function_path=None,
            custom_rm_path=None,
            eval_datasets=[
                EvalDatasetConfig(
                    name="eval", path="unused", custom_generate_function_path=_hook("dataset", dataset_setting=7)
                )
            ],
        )
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = args
        executor.use_legacy_rollout_v1 = True
        executor.generate_rollout = rollout
        executor.eval_generate_rollout = rollout
        executor.data_source = object()

        with pytest.raises(_HookBoundaryReached):
            if evaluation:
                await executor.eval(rollout_id=9)
            else:
                await executor._get_rollout_data(rollout_id=9)

        assert len(captured) == 1
        assert dict(captured[0])["eval_setting" if evaluation else "train_setting"] == (2 if evaluation else 1)
        assert captured[0].dataset_setting == 7
        assert "dataset_setting" not in dict(args)
