import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    pass


class _DummyModel:
    pass


class _DummyOptimizer:
    pass


class _DummyChainedOptimizer:
    pass


class _DummyDistributedOptimizer:
    pass


class _DummyScheduler:
    pass


class _DummyOptimizerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = types.SimpleNamespace(model_parallel_cuda_manual_seed=MagicMock())
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": MagicMock()})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.bridge_lora_helpers",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module("miles.backends.megatron_utils.model_provider", {"get_model_provider_func": MagicMock()})
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_called_once_with(increment=800)


def _train_args():
    return Namespace(
        debug_disable_optimizer=True,
        enable_mtp_training=False,
        manual_gc=False,
        overlap_param_gather=False,
        reset_optimizer_states=False,
        use_distributed_optimizer=False,
    )


def _run_one_train_step(model_module, outcome, *, indep_dp_size=1):
    model_chunk = MagicMock()
    data_iterator = MagicMock()
    parallel_state = types.SimpleNamespace(indep_dp=types.SimpleNamespace(size=indep_dp_size))
    config = types.SimpleNamespace()

    with ExitStack() as stack:
        stack.enter_context(patch.object(model_module, "get_args", return_value=_train_args()))
        stack.enter_context(patch.object(model_module, "get_parallel_state", return_value=parallel_state))
        stack.enter_context(patch.object(model_module, "get_model_config", return_value=config))
        stack.enter_context(patch.object(model_module, "train_one_step", return_value=({}, 0.0, outcome)))
        result = model_module.train(
            rollout_id=1,
            model=[model_chunk],
            optimizer=None,
            opt_param_scheduler=None,
            data_iterator=[data_iterator],
            num_microbatches=[1],
            num_rollouts=[1],
            witness_info=None,
            attempt=0,
        )

    return result


def test_discarded_step_pops_r3_stats_without_reduce_or_log():
    from miles.backends.megatron_utils import model as model_module

    manager = model_module.routing_replay_manager
    outcome = model_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY
    with (
        patch.object(manager, "enable_check_replay_result", True),
        patch.object(manager, "pop_check_stats", return_value=(3, 10)) as pop_stats,
        patch.object(model_module, "get_gloo_group") as get_group,
        patch.object(model_module, "reduce_check_stats") as reduce_stats,
        patch.object(model_module, "log_train_step") as log_step,
    ):
        result = _run_one_train_step(model_module, outcome, indep_dp_size=2)

    assert result == outcome
    pop_stats.assert_called_once_with()
    get_group.assert_not_called()
    reduce_stats.assert_not_called()
    log_step.assert_not_called()


def test_normal_non_main_rank_reduces_r3_stats_without_logging():
    from miles.backends.megatron_utils import model as model_module

    manager = model_module.routing_replay_manager
    outcome = model_module.TrainStepOutcome.NORMAL
    group = object()
    with (
        patch.object(manager, "enable_check_replay_result", True),
        patch.object(manager, "pop_check_stats", return_value=(3, 10)) as pop_stats,
        patch.object(model_module, "get_gloo_group", return_value=group),
        patch.object(model_module, "reduce_check_stats", return_value=(2, 8)) as reduce_stats,
        patch.object(model_module, "is_first_replica_megatron_main_rank", return_value=False),
        patch.object(model_module, "log_train_step") as log_step,
    ):
        result = _run_one_train_step(model_module, outcome)

    assert result == outcome
    pop_stats.assert_called_once_with()
    reduce_stats.assert_called_once_with((3, 10), group)
    log_step.assert_not_called()
