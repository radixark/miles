import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.misc import function_registry


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

    def train(self) -> None:
        pass


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
        _stub_module(
            "miles.backends.megatron_utils.ft.indep_dp",
            {"allreduce_grads_and_losses_across_replicas": MagicMock(return_value=(True, {}))},
        )
        _stub_module(
            "miles.backends.megatron_utils.local_weight_checksum", {"dump_local_weight_checksums": MagicMock()}
        )
        _stub_module("miles.utils.audit_utils.witness.allocator", {"WitnessInfo": MagicMock()})
        _stub_module("miles.utils.audit_utils.witness.module", {"witness_dump_and_clear_stale": MagicMock()})
        _stub_module(
            "miles.utils.dumper_utils",
            {
                "DumperMegatronUtil": MagicMock(),
                "DumperPhase": types.SimpleNamespace(FWD_BWD="fwd_bwd", FWD_ONLY="fwd_only"),
            },
        )
        _stub_module("miles.utils.memory_utils", {"clear_memory": MagicMock()})
        _stub_module("miles.utils.test_utils.ft_test_actions", {"FTTestActionActorExecutor": MagicMock()})
        _stub_module("miles.utils.tracking_utils.structured_log", {"log_structured": MagicMock()})
        _stub_module(
            "miles.backends.training_utils.ci_utils", {"check_grad_norm": MagicMock(), "check_kl": MagicMock()}
        )
        _stub_module("miles.backends.training_utils.data", {"DataIterator": MagicMock(), "get_batch": MagicMock()})
        _stub_module(
            "miles.backends.training_utils.log_utils",
            {
                "aggregate_forward_results": MagicMock(),
                "aggregate_train_losses": MagicMock(return_value={}),
                "log_train_step": MagicMock(),
            },
        )
        _stub_module("miles.backends.training_utils.loss", {"loss_function": MagicMock()})
        _stub_module("miles.backends.training_utils.parallel", {"get_parallel_state": MagicMock()})
        _stub_module(
            "miles.backends.megatron_utils.checkpoint",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
                "save_checkpoint_with_lora": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.ci_utils",
            {
                "check_model_hashes": MagicMock(),
                "check_peak_gpu_memory_after_load": MagicMock(),
                "compute_model_hashes_by_layer": MagicMock(),
                "save_model_hashes": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.initialize",
            {"is_first_replica_megatron_main_rank": MagicMock(return_value=True)},
        )
        _stub_module(
            "miles.backends.megatron_utils.lora_utils",
            {
                "is_lora_enabled": MagicMock(return_value=False),
                "is_lora_model": MagicMock(return_value=False),
                "save_lora_checkpoint": MagicMock(),
            },
        )
        _stub_module("miles.backends.megatron_utils.model_provider", {"get_model_provider_func": MagicMock()})
        _stub_module("miles.backends.megatron_utils.parallel", {"get_packed_seq_params": MagicMock()})
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


def test_train_invokes_after_train_step_hook_before_logging():
    from miles.backends.megatron_utils.ft.types import TrainStepOutcome
    from miles.backends.megatron_utils.model import train

    args = Namespace(
        debug_disable_optimizer=True,
        custom_megatron_after_train_step_hook_path="test:after_train_step_hook",
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        reset_optimizer_states=False,
        manual_gc=False,
        enable_mtp_training=False,
        ci_test=False,
    )
    model = [_FakeModelChunk()]
    data_iterator = [MagicMock()]
    config = SimpleNamespace(no_sync_func=None, param_sync_func=None)
    parallel_state = SimpleNamespace(indep_dp=SimpleNamespace(size=1))
    hook_calls = []

    def after_train_step_hook(
        hook_args,
        rollout_id,
        step_id,
        hook_model,
        optimizer,
        opt_param_scheduler,
        loss_dict,
        num_microbatches,
    ):
        loss_dict["custom_metric"] = 2.5
        hook_calls.append(
            (
                hook_args,
                rollout_id,
                step_id,
                hook_model,
                optimizer,
                opt_param_scheduler,
                loss_dict,
                num_microbatches,
            )
        )

    with function_registry.temporary("test:after_train_step_hook", after_train_step_hook), ExitStack() as stack:
        stack.enter_context(patch("miles.backends.megatron_utils.model.get_args", return_value=args))
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.get_parallel_state", return_value=parallel_state)
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.get_model_config", return_value=config))
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.should_disable_forward_pre_hook", return_value=False)
        )
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.train_one_step",
                return_value=({"loss": 1.0}, 0.5, TrainStepOutcome.NORMAL),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.is_first_replica_megatron_main_rank", return_value=True)
        )
        log_train_step = stack.enter_context(
            patch("miles.backends.megatron_utils.model.log_train_step", return_value={"loss": 1.0})
        )

        result = train(
            rollout_id=3,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            data_iterator=data_iterator,
            num_microbatches=[7],
            witness_info=None,
            attempt=0,
        )

    assert result == TrainStepOutcome.NORMAL
    assert hook_calls == [
        (
            args,
            3,
            0,
            model,
            None,
            None,
            {"loss": 1.0, "custom_metric": 2.5},
            7,
        )
    ]
    log_train_step.assert_called_once()
    assert log_train_step.call_args.kwargs["loss_dict"] == {"loss": 1.0, "custom_metric": 2.5}


def test_train_skips_after_train_step_hook_for_discarded_step():
    from miles.backends.megatron_utils.ft.types import TrainStepOutcome
    from miles.backends.megatron_utils.model import train

    args = Namespace(
        debug_disable_optimizer=True,
        custom_megatron_after_train_step_hook_path="test:after_train_step_hook_discarded",
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        reset_optimizer_states=False,
        manual_gc=False,
        enable_mtp_training=False,
        ci_test=False,
    )
    model = [_FakeModelChunk()]
    data_iterator = [MagicMock()]
    config = SimpleNamespace(no_sync_func=None, param_sync_func=None)
    parallel_state = SimpleNamespace(indep_dp=SimpleNamespace(size=1))
    after_train_step_hook = MagicMock()

    with function_registry.temporary(
        "test:after_train_step_hook_discarded", after_train_step_hook
    ), ExitStack() as stack:
        stack.enter_context(patch("miles.backends.megatron_utils.model.get_args", return_value=args))
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.get_parallel_state", return_value=parallel_state)
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.get_model_config", return_value=config))
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.should_disable_forward_pre_hook", return_value=False)
        )
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.train_one_step",
                return_value=({}, 0.0, TrainStepOutcome.DISCARDED_SHOULD_RETRY),
            )
        )
        log_train_step = stack.enter_context(patch("miles.backends.megatron_utils.model.log_train_step"))

        result = train(
            rollout_id=3,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            data_iterator=data_iterator,
            num_microbatches=[7],
            witness_info=None,
            attempt=0,
        )

    assert result == TrainStepOutcome.DISCARDED_SHOULD_RETRY
    after_train_step_hook.assert_not_called()
    log_train_step.assert_not_called()
