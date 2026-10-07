import sys
import types
from argparse import Namespace
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from miles.backends.megatron_utils.model import LoadCheckpointOutput

try:
    # Import the real scheduler before the module fixture below swaps megatron for stubs.
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler as _MegatronScheduler
except ImportError:
    _MegatronScheduler = None


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


class _DummyPackedSeqParams:
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
        core_module.tensor_parallel = _stub_module(
            "megatron.core.tensor_parallel",
            {"model_parallel_cuda_manual_seed": MagicMock()},
            is_package=True,
        )
        _stub_module(
            "megatron.core.tensor_parallel.random",
            {"_get_all_rng_states": MagicMock(), "_set_all_rng_states": MagicMock()},
        )
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
                "Adam": _DummyOptimizer,
                "CPUAdam": _DummyOptimizer,
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.emerging_optimizers", {"TensorParallelMuon": _DummyOptimizer})
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
        # A class rather than a MagicMock: megatron_utils/parallel.py subclasses it under @dataclass.
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": _DummyPackedSeqParams})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock(), "unwrap_model": MagicMock()})
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
            "miles.backends.megatron_utils.lora.bridge",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.model_provider",
            {
                "get_model_provider_func": MagicMock(),
                "LinearForLastLayer": _DummyModel,
            },
        )
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8, finetune=False)
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
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0, False)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8, finetune=False)
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
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0, False)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.step.assert_called_once_with(increment=800)


def _load_model_state_with(
    *, tmp_path: Path, finetune: bool, iteration: int, lora_rank: int = 0
) -> "LoadCheckpointOutput":
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir()
    (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    with ExitStack() as stack:
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(iteration, 0, False))
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            Namespace(
                use_checkpoint_opt_param_scheduler=True,
                global_batch_size=8,
                finetune=finetune,
                lora_rank=lora_rank,
                megatron_to_hf_mode="core",
                lora_adapter_path=None,
                load=str(load_dir),
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=None,
            role="actor",
            checkpointing_context=None,
        )


class TestWhereALoadSaysTheRunStarts:
    def test_a_finetune_load_starts_the_run_at_rollout_zero(self, tmp_path: Path):
        """--finetune means there is no run to continue, so rollout 0 is still ahead rather than behind."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=0).start_rollout_id == 0

    def test_a_resumed_load_starts_the_run_after_the_checkpoint_it_read(self, tmp_path: Path):
        """The checkpoint's own rollout is done, so the run continues at the next one."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=100).start_rollout_id == 101

    def test_a_run_that_restored_the_iteration_zero_checkpoint_it_wrote_starts_at_one(self, tmp_path: Path):
        """A real resume from the very first checkpoint must not be read as a finetune that starts over."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=0).start_rollout_id == 1

    def test_a_finetune_load_that_found_a_checkpoint_is_refused(self, tmp_path: Path):
        """--finetune promises iteration 0; anything else means the two disagree about where the run stands."""
        with pytest.raises(AssertionError, match="disagree about where this run stands"):
            _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100)


class TestALoraAdapterThatCarriesItsOwnIteration:
    def test_a_lora_resume_under_finetune_continues_after_the_iteration_the_adapter_names(self, tmp_path: Path):
        """LoRA saves write no tracker, so a lora resume always arrives here with --finetune set."""
        output = _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100, lora_rank=8)

        assert output.start_rollout_id == 101

    def test_a_lora_run_that_really_starts_from_scratch_still_starts_at_rollout_one(self, tmp_path: Path):
        """An adapter with no training state answers iteration 0, and the run continues from the next rollout."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=0, lora_rank=8).start_rollout_id == 1


_GLOBAL_BATCH_SIZE = 256
_OPTIMIZER_STEPS_PER_ROLLOUT = 2


def _megatron_scheduler(*, use_checkpoint_opt_param_scheduler: bool):
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0, "default_config": True}])
    return _MegatronScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=0.0,
        lr_warmup_steps=10 * _GLOBAL_BATCH_SIZE,
        lr_decay_steps=400 * _GLOBAL_BATCH_SIZE,
        lr_decay_style="cosine",
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=400 * _GLOBAL_BATCH_SIZE,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=False,
    )


def _load_model_state_into(
    scheduler, *, tmp_path: Path, iteration: int, use_checkpoint_opt_param_scheduler: bool, load_checkpoint
) -> "LoadCheckpointOutput":
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir()
    (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    with ExitStack() as stack:
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=load_checkpoint))
        _patch_initialize_side_effects(stack)
        return load_model_state(
            Namespace(
                use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
                global_batch_size=_GLOBAL_BATCH_SIZE,
                finetune=False,
                lora_rank=0,
                megatron_to_hf_mode="core",
                lora_adapter_path=None,
                load=str(load_dir),
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=scheduler,
            role="actor",
            checkpointing_context=None,
        )


def _restoring_load_checkpoint(*, saved_state: dict, iteration: int):
    """A load that restores the scheduler as Megatron's load_checkpoint does without --no-load-optim."""

    def load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        opt_param_scheduler.load_state_dict(saved_state)
        return iteration, 0, False

    return load_checkpoint


@pytest.mark.skipif(_MegatronScheduler is None, reason="needs Megatron-LM on PYTHONPATH, as the CPU CI has")
class TestTheSchedulerAfterALoad:
    @pytest.mark.parametrize("use_checkpoint_opt_param_scheduler", [False, True])
    @pytest.mark.parametrize("saved_rollout", [0, 4, 49, 149])
    def test_a_resume_continues_the_schedule_the_checkpoint_saved(
        self, tmp_path: Path, saved_rollout: int, use_checkpoint_opt_param_scheduler: bool
    ):
        """Megatron's load replays the saved num_steps whatever the flag says, so stepping again would skip ahead."""
        uninterrupted = _megatron_scheduler(use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler)
        for _ in range((saved_rollout + 1) * _OPTIMIZER_STEPS_PER_ROLLOUT):
            uninterrupted.step(increment=_GLOBAL_BATCH_SIZE)
        saved_state = uninterrupted.state_dict()

        resumed = _megatron_scheduler(use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler)
        output = _load_model_state_into(
            resumed,
            tmp_path=tmp_path,
            iteration=saved_rollout,
            use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
            load_checkpoint=_restoring_load_checkpoint(saved_state=saved_state, iteration=saved_rollout),
        )

        assert output.start_rollout_id == saved_rollout + 1
        assert resumed.num_steps == uninterrupted.num_steps
        assert resumed.optimizer.param_groups == uninterrupted.optimizer.param_groups

    def test_a_resume_from_a_checkpoint_saved_before_any_optimizer_step_stays_at_step_zero(self, tmp_path: Path):
        """fp16 loss scaling can skip every early optimizer step, so a later checkpoint may still hold num_steps 0."""
        resumed = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False)
        saved_state = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False).state_dict()

        _load_model_state_into(
            resumed,
            tmp_path=tmp_path,
            iteration=5,
            use_checkpoint_opt_param_scheduler=False,
            load_checkpoint=_restoring_load_checkpoint(saved_state=saved_state, iteration=5),
        )

        assert resumed.num_steps == 0

    def test_a_load_that_did_not_restore_the_scheduler_still_steps_it_by_the_iteration(self, tmp_path: Path):
        """With --no-load-optim Megatron leaves the scheduler alone, so this load is the only thing that moves it."""
        resumed = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False)

        _load_model_state_into(
            resumed,
            tmp_path=tmp_path,
            iteration=99,
            use_checkpoint_opt_param_scheduler=False,
            load_checkpoint=lambda *args, **kwargs: (99, 0, False),
        )

        assert resumed.num_steps == 99 * _GLOBAL_BATCH_SIZE

    @pytest.mark.parametrize("load_fails", [False, True])
    def test_the_scheduler_gets_its_own_load_state_dict_back_after_the_load(self, tmp_path: Path, load_fails: bool):
        """Nothing of the restore record stays on the scheduler, also when the load raises."""
        scheduler = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False)
        restoring_load_checkpoint = _restoring_load_checkpoint(saved_state=scheduler.state_dict(), iteration=0)

        def load_checkpoint(*args, **kwargs):
            result = restoring_load_checkpoint(*args, **kwargs)
            if load_fails:
                raise RuntimeError("the checkpoint read failed")
            return result

        with pytest.raises(RuntimeError, match="the checkpoint read failed") if load_fails else nullcontext():
            _load_model_state_into(
                scheduler,
                tmp_path=tmp_path,
                iteration=0,
                use_checkpoint_opt_param_scheduler=False,
                load_checkpoint=load_checkpoint,
            )

        assert "load_state_dict" not in vars(scheduler)
