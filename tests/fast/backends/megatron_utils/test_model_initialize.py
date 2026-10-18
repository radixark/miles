import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from miles.backends.megatron_utils.model import LoadCheckpointOutput


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
                "get_checkpoint_name": MagicMock(),
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
    optimizer = MagicMock()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, True)))
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
    optimizer = MagicMock()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, True)))
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
    *,
    tmp_path: Path,
    finetune: bool,
    iteration: int,
    restored_trained_iteration: bool | None = None,
    adapter_iteration: int | None = None,
    lora_adapter_path: str | None = None,
) -> "LoadCheckpointOutput":
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir()
    (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    if restored_trained_iteration is None:
        restored_trained_iteration = not finetune or iteration > 0

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.load_checkpoint",
                return_value=(iteration, restored_trained_iteration),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.restore_lora_adapter", return_value=adapter_iteration)
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            Namespace(
                use_checkpoint_opt_param_scheduler=True,
                global_batch_size=8,
                finetune=finetune,
                lora_rank=0 if lora_adapter_path is None else 8,
                megatron_to_hf_mode="core",
                lora_adapter_path=lora_adapter_path,
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
        output = _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=0)

        assert output.start_rollout_id == 1

    def test_weight_initialization_with_a_trained_iteration_is_refused(self, tmp_path: Path):
        """Weight initialization cannot claim a nonzero trained iteration."""
        with pytest.raises(AssertionError, match="Weight initialization returned a trained iteration"):
            _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100, restored_trained_iteration=False)


class TestALoraAdapterThatCarriesItsOwnIteration:
    """LoRA saves write no tracker, so a LoRA resume loads the base under --finetune (iteration 0, nothing
    trained) and the adapter checkpoint alone says whether, and where, training stopped."""

    def _start(self, tmp_path: Path, adapter_iteration: int | None) -> "LoadCheckpointOutput":
        return _load_model_state_with(
            tmp_path=tmp_path,
            finetune=True,
            iteration=0,
            adapter_iteration=adapter_iteration,
            lora_adapter_path=str(tmp_path / "adapter"),
        )

    def test_a_resume_at_iteration_zero_continues_after_it(self, tmp_path: Path) -> None:
        """An adapter's saved iteration zero is a trained iteration, not a missing training state."""
        assert self._start(tmp_path, adapter_iteration=0).start_rollout_id == 1

    def test_a_resume_continues_after_the_iteration_the_adapter_names(self, tmp_path: Path) -> None:
        output = self._start(tmp_path, adapter_iteration=100)

        assert (output.loaded_rollout_id, output.start_rollout_id) == (100, 101)

    def test_adapter_weights_without_training_state_start_at_rollout_zero(self, tmp_path: Path) -> None:
        """An HF adapter or shards saved without training state carry no iteration to continue from."""
        assert self._start(tmp_path, adapter_iteration=None).start_rollout_id == 0


class _MastersUnderChunkedOffload:
    """Where a --chunked-optimizer-state-offload optimizer keeps its fp32 masters: built on GPU, refreshed there by
    reload_model_params, and moved to host by offload_optimizer_state_for_forward."""

    def __init__(self):
        self.masters_on_gpu = True

    def reload_model_params(self):
        self.masters_on_gpu = True

    def offload_optimizer_state_for_forward(self):
        self.masters_on_gpu = False


class TestTheFirstForwardAfterALoad:
    def test_finds_the_masters_the_load_refreshed_already_on_host(self, tmp_path: Path):
        """Offloaded only after a step, they would share the first fwd/bwd with the weights and gradients;
        Megatron's loop moves them before its first forward too."""
        from miles.backends.megatron_utils.model import load_model_state

        load_dir = tmp_path / "hf"
        load_dir.mkdir()
        (load_dir / "config.json").write_text("{}")
        optimizer = _MastersUnderChunkedOffload()

        def load_hf_checkpoint(model, optimizer, *args, **kwargs):
            # as _load_checkpoint_hf does after writing the weights
            optimizer.reload_model_params()
            return 0, 0

        with ExitStack() as stack:
            stack.enter_context(
                patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=load_hf_checkpoint)
            )
            _patch_initialize_side_effects(stack)
            load_model_state(
                Namespace(
                    use_checkpoint_opt_param_scheduler=True,
                    global_batch_size=8,
                    finetune=True,
                    lora_rank=0,
                    load=str(load_dir),
                ),
                model=[_FakeModelChunk()],
                optimizer=optimizer,
                opt_param_scheduler=None,
                role="actor",
                checkpointing_context=None,
            )

        assert not optimizer.masters_on_gpu
