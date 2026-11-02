from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from tests.fast.backends.megatron_utils.test_model import train_one_step_env
from tests.fast.backends.megatron_utils.test_shared_ppo_lifecycle import actor_module
from tests.fast.fixtures.args_fixtures import make_trainer_config
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.backends.megatron_utils import initialize
from miles.backends.megatron_utils import model as model_module
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.args.runtime import TrainerConfig
from miles.utils.function_registry import function_registry

if TYPE_CHECKING:
    from tests.fast.backends.megatron_utils.test_model import TrainOneStepEnv

__all__ = ["actor_module", "train_one_step_env"]


class _HookReached(Exception):
    pass


class TestTypedModelHooks:
    def test_before_train_step_hook_receives_backend_and_plugin_fields_at_the_real_callsite(
        self, train_one_step_env: TrainOneStepEnv
    ) -> None:
        """The train-step hook receives merged backend fields and original step operands before SDK execution."""
        env = train_one_step_env
        hook = _hook()
        args = TrainerConfig.model_construct(**vars(env.args)).model_copy(
            update={"custom_megatron_before_train_step_hook_path": hook}
        )

        def callback(
            view: ImmutableNamespace, rollout_id: int, step_id: int, models: Any, optimizer: Any, scheduler: Any
        ) -> None:
            assert view.hook_marker == 41
            assert view.micro_batch_size == args.backend.micro_batch_size
            assert (rollout_id, step_id) == (7, 3)
            assert models is env.model
            assert optimizer is scheduler is None
            raise _HookReached

        with function_registry.temporary(hook.path, callback), pytest.raises(_HookReached):
            model_module.train_one_step(
                args, 7, 3, env.data_iterator, env.model, None, None, 1, 1, witness_info=None, attempt=0
            )

    def test_before_log_prob_hook_receives_backend_fields_and_store_identity(self) -> None:
        """Log-prob hooks receive flattened trainer configuration and the selected score prefix."""
        hook = _hook()
        args = make_trainer_config(custom_megatron_before_log_prob_hook_path=hook, dumper_enable=False)
        chunk = torch.nn.Linear(1, 1)
        chunk.config = SimpleNamespace()
        models = [chunk]

        def callback(view: ImmutableNamespace, received: list[torch.nn.Module], prefix: str) -> None:
            assert view.hook_marker == 41
            assert view.lr == args.backend.lr
            assert received is models
            assert not chunk.training
            assert prefix == "reference_"
            raise _HookReached

        with function_registry.temporary(hook.path, callback), pytest.raises(_HookReached):
            model_module.forward_only(
                lambda *args, **kwargs: {}, args, models, [], [], rollout_id=7, store_prefix="reference_"
            )

    def test_initialization_hook_receives_flat_custom_fields_after_tokenizer_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The initialization hook merges plugin and backend fields after the SDK tokenizer mutation completes."""
        hook = _hook()
        args = make_trainer_config(
            custom_megatron_init_path=hook,
            enable_experimental=False,
            rank=0,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
            deterministic_mode=False,
            tp_comm_overlap=False,
            indep_dp=False,
            debug_deterministic_collective=False,
        )
        for name in (
            "set_args",
            "_initialize_distributed",
            "create_indep_dp_group",
            "create_megatron_parallel_state",
            "set_parallel_state",
            "set_random_seed_from_args",
            "register_hf_config_aliases",
            "init_num_microbatches_calculator",
        ):
            monkeypatch.setattr(initialize, name, lambda *args, **kwargs: None)
        monkeypatch.setattr(initialize, "_build_tokenizer", lambda backend: setattr(backend, "padded_vocab_size", 128))
        monkeypatch.setattr(initialize.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(initialize.dist, "get_world_size", lambda: 1)
        seen = []

        def callback(view: ImmutableNamespace) -> None:
            assert view.hook_marker == 41
            assert view.padded_vocab_size == 128
            assert view.lr == args.backend.lr
            seen.append(view)

        initialize_args = args
        with initialize_args.backend.mutable():
            initialize_args.backend.rank = 0
        with function_registry.temporary(hook.path, callback):
            initialize.init(initialize_args)
        assert len(seen) == 1
        assert "hook_marker" not in dict(args)

    @pytest.mark.parametrize("rank,expected_calls", [(0, 1), (1, 0)])
    def test_post_save_hook_is_main_rank_only_and_receives_actual_checkpoint_destination(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, rank: int, expected_calls: int
    ) -> None:
        """The save callback retains plugin settings and checkpoint location while non-main ranks skip it."""
        hook = _hook()
        worker = object.__new__(actor_module.MegatronTrainRayActor)
        worker.args = make_trainer_config(
            custom_megatron_post_save_hook_path=hook,
            save="/checkpoints",
            async_save=False,
            save_hf=None,
            ci_test=False,
            use_distributed_optimizer=False,
        )
        worker.role = "actor"
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        worker.model = [torch.nn.Linear(1, 1)]
        worker.model[0].lora_attached = False
        worker.optimizer = worker.opt_param_scheduler = worker.snapshot_publisher = None
        saved = []
        monkeypatch.setattr(model_module, "save_checkpoint", lambda *args, **kwargs: saved.append(args[0]))
        monkeypatch.setattr(actor_module.dist, "get_rank", lambda: rank)
        seen = []

        def callback(
            view: ImmutableNamespace, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None
        ) -> None:
            assert view.hook_marker == 41
            assert view.save == "/checkpoints"
            assert rollout_id == 7
            assert checkpoint_dir == "/checkpoints/iter_0000007"
            assert hf_checkpoint_dir is None
            seen.append(view)

        with function_registry.temporary(hook.path, callback):
            worker.save_model(7)
        assert saved == [7]
        assert len(seen) == expected_calls

    def test_rollout_postprocess_receives_plugin_fields_before_the_training_step(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The actor's rollout postprocessor gets custom values before downstream training consumes the batch."""
        from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

        make_parallel_state()
        hook = _hook()
        worker = object.__new__(actor_module.MegatronTrainRayActor)
        worker.args = make_trainer_config(
            rollout_data_postprocess_path=hook,
            compute_advantages_and_returns=False,
            skip_actor_forward_only=False,
            use_dynamic_global_batch_size=False,
        )
        worker.model = [torch.nn.Linear(1, 1)]
        worker.model[0].lora_attached = False
        rollout_data = {"micro_batch_indices": [[0]], "num_microbatches": [1]}

        def callback(view: ImmutableNamespace) -> None:
            assert view.hook_marker == 41
            assert view.lr == worker.args.backend.lr
            assert "hook_marker" not in dict(worker.args)
            raise _HookReached

        worker.rollout_data_postprocess = callback
        from miles.utils.timer import Timer

        clock = Timer()
        monkeypatch.setattr(clock, "start_time", {})
        monkeypatch.setattr(clock, "timers", {})
        clock.start("train_wait")
        with pytest.raises(_HookReached):
            worker._train_actor(7, rollout_data, witness_info=None, attempt=0)
        clock.end("train_wait")
