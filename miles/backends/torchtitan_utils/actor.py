"""TrainRayActor over torchtitan's Trainer.

miles builds the config tree, instantiates the trainer, drives it through the
shared RL loop, and streams weights to the rollout engines. Nothing here
touches torchtitan internals.
"""

import logging
from argparse import Namespace

import torch.distributed as dist

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.torchtitan_utils.config import build_trainer_config
from miles.backends.torchtitan_utils.parallel import create_titan_parallel_state, parallel_dims_from_config
from miles.backends.torchtitan_utils.trainer import TitanTrainer
from miles.backends.torchtitan_utils.weight_bridge import get_hf_weight_iterator
from miles.backends.training_utils.model_assets import load_model_assets
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native_actor import TorchNativeTrainRayActor
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.context_utils import with_defer
from miles.utils.memory_utils import clear_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _steps_per_rollout(args: Namespace) -> int:
    return max(args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size, 1)


class TorchtitanTrainRayActor(TorchNativeTrainRayActor):
    routing_replay = routing_replay

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info=None,
    ) -> int | None:  # type: ignore[override]
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        assert recv_ckpt_src_rank is None, "torchtitan backend does not support checkpoint healing"
        assert not with_opd_teacher, "torchtitan backend does not support on-policy distillation yet"

        config = build_trainer_config(
            args,
            hf_assets_path=args.hf_checkpoint,
            lr_total_steps=args.num_rollout * _steps_per_rollout(args),
            dump_subdir="actor",
        )

        routing_replay.enable(args)

        if args.debug_rollout_only:
            set_parallel_state(create_titan_parallel_state(parallel_dims_from_config(config.parallelism)))
            self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
            return 0

        self.prof = TrainProfiler(args)

        assets = load_model_assets(args)
        self.hf_config = assets.hf_config
        self.tokenizer = assets.tokenizer

        self.trainer = TitanTrainer(config)
        self.model_parts = self.trainer.model_parts
        self.optimizers = self.trainer.optimizers.optimizers
        self.align_token_side_channel = self.trainer.align_token_side_channel
        self._init_flops(self.hf_config)
        self.trainer.enable_context_parallel_gather()
        set_parallel_state(
            create_titan_parallel_state(self.trainer.parallel_dims, is_pp_last_stage=self.trainer.has_last_stage())
        )
        self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
        routing_replay.install(self.trainer.model_parts)

        state = get_parallel_state()
        cp_mesh = self.trainer.parallel_dims.get_optional_mesh("cp")
        reports = cp_mesh is None or dist.get_rank(cp_mesh.get_group()) == 0
        if reports and state.effective_dp_cp.rank == 0 and state.tp.rank == 0 and state.is_pp_last_stage:
            init_tracking(args, primary=False)

        self.trainer.checkpointer.load()
        start_rollout_id = self.trainer.step // _steps_per_rollout(args)

        if with_ref:
            self.ref_runner = self._build_ref_runner(args)

        self.weight_updater = WeightUpdater(
            args,
            self.trainer,
            weights_getter=lambda: None,
            model_name=type(self.hf_config).__name__.lower() if args.model_name is None else args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            iterator_factory=get_hf_weight_iterator,
            parallel_state=get_parallel_state(),
            is_lora=False,
        )

        clear_memory()
        if args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return int(getattr(args, "start_rollout_id", None) or start_rollout_id)

    def step_runner(self):
        return self.trainer.step_runner()

    def _build_ref_runner(self, args: Namespace):
        """A frozen, CPU-offloaded second trainer for reference log probs."""
        if not args.ref_load:
            raise ValueError("--ref-load is required to build a torchtitan reference model")
        ref_config = build_trainer_config(args, hf_assets_path=args.ref_load, lr_total_steps=1, dump_subdir="ref")
        ref_config.training.enable_cpu_offload = True
        ref_trainer = TitanTrainer(ref_config)
        ref_trainer.checkpointer.load()
        for part in ref_trainer.model_parts:
            part.eval()
            part.requires_grad_(False)
        logger.info(f"Built a CPU-offloaded torchtitan reference trainer from {args.ref_load}")
        return ref_trainer.step_runner()

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.args.save is None:
            return
        assert not self.args.async_save, "TorchtitanTrainRayActor does not support async_save yet."
        self.trainer.checkpointer.save(self.trainer.step, last_step=True)
