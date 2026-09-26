import logging
from argparse import Namespace
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.backends.torchtitan_utils import compat
from miles.backends.torchtitan_utils.config import build_trainer_config
from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
from miles.backends.torchtitan_utils.parallel import create_titan_parallel_state, parallel_dims_from_config
from miles.backends.torchtitan_utils.routing_replay import install as install_routing_replay
from miles.backends.torchtitan_utils.trainer import TitanTrainer
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.replay.routing_replay import enable as enable_routing_replay
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.memory_utils import clear_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _steps_per_rollout(args: Namespace) -> int:
    return max(args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size, 1)


class TorchtitanTrainRayActor(TorchNativeTrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo | None = None,
    ) -> int | None:  # type: ignore[override]
        compat.install()
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)
        assert indep_dp_info is None or indep_dp_info.quorum_id == 0

        assert recv_ckpt_src_rank is None, "torchtitan backend does not support checkpoint healing"
        assert not with_opd_teacher, "torchtitan backend does not support on-policy distillation yet"

        config = build_trainer_config(
            args,
            hf_assets_path=args.hf_checkpoint,
            lr_total_steps=args.num_rollout * _steps_per_rollout(args),
            dump_subdir="actor",
        )

        enable_routing_replay(args)

        if args.debug_rollout_only:
            set_parallel_state(create_titan_parallel_state(parallel_dims_from_config(config.parallelism)))
            return 0

        self.prof = TrainProfiler(args)
        self.load_hf_assets()

        self.trainer = TitanTrainer(config)
        self.model_parts = self.trainer.model_parts
        self.optimizers = self.trainer.optimizers.optimizers
        self.align_token_side_channel = self.trainer.align_token_side_channel
        self.trainer.configure_loss_reduction()
        set_parallel_state(
            create_titan_parallel_state(self.trainer.parallel_dims, is_pp_last_stage=self.trainer.has_last_stage())
        )
        install_routing_replay(self.trainer.model_parts)

        cp_mesh = self.trainer.parallel_dims.get_optional_mesh("cp")
        cp_rank0 = cp_mesh is None or dist.get_rank(cp_mesh.get_group()) == 0
        if cp_rank0 and get_parallel_state().is_metrics_rank:
            init_tracking(args, primary=False)

        self.trainer.checkpointer.load()
        start_rollout_id = self.trainer.step // _steps_per_rollout(args)

        if with_ref:
            self.ref_runner = self._build_ref_runner(args)

        self.weight_updater = self._build_weight_updater(self.trainer, TitanHfWeightIterator.build)

        clear_memory()
        if args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return args.start_rollout_id if args.start_rollout_id is not None else start_rollout_id

    def step_runner(self):
        return self.trainer.step_runner()

    def _build_ref_runner(self, args: Namespace):
        if not args.ref_load:
            raise ValueError("--ref-load is required to build a torchtitan reference model")
        ref_config = build_trainer_config(args, hf_assets_path=args.ref_load, lr_total_steps=1, dump_subdir="ref")
        ref_trainer = TitanTrainer(ref_config)
        ref_trainer.configure_loss_reduction()
        ref_trainer.checkpointer.load()
        for part in ref_trainer.model_parts:
            part.eval()
            part.requires_grad_(False)
            part.cpu()
        torch.cuda.empty_cache()
        self._ref_parts = ref_trainer.model_parts
        logger.info(f"Built a torchtitan reference trainer from {args.ref_load}; it lives on the host between passes")
        return ref_trainer.step_runner()

    @contextmanager
    def ref_context(self):
        for part in self._ref_parts:
            part.cuda()
        try:
            yield
        finally:
            for part in self._ref_parts:
                part.cpu()
            torch.cuda.empty_cache()

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.args.save is None:
            return
        assert not self.args.async_save, "TorchtitanTrainRayActor does not support async_save yet."
        self.trainer.checkpointer.save(self.trainer.step, last_step=True)
