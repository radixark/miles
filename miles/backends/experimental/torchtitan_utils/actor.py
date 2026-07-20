"""TorchTitanTrainRayActor: the TrainRayActor contract implemented on torchtitan.

Loop bodies mirror fsdp_utils/actor.py's shape over the same shared training_utils
(shared loss/data/logging), so wandb curves stay apples-to-apples with the FSDP and
megatron backends. The only forward-shaped seam is _get_model_inputs_args (titan
Decoder.forward(tokens, positions, attention_masks) instead of an HF .logits call) and
the build/load/optimizer path, which is torchtitan-native end to end per the project
directive to reuse torchtitan's training logic and kernels wholesale.
"""

import logging
from argparse import Namespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.ray.train_actor import TrainRayActor
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.hf_config import load_hf_config
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking

from ....utils.profile_utils import TrainProfiler
from ...training_utils.ci_utils import check_grad_norm
from ...training_utils.data import get_batch, get_data_iterator, get_rollout_data
from ...training_utils.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_rollout_data,
    log_train_step,
)
from ...training_utils.loss import compute_advantages_and_returns, loss_function
from ...training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from ...training_utils.parallel import get_parallel_state, set_parallel_state
from . import checkpoint, models
from .model import build_and_load_model, build_optimizer_and_lr_scheduler, clip_grad_norm
from .parallel import create_torchtitan_parallel_state
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_manager import EnginesAndLock
    from miles.utils.audit_utils.witness.allocator import WitnessInfo

logger = logging.getLogger(__name__)


def move_optimizer_state(optimizers, device: str) -> None:
    """Move every inner optimizer's state tensors between CPU and CUDA, generalizing
    fsdp_utils' move_torch_optimizer to OptimizersContainer's multiple inner optimizers."""
    for opt in optimizers.optimizers:
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


class TorchTitanTrainRayActor(TrainRayActor):
    """torchtitan-backed TrainRayActor: torchtitan supplies the model/parallelize/
    optimizer/grad-clip; miles supplies rollout-data packing, loss, weight sync, and
    checkpoint conventions."""

    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
    ) -> int | None:  # type: ignore[override]
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        assert recv_ckpt_src_rank is None, "torchtitan backend v1 does not support checkpoint fan-out"
        assert indep_dp_info.quorum_id == 0
        assert role == "actor", "torchtitan backend v1 supports the actor role only"
        assert args.tt_expert_parallel_size == 1, "EP>1 FSDP wrapping is broken on torch 2.11 (v1 scope: EP=1)"
        assert args.context_parallel_size == 1, "torchtitan CP is unwired in this backend"

        set_parallel_state(create_torchtitan_parallel_state(args))
        torch.manual_seed(args.seed)

        self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}

        if args.debug_rollout_only:
            return 0

        if args.offload_train:
            logger.info("offload_train enabled: torchtitan model/optimizer will sleep between rollouts")

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = load_hf_config(self.args.hf_checkpoint)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        spec, hf = models.spec_from_hf(self.args.hf_checkpoint, attn_backend=self.args.tt_attn_backend)
        parallel_dims = get_parallel_state().parallel_dims
        self.model, self.adapter = build_and_load_model(
            spec,
            self.args.hf_checkpoint,
            parallel_dims=parallel_dims,
            seq_len=self.args.rollout_max_response_len + 4096,  # headroom for prompt tokens
            args=self.args,
        )
        self.spec = spec
        self.model.train()

        self.optimizer, self.lr_scheduler = build_optimizer_and_lr_scheduler(
            self.model, spec, self.args, parallel_dims, training_steps=self.args.num_rollout
        )

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        self.ref_model = None
        if with_ref:
            raise NotImplementedError("torchtitan backend v1 does not yet support a reference model")

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model, self.adapter)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model, self.adapter)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        self.max_tokens_per_gpu = args.max_tokens_per_gpu

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()
        return int(getattr(self.args, "start_rollout_id", 0))

    def sleep(self) -> None:
        if not self.args.offload_train:
            return
        print_memory("before offload model")
        self.model.cpu()
        move_optimizer_state(self.optimizer, "cpu")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return
        self.model.cuda()
        move_optimizer_state(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.args.save is None:
            return
        assert not self.args.async_save, "TorchTitanTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

    def _get_model_inputs_args(self, batch: dict) -> dict:
        return {
            "tokens": batch["tokens"],
            "positions": batch["position_ids"],
            "attention_masks": self.model.get_attention_masks(batch["position_ids"]),
        }

    def _compute_log_prob(self, model_tag, data_iterator, num_microbatches, store_prefix: str = ""):
        results = []
        for batch in data_iterator:
            model_args = self._get_model_inputs_args(batch)
            with torch.no_grad():
                logits = self.model(**model_args).float()
            log_probs = get_log_probs_and_entropy(
                logits,
                args=self.args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                with_entropy=False,
            )
            results.append(log_probs)
        return aggregate_forward_results(results, store_prefix=store_prefix)

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info: "WitnessInfo | None" = None,
        attempt: int = 0,
    ) -> None:
        assert witness_info is None, "torchtitan backend v1 does not support witness-based fault tolerance"
        assert attempt == 0
        if self.args.debug_rollout_only:
            return

        rollout_data = get_rollout_data(self.args, rollout_data_ref, witness_info)
        log_rollout_data(self.args, rollout_id, rollout_data)
        compute_advantages_and_returns(self.args, rollout_data)
        self._train_core(rollout_id, rollout_data)

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        data_iterators, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        data_iterator = data_iterators[0]
        assert len(num_microbatches) > 0

        train_losses = []
        for step_id, n_mb in enumerate(num_microbatches):
            train_losses.append(self._train_step(data_iterator, step_id, n_mb))
            self.global_step += 1

        grad_norm = clip_grad_norm(self.model, self.args.clip_grad, get_parallel_state().parallel_dims)
        grad_norm = grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else float(grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if self.args.ci_test:
            check_grad_norm(self.args, grad_norm, rollout_id, self.global_step)

        log_train_step(self.args, rollout_id, aggregate_train_losses(train_losses), grad_norm=grad_norm)

    def _train_step(self, data_iterator, step_id, num_microbatches):
        batch = next(data_iterator)
        model_args = self._get_model_inputs_args(batch)
        logits = self.model(**model_args).float()

        loss, normalizer, log_dict = loss_function(
            args=self.args,
            batch=batch,
            num_microbatches=num_microbatches,
            logits=logits,
            apply_megatron_loss_scaling=False,
        )
        (loss / normalizer).backward()
        self.micro_step += 1
        return log_dict

    @timer
    def update_weights(self, info: "EnginesAndLock") -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        was_offloaded = self.args.offload_train
        if was_offloaded:
            self.wake_up()

        self.weight_updater.connect_rollout_engines(info.rollout_engines, info.rollout_engine_lock)
        self.weight_updater.update_weights()

        if was_offloaded:
            self.sleep()
        clear_memory()

    def _get_parallel_config(self):
        return self.train_parallel_config
