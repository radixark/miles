#!/usr/bin/env python3
"""
Ray-free SFT Training Script for Miles.

This script provides a simplified training path for Supervised Fine-Tuning (SFT)
that bypasses Ray entirely and uses torchrun for distributed training.

Usage:
    torchrun --nproc_per_node=2 train_sft.py \
        --hf-checkpoint /path/to/model \
        --prompt-data /path/to/data.parquet \
        --input-key messages \
        --apply-chat-template \
        ...

This is equivalent to the Ray-based SFT with --debug-train-only, but without
the Ray overhead.
"""

import logging
import os
from argparse import Namespace
from datetime import timedelta
from itertools import accumulate

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm
from transformers import AutoConfig

from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.actor import (
    apply_fsdp2,
    get_logprob_and_entropy_with_cp,
    sum_of_sample_mean,
)
from miles.backends.fsdp_utils.data_packing import (
    pack_sequences,
    pad_packed_sequence_with_cp,
    unpack_sequences,
)
from miles.backends.fsdp_utils.lr_scheduler import get_lr_scheduler
from miles.rollout.data_source import RolloutDataSource
from miles.utils import tracking_utils
from miles.utils.arguments import parse_args
from miles.utils.data import get_minimum_num_micro_batch_size
from miles.utils.distributed_utils import get_gloo_group, init_gloo_group
from miles.utils.logging_utils import configure_logger
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.misc import should_run_periodic_action
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    A simplified trainer for SFT that runs without Ray.

    This class combines the functionality of:
    - FSDPTrainRayActor (model initialization, FSDP wrapping, training)
    - RolloutManager (data loading via generate_rollout)
    - The main training loop from train.py
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.device = torch.device("cuda")

        self._init_distributed()

        self._setup_device_mesh()

        torch.manual_seed(args.seed)

        self._enable_true_on_policy_optimizations()

        if dist.get_rank() == 0:
            init_tracking(args, primary=True)

        self._load_tokenizer_and_config()

        self._init_data_source()

        self._init_model()

        self._init_optimizer()

        self._load_checkpoint()

        self.prof = TrainProfiler(args)
        self.prof.on_init_end()

        logger.info(f"[Rank {dist.get_rank()}] SFTTrainer initialized successfully")

    def _init_distributed(self):
        """Initialize distributed training."""
        # torchrun sets these environment variables
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        backend = self.args.distributed_backend
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=self.args.distributed_timeout_minutes),
        )
        init_gloo_group()

        self.args.rank = dist.get_rank()
        self.args.world_size = dist.get_world_size()

        logger.info(
            f"[Rank {self.args.rank}] Distributed initialized: "
            f"world_size={self.args.world_size}, local_rank={local_rank}"
        )

    def _setup_device_mesh(self):
        """Setup device mesh for FSDP (no context parallelism for SFT)."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.cp_size = self.args.context_parallel_size
        self.dp_size = world_size // self.cp_size

        self.mesh = init_device_mesh(
            "cuda",
            mesh_shape=(self.dp_size, self.cp_size),
            mesh_dim_names=("dp", "cp"),
        )

        self.dp_group = self.mesh.get_group("dp")
        self.cp_group = self.mesh.get_group("cp")
        self.dp_mesh = self.mesh["dp"]

        self.dp_rank = rank // self.cp_size
        self.cp_rank = rank % self.cp_size

        logger.info(
            f"[Rank {rank}] Device mesh: dp_size={self.dp_size}, cp_size={self.cp_size}, "
            f"dp_rank={self.dp_rank}, cp_rank={self.cp_rank}"
        )

        # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
        if self.cp_size > 1:
            substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
            logger.info(f"[Rank {rank}] CP initialized via device mesh")

    def _enable_true_on_policy_optimizations(self):
        """Enable true on-policy optimizations or apply MoE patches."""
        if self.args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            from miles.backends.fsdp_utils.models.qwen3_moe import (
                apply_true_on_policy_patch_for_qwen3_moe,
            )

            logger.info("SFTTrainer: enabling batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope uses bmm; disabling makes it aligned
                enable_bmm=False,
            )

            apply_true_on_policy_patch_for_qwen3_moe()
        else:
            from miles.backends.fsdp_utils.models.qwen3_moe_hf import apply_fsdp_moe_patch

            apply_fsdp_moe_patch()

    def _load_tokenizer_and_config(self):
        """Load tokenizer and model config sequentially to avoid race conditions."""
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(
                    self.args.hf_checkpoint, trust_remote_code=True
                )
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, trust_remote_code=True
                )
                self.processor = None
                if self.args.multimodal_keys:
                    self.processor = load_processor(
                        self.args.hf_checkpoint, trust_remote_code=True
                    )
            dist.barrier(group=get_gloo_group())

        # Initialize loss mask generator for SFT
        self.mask_generator = MultiTurnLossMaskGenerator(
            self.tokenizer,
            tokenizer_type=getattr(self.args, "loss_mask_type", None),
        )

    def _init_data_source(self):
        """Initialize the data source for SFT training."""
        self.data_source = RolloutDataSource(self.args)

        # Calculate num_rollout from dataset size
        if self.args.num_rollout is None:
            num_rollout_per_epoch = len(self.data_source.dataset) // self.args.rollout_batch_size
            self.args.num_rollout = num_rollout_per_epoch * self.args.num_epoch
            self.num_rollout_per_epoch = num_rollout_per_epoch
        else:
            self.num_rollout_per_epoch = None

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        logger.info(
            f"[Rank {dist.get_rank()}] Data source initialized: "
            f"dataset_size={len(self.data_source.dataset)}, "
            f"num_rollout={self.args.num_rollout}"
        )

    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization."""
        from accelerate import init_empty_weights

        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with broadcast from rank 0."""
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(
            full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True
        )

        set_model_state_dict(model, full_state, options=options)

        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    def _get_model_cls(self):
        """Get the appropriate model class based on config."""
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _init_model(self):
        """Initialize and wrap model with FSDP."""
        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)

        init_context = self._get_init_weight_context_manager()

        with init_context():
            model = self._get_model_cls().from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )

        model.train()
        full_state = model.state_dict()

        model = apply_fsdp2(
            model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args
        )

        model = self._fsdp2_load_full_state_dict(
            model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = model

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info(f"[Rank {dist.get_rank()}] Model initialized with FSDP")

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_eps,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

        self.lr_scheduler = get_lr_scheduler(self.args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

    def _load_checkpoint(self):
        """Load checkpoint if available."""
        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

    def generate_sft_rollout(self, rollout_id: int) -> list[Sample]:
        """Generate SFT rollout data (tokenize and create loss masks)."""
        samples = self.data_source.get_samples(self.args.rollout_batch_size)

        result = []
        for i, (sample,) in enumerate(samples):
            messages = sample.prompt
            token_ids, loss_mask = self.mask_generator.get_loss_mask(messages)
            response_length = self.mask_generator.get_response_lengths([loss_mask])[0]

            sample.tokens = token_ids
            sample.response_length = response_length
            sample.reward = 0
            sample.loss_mask = loss_mask[-response_length:]
            result.append(sample)

            if i == 0 and rollout_id == 0 and dist.get_rank() == 0:
                logger.info(
                    f"SFT rollout sample: tokens_len={len(token_ids)}, "
                    f"response_length={response_length}"
                )

        return result

    def _convert_samples_to_train_data(self, samples: list[Sample]) -> dict:
        """Convert samples to training data format."""
        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "rewards": [0.0 for _ in samples],
            "raw_reward": [0.0 for _ in samples],
            "truncated": [0 for _ in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        loss_masks = []
        for sample in samples:
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        return train_data

    def _split_train_data_by_dp(self, data: dict) -> dict:
        """Split training data for current DP rank."""
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        # Simple round-robin partitioning
        partition = list(range(self.dp_rank, len(total_lengths), self.dp_size))

        rollout_data = {"partition": partition, "total_lengths": total_lengths}

        for key in [
            "tokens",
            "response_lengths",
            "rewards",
            "raw_reward",
            "truncated",
            "loss_masks",
            "sample_indices",
        ]:
            if key in data:
                rollout_data[key] = [data[key][j] for j in partition]

        return rollout_data

    def _packed_data(self, rollout_data: dict) -> tuple[list[dict], list[int]]:
        """Pack variable-length sequences for efficient processing."""
        tokens = rollout_data["tokens"]

        packed_batches = []
        mbs_size_list = []
        local_batch_size = self.args.global_batch_size // self.dp_size

        if self.args.use_dynamic_batch_size:
            max_tokens = self.args.max_tokens_per_gpu
            if self.cp_size > 1:
                max_tokens = max_tokens * self.cp_size

            for i in range(0, len(tokens), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                        max_tokens,
                    )
                )
            num_microbatches = torch.tensor(
                mbs_size_list, dtype=torch.int, device=torch.cuda.current_device()
            )
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
            num_microbatches = num_microbatches.tolist()
        else:
            num_microbatches = [
                self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)
            ] * (len(tokens) // local_batch_size)

        start = 0
        for mbs_size in num_microbatches:
            end = start + local_batch_size
            # Create dummy advantages/returns for SFT (not used but required by pack_sequences)
            dummy_advantages = [
                torch.zeros(rollout_data["response_lengths"][i])
                for i in range(start, end)
            ]
            packed_batches.extend(
                pack_sequences(
                    rollout_data["tokens"][start:end],
                    rollout_data["loss_masks"][start:end],
                    rollout_data["rewards"][start:end],
                    rollout_data["raw_reward"][start:end],
                    rollout_data["response_lengths"][start:end],
                    dummy_advantages,  # advantages
                    dummy_advantages,  # returns
                    num_packs=mbs_size,
                )
            )
            start = end

        grad_accum = list(accumulate(num_microbatches))
        return packed_batches, grad_accum

    def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
        """Prepare model input arguments from packed sequence."""
        input_ids = packed_sequence["tokens"].unsqueeze(0)
        position_ids = packed_sequence["position_ids"].unsqueeze(0)

        if self.cp_size > 1:
            packed_sequence = pad_packed_sequence_with_cp(packed_sequence, self.cp_size)

            if not packed_sequence["cu_seqlens"].is_cuda:
                packed_sequence["cu_seqlens"] = packed_sequence["cu_seqlens"].cuda()
            cu_seqlens = packed_sequence["cu_seqlens"]
            update_ring_flash_attn_params(cu_seqlens, self.cp_group)

            input_ids = torch.chunk(
                packed_sequence["tokens"].unsqueeze(0), self.cp_size, dim=1
            )[self.cp_rank]
            position_ids = torch.chunk(
                packed_sequence["position_ids"].unsqueeze(0), self.cp_size, dim=1
            )[self.cp_rank]

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }

        if packed_sequence.get("multimodal_inputs"):
            model_args.update(packed_sequence["multimodal_inputs"])

        return model_args

    def _compute_sft_loss(self, unpacked_batches: list[dict], logits: torch.Tensor):
        """Compute SFT loss (negative log likelihood)."""
        loss_masks = [
            batch["loss_masks"].to(device=logits.device) for batch in unpacked_batches
        ]
        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]
        log_probs = torch.cat(
            [batch["cur_log_probs"] for batch in unpacked_batches], dim=0
        )
        loss = -sum_of_sample_mean(log_probs, response_lengths, loss_masks)

        if log_probs.numel() == 0:
            loss += 0 * logits.sum()

        return loss, {"loss": loss.detach()}

    def _train_step(
        self,
        packed_batch: dict,
        reported_accum: dict,
        mbs_id: int,
        grad_accum: list[int],
    ):
        """Execute one training step."""
        # Prepare model inputs
        model_args = self._get_model_inputs_args(packed_batch)
        logits = self.model(**model_args).logits.squeeze(0).float()

        # Compute log probs and entropy (unified for both CP and non-CP modes)
        log_probs, entropy_result = get_logprob_and_entropy_with_cp(
            logits=logits,
            target_tokens=packed_batch["tokens"],
            cp_rank=self.cp_rank,
            cp_size=self.cp_size,
            cp_group=self.cp_group,
            model_input_ids=model_args["input_ids"],
            allow_compile=not self.args.true_on_policy_mode,
            temperature=self.args.rollout_temperature,
        )
        packed_batch["cur_log_probs"] = log_probs
        packed_batch["entropy"] = entropy_result

        unpacked_batches = unpack_sequences(packed_batch)
        loss, reported = self._compute_sft_loss(unpacked_batches, logits)

        # Scale loss for gradient accumulation
        loss = loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        # Accumulate reported metrics (store tensors for later mean)
        for k, v in reported.items():
            reported_accum.setdefault(k, []).append(v)

        if (mbs_id + 1) in grad_accum:
            # TODO: check if the grad norm is global grad norm.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            # the grad norm used to be of DTensor
            grad_norm = float(grad_norm)

            self.optimizer.step()
            # Update learning rate
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Aggregate logs
            aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
            # TODO: change this, this is slow.
            reduced_aggregated = [None] * self.dp_size
            dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
            aggregated = {}
            for k in reported_accum.keys():
                aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
            reported_accum.clear()
            if dist.get_rank() == 0:
                log_dict = {
                    f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                }
                log_dict["train/grad_norm"] = grad_norm

                # Log learning rate per parameter group; use scheduler's last computed LRs
                lr_values = self.lr_scheduler.get_last_lr()
                for gid, _group in enumerate(self.optimizer.param_groups):
                    log_dict[f"train/lr_{gid}"] = lr_values[gid]

                logger.info(f"step {self.global_step}: {log_dict}")
                log_dict["train/step"] = self.global_step
                tracking_utils.log(self.args, log_dict, step_key="train/step")
            self.global_step += 1

    def train_one_rollout(self, rollout_id: int):
        """Execute one rollout's worth of training."""
        samples = self.generate_sft_rollout(rollout_id)

        train_data = self._convert_samples_to_train_data(samples)

        rollout_data = self._split_train_data_by_dp(train_data)

        packed_batches, grad_accum = self._packed_data(rollout_data)

        if len(grad_accum) == 0:
            logger.warning(f"[Rank {dist.get_rank()}] No batches to train on rollout {rollout_id}")
            return

        with timer("actor_train"):
            reported_accum = {}
            self.optimizer.zero_grad(set_to_none=True)

            for mbs_id, packed_batch in enumerate(
                tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0)
            ):
                self._train_step(packed_batch, reported_accum, mbs_id, grad_accum)

        self.prof.step(rollout_id=rollout_id)

    def save_model(self, iteration: int):
        """Save model checkpoint."""
        if self.args.save is None:
            return
        checkpoint.save(self, iteration)

    def train(self):
        """Main training loop."""
        logger.info(
            f"[Rank {dist.get_rank()}] Starting training: "
            f"rollout_id {self.args.start_rollout_id} -> {self.args.num_rollout}"
        )

        for rollout_id in range(self.args.start_rollout_id, self.args.num_rollout):
            self.train_one_rollout(rollout_id)

            # Save checkpoint periodically
            if should_run_periodic_action(
                rollout_id, self.args.save_interval, self.num_rollout_per_epoch
            ):
                self.save_model(rollout_id)

        logger.info(f"[Rank {dist.get_rank()}] Training completed!")


def set_sft_defaults(args: Namespace) -> Namespace:
    """Set default values appropriate for SFT training."""
    if not hasattr(args, "loss_type") or args.loss_type is None:
        args.loss_type = "sft_loss"

    if not hasattr(args, "advantage_estimator"):
        args.advantage_estimator = None

    args.offload_train = False
    args.offload_rollout = False
    args.colocate = False

    return args


def main():
    configure_logger()

    args = parse_args()

    args = set_sft_defaults(args)

    trainer = SFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()


