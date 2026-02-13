from __future__ import annotations

import logging
import os
from argparse import Namespace
import json

import torch
import torch.distributed as dist
import numpy as np
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from flow_grpo import rewards as flow_rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.ema import EMAModuleWrapper

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.data import process_rollout_data
from miles.utils.diffusion_protocol import broadcast_advantage
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.timer import Timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.metric_utils import compute_rollout_step
from miles.utils import tracking_utils

from .actor import apply_fsdp2, move_torch_optimizer
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


class DiffusionFSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO (Stage3 minimal)."""

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
        }

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.diffusion_model,
            torch_dtype=torch.float16 if args.diffusion_dtype == "fp16" else torch.float32,
        )
        # Keep the full pipeline for scheduler/encoding, but only train the transformer.
        self.pipeline.to(torch.cuda.current_device())
        self.pipeline.transformer.train()

        # Wrap the transformer with FSDP to enable data-parallel training.
        self.pipeline.transformer = apply_fsdp2(
            self.pipeline.transformer,
            mesh=self.parallel_state.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
        )
        self.model = self.pipeline.transformer

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.ema = None
        self.ema_parameters = None
        if getattr(self.args, "diffusion_ema", False):
            self.ema_parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.ema = EMAModuleWrapper(
                self.ema_parameters,
                decay=getattr(self.args, "diffusion_ema_decay", 0.9),
                update_step_interval=getattr(self.args, "diffusion_ema_update_interval", 8),
                device=torch.cuda.current_device(),
            )

        return int(getattr(self.args, "start_rollout_id", 0))

    @timer
    def sleep(self) -> None:  # type: ignore[override]
        if not self.args.offload_train:
            return

        print_memory("before offload diffusion model")
        # Some diffusers pipelines do not expose .cpu(); use .to("cpu") instead.
        self.pipeline.to("cpu")
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload diffusion model")

    @timer
    def wake_up(self) -> None:  # type: ignore[override]
        if not self.args.offload_train:
            return

        self.pipeline.to(torch.cuda.current_device())
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up diffusion model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        logger.warning("DiffusionFSDPTrainRayActor save_model is not implemented; skipping checkpoint save.")

    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.colocate and getattr(self.args, "diffusion_train", False):
            dist.barrier(group=get_gloo_group())
            logger.info("update_weights skipped in colocate diffusion mode (same-process rollout uses latest weights)")
            return

        # Diffusion rollout uses a local pipeline; sync weights via disk.
        rank = dist.get_rank()
        logger.info("update_weights start (rank=%s)", rank)
        base_dir = getattr(self.args, "save", None) or os.path.join("/tmp", "miles_rollout_weights")
        os.makedirs(base_dir, exist_ok=True)
        weights_path = os.path.join(base_dir, "diffusion_transformer.pt")
        meta_path = os.path.join(base_dir, "diffusion_transformer.meta.json")

        # Export full state dict through FSDP2 checkpoint API.
        # This avoids persisting local shards and prevents rollout-side shape mismatch.
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state = get_model_state_dict(self.model, options=options)

        converted_state: dict[str, torch.Tensor] = {}
        dtensor_count = 0
        # Iterate on all ranks; DTensor.full_tensor may require collective participation.
        for k, v in list(model_state.items()):
            if isinstance(v, DTensor):
                dtensor_count += 1
                v = v.full_tensor()
            if isinstance(v, torch.Tensor) and rank == 0:
                converted_state[k] = v.detach().cpu()

        if rank == 0:
            logger.info("update_weights rank=0 converted state dict (dtensor_count=%s)", dtensor_count)
            logger.info("update_weights rank=0 start torch.save: %s", weights_path)
            torch.save(converted_state, weights_path)
            logger.info("update_weights rank=0 torch.save done: %s", weights_path)
            version = int(getattr(self, "_rollout_weight_version", 0)) + 1
            meta = {"version": version}
            tmp_meta_path = meta_path + ".tmp"
            with open(tmp_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            os.replace(tmp_meta_path, meta_path)
            self._rollout_weight_version = version
            logger.info("update_weights wrote version=%s", version)
        else:
            logger.info("update_weights skip write on rank=%s", rank)

        # Keep rank progress aligned; rank0 may take much longer due disk write.
        dist.barrier(group=get_gloo_group())
        logger.info("update_weights done (rank=%s)", rank)
        return

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        # Diffusion GRPO does not use a critic; keep the hook for interface compliance.
        return

    def _get_parallel_config(self) -> dict:  # type: ignore[override]
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log with Flow-GRPO-aligned keys."""
        if "lr" not in log_dict and hasattr(self, "optimizer"):
            try:
                log_dict["lr"] = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if self.parallel_state.dp_cp_rank == 0:
            dp_size = self.parallel_state.dp_cp_size
            gathered = [None] * dp_size
            dist.gather_object(
                log_dict,
                gathered,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )
            reduced = {k: sum(d[k] for d in gathered) / dp_size for k in log_dict}
            reduced["epoch"] = float(rollout_id)
            reduced["inner_epoch"] = 0.0
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            reduced["global_step"] = float(step)
            if "reward_ocr" in reduced:
                logger.info(
                    "train step=%s lr=%.6g weight_decay=%.6g adam_beta1=%.3f adam_beta2=%.3f adam_eps=%.2e warmup_ratio=%.3f clipfrac=%.6f clipfrac_gt_one=%.6f clipfrac_lt_one=%.6f reward_avg=%.6f reward_ori_avg=%.6f reward_ocr=%.6f reward_ocr_min=%.6f reward_ocr_max=%.6f reward_ocr_zero_ratio=%.6f",
                    int(step),
                    reduced.get("lr", 0.0),
                    float(getattr(self.args, "weight_decay", -1.0)),
                    float(getattr(self.args, "adam_beta1", -1.0)),
                    float(getattr(self.args, "adam_beta2", -1.0)),
                    float(getattr(self.args, "adam_eps", -1.0)),
                    float(getattr(self.args, "warmup_ratio", -1.0)),
                    reduced.get("clipfrac", -1.0),
                    reduced.get("clipfrac_gt_one", -1.0),
                    reduced.get("clipfrac_lt_one", -1.0),
                    reduced.get("reward_avg", 0.0),
                    reduced.get("reward_ori_avg", 0.0),
                    reduced.get("reward_ocr", 0.0),
                    reduced.get("reward_ocr_min", 0.0),
                    reduced.get("reward_ocr_max", 0.0),
                    reduced.get("reward_ocr_zero_ratio", 0.0),
                )
            tracking_utils.log(self.args, reduced, step_key="global_step")
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )

    def _log_local_step_metrics(self, rollout_id: int, reduced: dict[str, float], step: int) -> None:
        """Always emit a local rank0 train-step line for easier debugging in colocate mode."""
        if self.parallel_state.dp_rank != 0:
            return
        logger.info(
            "local train step=%s rollout_id=%s loss=%.6f policy_loss=%.6f clipfrac=%.6f "
            "clipfrac_gt_one=%.6f clipfrac_lt_one=%.6f reward_avg=%.6f reward_ocr=%.6f reward_ocr_zero_ratio=%.6f",
            int(step),
            int(rollout_id),
            float(reduced.get("loss", 0.0)),
            float(reduced.get("policy_loss", 0.0)),
            float(reduced.get("clipfrac", 0.0)),
            float(reduced.get("clipfrac_gt_one", 0.0)),
            float(reduced.get("clipfrac_lt_one", 0.0)),
            float(reduced.get("reward_avg", 0.0)),
            float(reduced.get("reward_ocr", 0.0)),
            float(reduced.get("reward_ocr_zero_ratio", 0.0)),
        )

    def _encode_prompt(self, prompts: list[str]):
        # Encode prompts into embeddings on the training device.
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=[""] * len(prompts),
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=self.args.diffusion_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=torch.cuda.current_device(),
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if self.args.diffusion_cfg:
            # Concatenate negative/positive embeds for classifier-free guidance.
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        return prompt_embeds, pooled_prompt_embeds

    def _calculate_zero_std_ratio(
        self, prompts: list[str], rewards: list[float]
    ) -> tuple[float, float]:
        prompt_array = np.array(prompts)
        rewards_array = np.array(rewards, dtype=np.float64)
        if len(prompt_array) == 0:
            return 0.0, 0.0
        unique_prompts, inverse_indices, counts = np.unique(
            prompt_array, return_inverse=True, return_counts=True
        )
        if len(unique_prompts) == 0:
            return 0.0, 0.0
        grouped_rewards = rewards_array[np.argsort(inverse_indices)]
        split_indices = np.cumsum(counts)[:-1]
        reward_groups = np.split(grouped_rewards, split_indices)
        prompt_std_devs = np.array([np.std(group) for group in reward_groups])
        zero_std_count = np.count_nonzero(prompt_std_devs == 0)
        zero_std_ratio = zero_std_count / len(prompt_std_devs)
        return float(zero_std_ratio), float(prompt_std_devs.mean())

    def _compute_per_prompt_advantages(
        self, prompts: list[str], rewards: torch.Tensor, num_train_timesteps: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        use_per_prompt = not getattr(self.args, "diffusion_disable_per_prompt_stat_tracking", False)
        rewards_1d = rewards.detach().float()
        rewards_bt = rewards_1d.unsqueeze(1).repeat(1, num_train_timesteps)
        if not use_per_prompt:
            advantages = (rewards_bt - rewards_bt.mean()) / (rewards_bt.std() + 1e-4)
            stats = {
                "group_size": 0.0,
                "trained_prompt_num": 0.0,
                "zero_std_ratio": 0.0,
                "reward_std_mean": 0.0,
            }
            return advantages, stats

        # Gather prompts and rewards across ranks to match Flow-GRPO global grouping.
        group = self.parallel_state.dp_cp_group_gloo
        local_payload = {
            "prompts": prompts,
            "rewards": rewards_bt.detach().float().cpu().tolist(),
        }
        gathered = [None] * self.parallel_state.dp_cp_size
        dist.all_gather_object(gathered, local_payload, group=group)

        all_prompts: list[str] = []
        all_rewards: list[list[float]] = []
        lengths: list[int] = []
        for item in gathered:
            all_prompts.extend(item["prompts"])
            all_rewards.extend(item["rewards"])
            lengths.append(len(item["prompts"]))

        global_std = bool(getattr(self.args, "diffusion_global_std", 1))
        tracker = PerPromptStatTracker(global_std=global_std)
        advantages_all = tracker.update(all_prompts, all_rewards)
        group_size, trained_prompt_num = tracker.get_stats()
        zero_std_ratio, reward_std_mean = self._calculate_zero_std_ratio(all_prompts, all_rewards)
        tracker.clear()

        # Split back to local slice by rank order.
        local_rank = dist.get_rank(group=group)
        offset = 0
        local_adv = []
        for rank, length in enumerate(lengths):
            if rank == local_rank:
                local_adv = advantages_all[offset : offset + length]
                break
            offset += length

        stats = {
            "group_size": float(group_size),
            "trained_prompt_num": float(trained_prompt_num),
            "zero_std_ratio": float(zero_std_ratio),
            "reward_std_mean": float(reward_std_mean),
        }

        return torch.tensor(local_adv, device=torch.cuda.current_device(), dtype=torch.float32), stats

    def _parse_reward_spec(self) -> dict[str, float]:
        spec = getattr(self.args, "diffusion_reward", "pickscore")
        if isinstance(spec, dict):
            return spec
        if isinstance(spec, str):
            text = spec.strip()
            if text.startswith("{"):
                return json.loads(text)
            if ":" in text:
                reward_dict: dict[str, float] = {}
                for pair in [p for p in text.split(",") if p]:
                    name, weight = pair.split(":")
                    reward_dict[name.strip()] = float(weight)
                return reward_dict
            return {text: 1.0}
        return {"pickscore": 1.0}

    def _get_local_reward_fn(self):
        if not hasattr(self, "_local_reward_fn") or self._local_reward_fn is None:
            device = str(torch.device(f"cuda:{torch.cuda.current_device()}"))
            self._local_reward_fn = flow_rewards.multi_score(device, self._parse_reward_spec())
        return self._local_reward_fn

    def _make_generators(self, prompts: list[str], rollout_id: int) -> list[torch.Generator]:
        base_seed = int(getattr(self.args, "rollout_seed", 0))
        seed_offset = rollout_id * 1000 + self.parallel_state.dp_rank * 100000
        generators = []
        for idx, _ in enumerate(prompts):
            seed = (base_seed + seed_offset + idx) % (2**31)
            generators.append(torch.Generator().manual_seed(seed))
        return generators

    def _run_local_rollout(self, rollout_id: int, prompts: list[str]) -> dict:
        if len(prompts) == 0:
            return {"metadata": [], "rewards": [], "raw_reward": [], "reward_ocr": []}

        num_steps = int(getattr(self.args, "diffusion_num_steps", 10))
        guidance_scale = float(getattr(self.args, "diffusion_guidance_scale", 4.5))
        noise_level = float(getattr(self.args, "diffusion_noise_level", 0.7))
        height = int(getattr(self.args, "diffusion_height", 512))
        width = int(getattr(self.args, "diffusion_width", 512))
        generators = self._make_generators(prompts, rollout_id)

        was_training = self.pipeline.transformer.training
        self.pipeline.transformer.eval()
        with torch.no_grad():
            images, all_latents, all_log_probs = pipeline_with_logprob(
                self.pipeline,
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generators,
                output_type="pil",
                noise_level=noise_level,
            )
        if was_training:
            self.pipeline.transformer.train()

        timesteps, _ = retrieve_timesteps(self.pipeline.scheduler, num_steps, torch.device("cpu"))
        sigmas = getattr(self.pipeline.scheduler, "sigmas", None)
        if sigmas is None:
            raise ValueError("Scheduler missing sigmas in local diffusion rollout.")
        sigmas = torch.as_tensor(sigmas).cpu()

        latents = torch.stack(all_latents[:-1], dim=1)
        next_latents = torch.stack(all_latents[1:], dim=1)
        log_prob_old = torch.stack(all_log_probs, dim=1)

        metadata = []
        timesteps_cpu = timesteps.cpu()
        for i in range(len(prompts)):
            metadata.append(
                {
                    "timesteps": timesteps_cpu.clone(),
                    "sigmas": sigmas.clone(),
                    "latents": latents[i].detach().cpu().clone(),
                    "next_latents": next_latents[i].detach().cpu().clone(),
                    "log_prob_old": log_prob_old[i].detach().cpu().clone(),
                }
            )

        reward_fn = self._get_local_reward_fn()
        reward_dict, _ = reward_fn(images, prompts, [{} for _ in range(len(prompts))])
        reward_dict = {k: np.asarray(v, dtype=np.float64).tolist() for k, v in reward_dict.items()}
        rewards_avg = reward_dict.get("avg", reward_dict.get("ocr", []))
        if not rewards_avg:
            rewards_avg = [0.0] * len(prompts)
        reward_ocr = reward_dict.get("ocr", [0.0] * len(prompts))
        if reward_ocr:
            zero_ratio = float(np.mean([1.0 if r == 0.0 else 0.0 for r in reward_ocr]))
            logger.info(
                "local rollout reward_ocr stats: n=%d min=%.6f max=%.6f zero_ratio=%.6f",
                len(reward_ocr),
                float(min(reward_ocr)),
                float(max(reward_ocr)),
                zero_ratio,
            )

        return {
            "metadata": metadata,
            "rewards": rewards_avg,
            "raw_reward": rewards_avg,
            "reward_ocr": reward_ocr,
        }

    def train(self, rollout_id: int, rollout_data_ref):  # type: ignore[override]
        if self.args.offload_train:
            self.wake_up()

        with timer("train"):
            if self.parallel_state.dp_rank == 0 and not getattr(self, "_env_logged", False):
                logger.info(
                    "env CUDA_VISIBLE_DEVICES=%s current_device=%s device_count=%s dp_rank=%s dp_size=%s",
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                    torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
                    torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    self.parallel_state.dp_rank,
                    self.parallel_state.dp_size,
                )
                self._env_logged = True
            # Fetch rollout data for this DP rank; metadata carries diffusion trajectories.
            rollout_data = process_rollout_data(
                self.args,
                rollout_data_ref,
                self.parallel_state.dp_rank,
                self.parallel_state.dp_size,
            )

            fallback_n = len(rollout_data.get("metadata", []))
            prompts = rollout_data.get("prompt", [""] * fallback_n)
            if "metadata" not in rollout_data:
                if self.args.colocate and getattr(self.args, "diffusion_train", False):
                    logger.info(
                        "train rank=%s using local diffusion rollout for colocate (rollout_id=%s, prompts=%s)",
                        self.parallel_state.dp_rank,
                        rollout_id,
                        len(prompts),
                    )
                    rollout_data.update(self._run_local_rollout(rollout_id, prompts))
                    dist.barrier(group=get_gloo_group())
                else:
                    raise ValueError("Diffusion training requires rollout metadata.")

            prompts = rollout_data.get("prompt", [""] * len(rollout_data["metadata"]))
            rewards = torch.tensor(
                rollout_data["rewards"], device=torch.cuda.current_device(), dtype=torch.float32
            )
            raw_rewards = rollout_data.get("raw_reward", rollout_data.get("rewards", []))
            reward_ocr = rollout_data.get("reward_ocr", None)

            batch_size = len(rollout_data["metadata"])
            micro_batch = self.args.diffusion_train_batch_size
            if micro_batch is None or micro_batch <= 0:
                micro_batch = batch_size

            total_steps = rollout_data["metadata"][0]["timesteps"].shape[0] if batch_size > 0 else 0
            fraction = float(getattr(self.args, "diffusion_timestep_fraction", 1.0))
            num_train_timesteps = max(1, int(total_steps * fraction)) if total_steps else 1

            advantages_all, per_prompt_stats = self._compute_per_prompt_advantages(
                prompts, rewards, num_train_timesteps
            )
            accum_steps = max(1, int(getattr(self.args, "diffusion_grad_accum_steps", 1)))
            accum_counter = 0
            # How many backward calls to accumulate before stepping.
            effective_accum = max(1, accum_steps * num_train_timesteps)

            log_stats = {
                "loss": [],
                "policy_loss": [],
                "kl_loss": [],
                "approx_kl": [],
                "clipfrac": [],
                "clipfrac_gt_one": [],
                "clipfrac_lt_one": [],
                "reward_avg": [],
                "reward_ori_avg": [],
                "reward_ocr": [],
                "reward_ocr_min": [],
                "reward_ocr_max": [],
                "reward_ocr_zero_ratio": [],
                "raw_reward_min": [],
                "raw_reward_max": [],
            }

            for start in range(0, batch_size, micro_batch):
                end = min(batch_size, start + micro_batch)
                batch_meta = rollout_data["metadata"][start:end]
                batch_prompts = prompts[start:end]
                batch_rewards = rewards[start:end]
                batch_advantages = advantages_all[start:end]
                batch_raw_rewards = torch.as_tensor(
                    raw_rewards[start:end], device=torch.cuda.current_device(), dtype=torch.float32
                )
                if reward_ocr is not None:
                    batch_reward_ocr = torch.as_tensor(
                        reward_ocr[start:end], device=torch.cuda.current_device(), dtype=torch.float32
                    )
                else:
                    batch_reward_ocr = torch.zeros_like(batch_rewards)

                timesteps = torch.stack([m["timesteps"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )
                sigmas = torch.stack([m["sigmas"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )
                latents = torch.stack([m["latents"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )
                next_latents = torch.stack([m["next_latents"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )
                log_prob_old = torch.stack([m["log_prob_old"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )

                # Optional: train on a fraction of timesteps.
                if fraction < 1.0:
                    timesteps = timesteps[:, :num_train_timesteps]

                # Align sigmas length with timesteps (FlowMatch expects T+1 sigmas).
                if sigmas.shape[1] >= timesteps.shape[1] + 1:
                    sigmas = sigmas[:, : timesteps.shape[1] + 1]
                elif sigmas.shape[1] == timesteps.shape[1]:
                    # Pad with last sigma to avoid out-of-bounds in sde_step_with_logprob.
                    sigmas = torch.cat([sigmas, sigmas[:, -1:]], dim=1)
                else:
                    raise ValueError(
                        f"Invalid sigmas length {sigmas.shape[1]} for timesteps {timesteps.shape[1]}"
                    )
                    latents = latents[:, :num_train_timesteps]
                    next_latents = next_latents[:, :num_train_timesteps]
                    log_prob_old = log_prob_old[:, :num_train_timesteps]

                # Broadcast per-sample reward into per-timestep advantage if needed.
                if batch_advantages.ndim == 2:
                    advantage = batch_advantages
                    if advantage.shape[1] != timesteps.shape[1]:
                        advantage = advantage[:, : timesteps.shape[1]]
                else:
                    advantage = broadcast_advantage(batch_advantages, timesteps)
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = self._encode_prompt(batch_prompts)
                prompt_embeds = prompt_embeds.detach()
                pooled_prompt_embeds = pooled_prompt_embeds.detach()

                # Prepare scheduler timesteps/sigmas on the same device (once per micro-batch).
                device = timesteps.device
                steps = timesteps.shape[1]
                rollout_ts = timesteps[0].detach()
                rollout_sigmas = sigmas[0].detach()
                if not getattr(self, "_logged_rollout_sigmas_len", False):
                    print(
                        f"[train] timesteps_len={int(steps)} sigmas_len={int(rollout_sigmas.numel())}",
                        flush=True,
                    )
                    self._logged_rollout_sigmas_len = True
                if rollout_sigmas.numel() < 2:
                    raise ValueError(
                        f"Invalid sigmas length {int(rollout_sigmas.numel())} for timesteps {int(steps)}"
                    )
                if hasattr(self.pipeline.scheduler, "set_timesteps"):
                    # Initialize internal buffers; we will override with rollout-provided values.
                    self.pipeline.scheduler.set_timesteps(steps, device=device)
                if hasattr(self.pipeline.scheduler, "timesteps"):
                    self.pipeline.scheduler.timesteps = rollout_ts.to(device)
                if hasattr(self.pipeline.scheduler, "sigmas"):
                    self.pipeline.scheduler.sigmas = rollout_sigmas.to(device)
                if not getattr(self, "_logged_rollout_scheduler_alignment", False):
                    print(
                        "[diffusion] rollout/scheduler alignment: "
                        f"timesteps head={rollout_ts[:3].tolist()} "
                        f"sigmas head={rollout_sigmas[:3].tolist()} "
                        f"sigmas_len={int(rollout_sigmas.numel())} steps={int(steps)}",
                        flush=True,
                    )
                    self._logged_rollout_scheduler_alignment = True

                advantages = torch.clamp(
                    advantage,
                    -self.args.diffusion_adv_clip_max,
                    self.args.diffusion_adv_clip_max,
                )

                # Accumulate gradients over timesteps (flow_grpo-style).
                num_timesteps = timesteps.shape[1]
                # Only clear grads when starting a new accumulation window.
                if accum_counter % effective_accum == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                for j in range(num_timesteps):
                    # Compute log_prob_new for this timestep only to avoid retaining graphs for all steps.
                    if self.args.diffusion_cfg:
                        latent_model_input = torch.cat([latents[:, j]] * 2)
                        timestep = torch.cat([timesteps[:, j]] * 2)
                        noise_pred = self.model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.args.diffusion_guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred = self.model(
                            hidden_states=latents[:, j],
                            timestep=timesteps[:, j],
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    _, log_prob_new_j, _, _ = sde_step_with_logprob(
                        self.pipeline.scheduler,
                        noise_pred.float(),
                        timesteps[:, j],
                        latents[:, j].float(),
                        prev_sample=next_latents[:, j].float(),
                        noise_level=self.args.diffusion_noise_level,
                    )

                    ratio = torch.exp(log_prob_new_j - log_prob_old[:, j])
                    unclipped = -advantages[:, j] * ratio
                    clipped = -advantages[:, j] * torch.clamp(
                        ratio, 1.0 - self.args.diffusion_clip_range, 1.0 + self.args.diffusion_clip_range
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped, clipped))
                    kl_loss = torch.zeros((), device=policy_loss.device)
                    loss = policy_loss + kl_loss

                    loss.backward()
                    accum_counter += 1

                    log_stats["loss"].append(loss.detach().float())
                    log_stats["policy_loss"].append(policy_loss.detach().float())
                    log_stats["kl_loss"].append(kl_loss.detach().float())
                    log_stats["approx_kl"].append(
                        0.5 * torch.mean((log_prob_new_j - log_prob_old[:, j]) ** 2).detach().float()
                    )
                    log_stats["clipfrac"].append(
                        torch.mean((torch.abs(ratio - 1.0) > self.args.diffusion_clip_range).float()).detach().float()
                    )
                    log_stats["clipfrac_gt_one"].append(
                        torch.mean((ratio - 1.0 > self.args.diffusion_clip_range).float()).detach().float()
                    )
                    log_stats["clipfrac_lt_one"].append(
                        torch.mean((1.0 - ratio > self.args.diffusion_clip_range).float()).detach().float()
                    )
                    log_stats["reward_avg"].append(batch_rewards.mean().detach().float())
                    log_stats["reward_ori_avg"].append(batch_raw_rewards.mean().detach().float())
                    log_stats["reward_ocr"].append(batch_reward_ocr.mean().detach().float())
                    log_stats["reward_ocr_min"].append(batch_reward_ocr.min().detach().float())
                    log_stats["reward_ocr_max"].append(batch_reward_ocr.max().detach().float())
                    log_stats["reward_ocr_zero_ratio"].append(
                        torch.mean((batch_reward_ocr == 0).float()).detach().float()
                    )
                    log_stats["raw_reward_min"].append(batch_raw_rewards.min().detach().float())
                    log_stats["raw_reward_max"].append(batch_raw_rewards.max().detach().float())

                    if accum_counter % effective_accum == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.global_step += 1
                        if self.ema is not None and self.ema_parameters is not None:
                            self.ema.step(self.ema_parameters, self.global_step)
                        self.optimizer.zero_grad(set_to_none=True)

                        reduced = {k: torch.stack(v).mean().item() for k, v in log_stats.items()}
                        reduced.update(per_prompt_stats)
                        self._log_local_step_metrics(rollout_id, reduced, step=self.global_step)
                        self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)
                        log_stats = {k: [] for k in log_stats}

            # Flush remaining gradients/logs if the rollout ended mid-accumulation.
            if accum_counter % max(1, accum_steps * timesteps.shape[1]) != 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1
                if self.ema is not None and self.ema_parameters is not None:
                    self.ema.step(self.ema_parameters, self.global_step)
                self.optimizer.zero_grad(set_to_none=True)

                if log_stats["loss"]:
                    reduced = {k: torch.stack(v).mean().item() for k, v in log_stats.items()}
                    reduced.update(per_prompt_stats)
                    self._log_local_step_metrics(rollout_id, reduced, step=self.global_step)
                    self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

        if self.args.offload_train:
            self.sleep()

        dist.barrier(group=get_gloo_group())
