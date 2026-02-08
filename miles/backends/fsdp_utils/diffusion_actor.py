from __future__ import annotations

import logging
import os
from argparse import Namespace

import torch
import torch.distributed as dist
import numpy as np
from diffusers import StableDiffusion3Pipeline

from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.ema import EMAModuleWrapper

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.data import process_rollout_data
from miles.utils.diffusion_protocol import broadcast_advantage, validate_train_inputs
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.timer import Timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.metric_utils import compute_rollout_step
from miles.utils import tracking_utils

from .actor import apply_fsdp2, move_torch_optimizer
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from ..training_utils.log_utils import gather_log_data

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
        # Diffusion rollout does not use rollout engines yet, so there is nothing to sync.
        return

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        # Diffusion GRPO does not use a critic; keep the hook for interface compliance.
        return

    def _get_parallel_config(self) -> dict:  # type: ignore[override]
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log with Flow-GRPO-aligned keys."""
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
            tracking_utils.log(self.args, reduced, step_key="global_step")
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
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

    def _compute_log_prob_new(
        self,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure scheduler tensors (timesteps/sigmas) live on the same device as latents.
        device = latents.device
        steps = timesteps.shape[1]
        if hasattr(self.pipeline.scheduler, "set_timesteps"):
            self.pipeline.scheduler.set_timesteps(steps, device=device)
        if hasattr(self.pipeline.scheduler, "timesteps") and self.pipeline.scheduler.timesteps is not None:
            self.pipeline.scheduler.timesteps = self.pipeline.scheduler.timesteps.to(device)
        if hasattr(self.pipeline.scheduler, "sigmas") and self.pipeline.scheduler.sigmas is not None:
            self.pipeline.scheduler.sigmas = self.pipeline.scheduler.sigmas.to(device)
        if hasattr(self.pipeline.scheduler, "timesteps") and self.pipeline.scheduler.timesteps is not None:
            # Use scheduler timesteps to avoid float-mismatch/index lookup failures.
            sched_ts = self.pipeline.scheduler.timesteps[:steps].to(device)
            timesteps = sched_ts.unsqueeze(0).expand(latents.shape[0], -1)

        # Recompute per-step log_prob under the current model parameters.
        log_probs = []
        for j in range(timesteps.shape[1]):
            if self.args.diffusion_cfg:
                # CFG: duplicate batch for uncond/cond pass, then combine.
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

            # Use the same SDE step as rollout to compute log_prob_new.
            _, log_prob, _, _ = sde_step_with_logprob(
                self.pipeline.scheduler,
                noise_pred.float(),
                timesteps[:, j],
                latents[:, j].float(),
                prev_sample=next_latents[:, j].float(),
                noise_level=self.args.diffusion_noise_level,
            )
            log_probs.append(log_prob)

        return torch.stack(log_probs, dim=1)

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

            if "metadata" not in rollout_data:
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

                # Prepare scheduler timesteps on the same device (once per micro-batch).
                device = timesteps.device
                steps = timesteps.shape[1]
                if hasattr(self.pipeline.scheduler, "set_timesteps"):
                    self.pipeline.scheduler.set_timesteps(steps, device=device)
                if hasattr(self.pipeline.scheduler, "timesteps") and self.pipeline.scheduler.timesteps is not None:
                    self.pipeline.scheduler.timesteps = self.pipeline.scheduler.timesteps.to(device)
                if hasattr(self.pipeline.scheduler, "sigmas") and self.pipeline.scheduler.sigmas is not None:
                    self.pipeline.scheduler.sigmas = self.pipeline.scheduler.sigmas.to(device)
                if hasattr(self.pipeline.scheduler, "timesteps") and self.pipeline.scheduler.timesteps is not None:
                    sched_ts = self.pipeline.scheduler.timesteps[:steps].to(device)
                    timesteps = sched_ts.unsqueeze(0).expand(latents.shape[0], -1)

                advantages = torch.clamp(
                    advantage,
                    -self.args.diffusion_adv_clip_max,
                    self.args.diffusion_adv_clip_max,
                )

                # Accumulate gradients over timesteps (flow_grpo-style).
                num_timesteps = timesteps.shape[1]
                self.optimizer.zero_grad(set_to_none=True)
                effective_accum = accum_steps * num_timesteps
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
                    self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

        if self.args.offload_train:
            self.sleep()

        dist.barrier(group=get_gloo_group())
