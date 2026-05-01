import logging
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils import train_metric_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.sde_log_prob import sde_step_with_logprob
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils import tracking_utils
from miles.utils.profile_utils import TrainProfiler

from .configs.train_pipeline_config import get_train_pipeline_config
import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config

from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from .diffusion_update_weight_utils import DiffusionUpdateWeightFromTensor, DiffusionUpdateWeightFromTensorLoRA

logger = logging.getLogger(__name__)

class FSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO.

    Loads only the DiT (transformer) from a diffusers pipeline, wraps it with
    FSDP, and trains with a PPO-clipped objective aligned with flow GRPO.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)
        
        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        self._master_dtype = _resolve_dtype(args.fsdp_master_dtype)
        self._forward_dtype = _resolve_dtype(args.diffusion_forward_dtype)

        with self._get_init_weight_context_manager():
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.hf_checkpoint,
                torch_dtype=self._master_dtype,
                trust_remote_code=True,
                text_encoder=None,
                vae=None,
                tokenizer=None,
            )
            model = pipeline.transformer
            self.scheduler = pipeline.scheduler
            del pipeline

        self.train_pipeline_config = get_train_pipeline_config(args.diffusion_model)

        if getattr(args, "use_lora", False):
            model = apply_lora(model, args, self.train_pipeline_config)

        model.train()

        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        model.to(torch.cuda.current_device())

        self.train_pipeline_config.preprocess_model_before_fsdp(model)

        model = apply_fsdp2(
            model,
            mesh=self.parallel_state.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
        )
        # Force a sync to ensure sharding is complete and old memory is freed.
        torch.cuda.synchronize()
        clear_memory()
        self.model = model

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # sglang-d now supports /update_weights_from_tensor (PR #20464).
        self.weight_updater = (
            DiffusionUpdateWeightFromTensorLoRA(self.args, self.model)
            if getattr(self.args, "use_lora", False)
            else DiffusionUpdateWeightFromTensor(self.args, self.model)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        return

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return

        print_memory("before offload DiT")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after sleep DiT")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.weight_updater is None:
            dist.barrier(group=get_gloo_group())
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()
        clear_memory()
    
    def _get_init_weight_context_manager(self):
        """Return a context manager for model initialization.

        Non-rank-0 ranks use accelerate's ``init_empty_weights`` (params on
        meta device, no allocation). Rank 0 uses ``torch.device("cpu")``
        (already a context manager since PyTorch 1.X — sets default device
        for tensor construction inside the block).
        """
        from accelerate import init_empty_weights

        if dist.get_rank() != 0:
            return init_empty_weights()
        return torch.device("cpu")

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log.

        ``train/lr`` and ``train/epoch`` are placed under the ``train/`` prefix
        so wandb's ``define_metric("train/*", step_metric="train/step")`` puts
        them in the same Charts row as the other train scalars, with x-axis =
        ``train/step``. Logging them at root (``"lr"``, ``"epoch"``) would put
        them in wandb's default "Charts" section against the auto-incrementing
        internal commit step instead.
        """
        if "train/lr" not in log_dict and hasattr(self, "optimizer"):
            try:
                log_dict["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
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
            reduced["train/epoch"] = float(rollout_id)
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            # wandb.define_metric("train/*", step_metric="train/step") pulls the
            # x-axis value from this key; ``train/step`` subsumes what we used
            # to also log as a separate ``global_step`` metric.
            reduced["train/step"] = float(step)
            tracking_utils.log(self.args, reduced, step_key="train/step")
            # Stdout mirror so we can spot misalignment / divergence without wandb.
            # Use scientific notation with 6 significant digits so fp32-level
            # alignment metrics (log_prob_mean_abs_diff, ratio_abs_minus_1, etc.)
            # don't get truncated to 0.0000 by a fixed-decimal format.
            print(
                f"[train step {int(step)}] rollout={rollout_id} "
                + " ".join(f"{k}={v:.6e}" for k, v in sorted(reduced.items()) if k not in ("train/epoch", "rollout/step", "train/step")),
                flush=True,
            )
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )

    def train(self, rollout_id: int, rollout_data_ref) -> None:  # type: ignore[override]
        # Always wake_up: first call moves from CPU to GPU; subsequent calls
        # are no-ops (offload_train=False) or re-load from CPU (offload_train=True).
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            # Fetch this DP rank's data directly — already split by
            # _split_train_data_by_dp in the RolloutManager.
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
        
        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Diffusion GRPO training loop, aligned with flow GRPO.

        Per optim window of M samples × T_sde timesteps, slide a
        (sample_microbatch, tstep_microbatch) tile across the (M, T_sde) grid
        and accumulate gradients. Two presets:

          sample_microbatch=M, tstep_microbatch=1, iter_order=sample_major
            equivalent to "batch by samples" (forward batch = M, loop T_sde
            times); peak activation memory ∝ M.

          sample_microbatch=1, tstep_microbatch=T_sde, iter_order=timestep_major
            equivalent to "batch by timesteps" (forward batch = T_sde, loop M
            times); peak activation memory ∝ T_sde.

        Loss scaling is uniform across plans: per-tile mean PPO loss / n_tiles
        → net gradient = mean over (M, T_sde), matching the all-timesteps
        accumulation grad scale flow_grpo and the previous miles-d sample-major
        loop produce.
        """
        device = torch.cuda.current_device()

        denoising_envs = rollout_data["denoising_env"]
        dit_trajectories = rollout_data["dit_trajectory"]
        rewards = torch.tensor(rollout_data["rewards"], device=device, dtype=torch.float32)
        rollout_log_probs_list = rollout_data["rollout_log_probs"]
        sde_step_indices_list = rollout_data.get("sde_step_indices") or [None] * len(denoising_envs)
        # Optional: per-sample post-CFG model_output recorded by sgld during the rollout
        # (only present when --diffusion-debug-mode is set). Used to log the per-tile
        # rollout-vs-train DiT-output diff to wandb.
        rollout_debug_tensors_list = rollout_data.get("rollout_debug_tensors") or [None] * len(denoising_envs)

        batch_size = len(denoising_envs)
        guidance_scale = float(getattr(self.args, "diffusion_guidance_scale", 0))
        true_cfg_scale_arg = getattr(self.args, "diffusion_true_cfg_scale", None)
        true_cfg_scale = float(true_cfg_scale_arg) if true_cfg_scale_arg is not None else None
        # Mirror sglang-d: use true_cfg_scale when set, else guidance_scale.
        cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        use_cfg = cfg_scale > 0
        clip_range = float(getattr(self.args, "diffusion_clip_range", 1e-4))
        adv_clip_max = float(getattr(self.args, "diffusion_adv_clip_max", 5.0))
        noise_level = float(getattr(self.args, "diffusion_noise_level", 0.7))
        num_timesteps = dit_trajectories[0].timesteps.shape[0]

        advantages = rewards.unsqueeze(1).expand(-1, num_timesteps).clone()
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Use rollout's exact timesteps AND derive matching sigmas.
        # Qwen-Image flow-match uses `use_dynamic_shifting=True` (mu depends on
        # image resolution), so the rollout's timesteps are post-shift.
        timesteps_ref = dit_trajectories[0].timesteps.to(device).float()
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        sigmas_ref = timesteps_ref / float(num_train_timesteps)
        sigmas_ref = torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

        self.scheduler.timesteps = timesteps_ref
        self.scheduler.sigmas = sigmas_ref
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

        train_num_timesteps = max(1, num_timesteps)
        local_batch_size = max(1, int(getattr(self.args, "local_batch_size", 1)))
        num_optim_steps_per_rollout = (batch_size + local_batch_size - 1) // local_batch_size

        sample_microbatch_arg = getattr(self.args, "micro_batch_size_sample", None)
        tstep_microbatch_arg = getattr(self.args, "micro_batch_size_tstep", None)
        iter_order = getattr(self.args, "diffusion_train_iter_order", "sample_major")
        assert iter_order in ("sample_major", "timestep_major"), iter_order

        with timer("actor_train"):
            for step_id in range(num_optim_steps_per_rollout):
                self.optimizer.zero_grad(set_to_none=True)

                traj_start = step_id * local_batch_size
                traj_end = min(batch_size, traj_start + local_batch_size)
                grids = self._build_train_grids(
                    traj_start=traj_start,
                    traj_end=traj_end,
                    dit_trajectories=dit_trajectories,
                    denoising_envs=denoising_envs,
                    rollout_log_probs_list=rollout_log_probs_list,
                    sde_step_indices_list=sde_step_indices_list,
                    rollout_debug_tensors_list=rollout_debug_tensors_list,
                    advantages=advantages,
                    train_num_timesteps=train_num_timesteps,
                    use_cfg=use_cfg,
                    device=device,
                )

                local_batch_size = grids["local_batch_size"]
                sde_window_size = grids["sde_window_size"]
                sample_microbatch = min(
                    sample_microbatch_arg if sample_microbatch_arg is not None else local_batch_size,
                    local_batch_size,
                )
                tstep_microbatch = min(
                    tstep_microbatch_arg if tstep_microbatch_arg is not None else 1,
                    sde_window_size,
                )
                sample_microbatch = max(1, sample_microbatch)
                tstep_microbatch = max(1, tstep_microbatch)

                log_stats = self._run_optim_window(
                    grids=grids,
                    sample_microbatch=sample_microbatch,
                    tstep_microbatch=tstep_microbatch,
                    iter_order=iter_order,
                    use_cfg=use_cfg,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    clip_range=clip_range,
                    noise_level=noise_level,
                    num_train_timesteps=num_train_timesteps,
                )

                self.prof.step(rollout_id=rollout_id)
                if not getattr(self.args, "debug_skip_optimizer_step", False):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    log_stats["grad_norm"].append(grad_norm.detach())
                    self.optimizer.step()
                    self.lr_scheduler.step()
                else:
                    # Keep weights frozen so noise_pred / log_prob alignment
                    # checks remain interpretable across iterations.
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Prefix with "train/" so wandb groups these under the Train
                # panel and picks up define_metric("train/*",
                # step_metric="train/step") — otherwise they fall into the
                # default "Charts" section and plot against wandb's
                # auto-incrementing internal step.
                reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
                self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

    def _build_train_grids(
        self,
        *,
        traj_start: int,
        traj_end: int,
        dit_trajectories: list,
        denoising_envs: list,
        rollout_log_probs_list: list,
        sde_step_indices_list: list,
        rollout_debug_tensors_list: list,
        advantages: torch.Tensor,
        train_num_timesteps: int,
        use_cfg: bool,
        device: torch.device,
    ) -> dict:
        """Build per-window (sample × tstep, ...) grids ready for tile slicing.
        Per-sample SDE windows must have equal length so they stack cleanly."""
        train_pipeline_config = self.train_pipeline_config

        latents_list, next_latents_list, timesteps_list = [], [], []
        log_prob_old_list, advantage_list = [], []
        positive_cond_kwargs_list, negative_cond_kwargs_list = [], []
        rollout_model_outputs_list: list[torch.Tensor] = []
        sde_window_size: int | None = None

        for traj_idx in range(traj_start, traj_end):
            latents, next_latents, timesteps = train_pipeline_config.prepare_trajectory(
                dit_trajectories[traj_idx], device
            )
            denoising_env = denoising_envs[traj_idx]
            positive_cond_kwargs_list.append(
                train_pipeline_config.prepare_cond_kwargs(denoising_env.pos_cond_kwargs, device)
            )
            if use_cfg:
                negative_cond_kwargs_list.append(
                    train_pipeline_config.prepare_cond_kwargs(denoising_env.neg_cond_kwargs, device)
                )
            log_prob_old = rollout_log_probs_list[traj_idx].to(device, dtype=torch.float32)
            advantage = advantages[traj_idx]

            sde_step_indices = sde_step_indices_list[traj_idx]
            # Slice rollout-recorded model_output to the SDE window so it lines up with
            # latents/next_latents/timesteps. Only present when --diffusion-debug-mode.
            rollout_dbg = rollout_debug_tensors_list[traj_idx] if rollout_debug_tensors_list else None
            rollout_model_output = (
                rollout_dbg.rollout_model_outputs.to(device, dtype=torch.float32)
                if rollout_dbg is not None and rollout_dbg.rollout_model_outputs is not None
                else None
            )
            if sde_step_indices is not None:
                sde_indices_tensor = torch.as_tensor(sde_step_indices, device=device, dtype=torch.long)
                latents = latents[sde_indices_tensor]
                next_latents = next_latents[sde_indices_tensor]
                timesteps = timesteps[sde_indices_tensor]
                log_prob_old = log_prob_old[sde_indices_tensor]
                advantage = advantage[: sde_indices_tensor.numel()]
                if rollout_model_output is not None:
                    rollout_model_output = rollout_model_output[sde_indices_tensor]
                current_window_size = int(sde_indices_tensor.numel())
            else:
                current_window_size = train_num_timesteps

            if sde_window_size is None:
                sde_window_size = current_window_size
            else:
                assert current_window_size == sde_window_size, (
                    f"for now per-sample SDE window length must match across microbatch "
                    f"(got {sde_window_size} and {current_window_size})"
                )
            latents_list.append(latents)
            next_latents_list.append(next_latents)
            timesteps_list.append(timesteps)
            log_prob_old_list.append(log_prob_old)
            advantage_list.append(advantage)
            if rollout_model_output is not None:
                rollout_model_outputs_list.append(rollout_model_output)

        latents_window = torch.stack(latents_list, dim=0)
        next_latents_window = torch.stack(next_latents_list, dim=0)
        timesteps_window = torch.stack(timesteps_list, dim=0)
        log_prob_old_window = torch.stack(log_prob_old_list, dim=0)
        advantage_window = torch.stack(advantage_list, dim=0)
        rollout_model_outputs_window = (
            torch.stack(rollout_model_outputs_list, dim=0)
            if rollout_model_outputs_list and len(rollout_model_outputs_list) == (traj_end - traj_start)
            else None
        )

        # Skip the (possibly NotImplementedError-raising) collate when no tile
        # will ever have sample > 1: a model that only does timestep-only tiling
        # then doesn't need to override collate_cond_for_sample_batch.
        local_batch_size = int(traj_end - traj_start)
        sample_microbatch_arg = getattr(self.args, "micro_batch_size_sample", None)
        needs_multi_sample_tile = (
            sample_microbatch_arg is None or sample_microbatch_arg > 1
        ) and local_batch_size > 1

        if not needs_multi_sample_tile:
            cond_collated = None
        elif use_cfg:
            cond_collated = train_pipeline_config.collate_cond_for_sample_batch(
                positive_cond_kwargs_list + negative_cond_kwargs_list, device
            )
        else:
            cond_collated = train_pipeline_config.collate_cond_for_sample_batch(
                positive_cond_kwargs_list, device
            )

        return {
            "latents": latents_window,
            "next_latents": next_latents_window,
            "timesteps": timesteps_window,
            "log_prob_old": log_prob_old_window,
            "advantage": advantage_window,
            "cond": cond_collated,
            "per_sample_pos_cond": positive_cond_kwargs_list,
            "per_sample_neg_cond": negative_cond_kwargs_list,
            "local_batch_size": int(traj_end - traj_start),
            "sde_window_size": int(sde_window_size or 0),
            "rollout_model_outputs": rollout_model_outputs_window,
        }

    def _run_optim_window(
        self,
        *,
        grids: dict,
        sample_microbatch: int,
        tstep_microbatch: int,
        iter_order: str,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        clip_range: float,
        noise_level: float,
        num_train_timesteps: int,
    ) -> dict[str, list[torch.Tensor]]:
        """Iterate tiles across the window grid; one DiT forward + PPO loss +
        backward per tile. All tiles share `(loss / num_tiles).backward()` so
        net gradient = mean over (local_batch_size × sde_window_size) cells."""
        device = grids["latents"].device
        local_batch_size = grids["local_batch_size"]
        sde_window_size = grids["sde_window_size"]
        sample_chunks = _chunked_indices(local_batch_size, sample_microbatch, device)
        tstep_chunks = _chunked_indices(sde_window_size, tstep_microbatch, device)
        num_tiles = len(sample_chunks) * len(tstep_chunks)

        if iter_order == "sample_major":
            outer_chunks, inner_chunks = tstep_chunks, sample_chunks
        else:
            outer_chunks, inner_chunks = sample_chunks, tstep_chunks

        log_stats: dict[str, list[torch.Tensor]] = defaultdict(list)
        skip_optimizer_step = bool(getattr(self.args, "debug_skip_optimizer_step", False))

        for outer in outer_chunks:
            for inner in inner_chunks:
                if iter_order == "sample_major":
                    sample_indices, tstep_indices = inner, outer
                else:
                    sample_indices, tstep_indices = outer, inner
                loss = self._forward_tile(
                    sample_indices=sample_indices,
                    tstep_indices=tstep_indices,
                    grids=grids,
                    use_cfg=use_cfg,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    clip_range=clip_range,
                    noise_level=noise_level,
                    num_train_timesteps=num_train_timesteps,
                    log_stats=log_stats,
                )
                if not skip_optimizer_step:
                    (loss / num_tiles).backward()

        return log_stats

    def _forward_tile(
        self,
        *,
        sample_indices: torch.Tensor,
        tstep_indices: torch.Tensor,
        grids: dict,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        clip_range: float,
        noise_level: float,
        num_train_timesteps: int,
        log_stats: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        """One DiT forward over a tile of (tile_sample × tile_tstep) cells
        flattened to batch = tile_sample * tile_tstep."""
        device = grids["latents"].device
        forward_dtype = self._forward_dtype
        train_pipeline_config = self.train_pipeline_config
        tile_sample_count = int(sample_indices.numel())
        tile_tstep_count = int(tstep_indices.numel())
        local_batch_size = grids["local_batch_size"]

        latents_tile = grids["latents"][sample_indices][:, tstep_indices]
        next_latents_tile = grids["next_latents"][sample_indices][:, tstep_indices]
        timesteps_tile = grids["timesteps"][sample_indices][:, tstep_indices]
        log_prob_old_tile = grids["log_prob_old"][sample_indices][:, tstep_indices]
        advantage_tile = grids["advantage"][sample_indices][:, tstep_indices]

        latents_flat = latents_tile.reshape(
            tile_sample_count * tile_tstep_count, *latents_tile.shape[2:]
        )
        timesteps_flat = timesteps_tile.reshape(tile_sample_count * tile_tstep_count)

        # sgl-d's Qwen DiT divides timestep by num_train_timesteps inside
        # forward; diffusers' does not — pre-scale to match.
        timesteps_normalized = timesteps_flat / float(num_train_timesteps)

        # tile_sample==1: skip window-collated cond and use the per-sample
        # un-padded cond + expand along tstep.
        if tile_sample_count == 1:
            s = sample_indices.item()
            pos_cond_tile = train_pipeline_config.expand_cond_for_timestep_batch(
                grids["per_sample_pos_cond"][s], tile_tstep_count
            )
            neg_cond_tile = (
                train_pipeline_config.expand_cond_for_timestep_batch(
                    grids["per_sample_neg_cond"][s], tile_tstep_count
                )
                if use_cfg
                else None
            )
        else:
            assert grids["cond"] is not None, (
                "tile_sample_count > 1 but window cond_collated was skipped — "
                "the skip predicate in _build_train_grids and the resolve "
                "logic in _run_optim_window have diverged"
            )
            pos_cond_tile, neg_cond_tile = _tile_collated_cond(
                grids["cond"],
                sample_indices=sample_indices,
                tile_tstep_count=tile_tstep_count,
                local_batch_size=local_batch_size,
                use_cfg=use_cfg,
            )

        pos_cond_tile = _cast_cond_to_dtype(pos_cond_tile, forward_dtype)
        if neg_cond_tile is not None:
            neg_cond_tile = _cast_cond_to_dtype(neg_cond_tile, forward_dtype)

        # Cast inputs explicitly: FSDP MixedPrecisionPolicy casts params but
        # leaves fp32 inputs, which would run first matmul at higher precision
        # than rollout → systematic noise_pred drift.
        latents_input = latents_flat.to(forward_dtype)
        timesteps_input = timesteps_normalized.to(forward_dtype)

        def _forward(cond: dict) -> torch.Tensor:
            return self.model(
                hidden_states=latents_input,
                timestep=timesteps_input,
                return_dict=False,
                **cond,
            )[0]

        cfg_batching = bool(getattr(self.args, "fsdp_cfg_batching", False))

        if not use_cfg:
            noise_pred_flat = _forward(pos_cond_tile)
        elif cfg_batching:
            joint_cond = _pack_cond_for_joint_cfg(pos_cond_tile, neg_cond_tile)
            joint_out = self.model(
                hidden_states=torch.cat([latents_input, latents_input], dim=0),
                timestep=torch.cat([timesteps_input, timesteps_input], dim=0),
                return_dict=False,
                **joint_cond,
            )[0]
            noise_pred_pos, noise_pred_neg = joint_out.chunk(2, dim=0)
            noise_pred_flat = train_pipeline_config.cfg_combine(
                noise_pred_pos, noise_pred_neg, guidance_scale, true_cfg_scale=true_cfg_scale,
            )
        else:
            noise_pred_pos = _forward(pos_cond_tile)
            noise_pred_neg = _forward(neg_cond_tile)
            noise_pred_flat = train_pipeline_config.cfg_combine(
                noise_pred_pos, noise_pred_neg, guidance_scale, true_cfg_scale=true_cfg_scale,
            )

        _, log_prob_new_flat, _, _ = sde_step_with_logprob(
            self.scheduler,
            noise_pred_flat.float(),
            timesteps_flat,
            latents_flat.float(),
            prev_sample=next_latents_tile.reshape(
                tile_sample_count * tile_tstep_count, *next_latents_tile.shape[2:]
            ).float(),
            noise_level=noise_level,
        )
        log_prob_new = log_prob_new_flat.reshape(tile_sample_count, tile_tstep_count)
        ratio = torch.exp(log_prob_new - log_prob_old_tile)
        unclipped = -advantage_tile * ratio
        clipped = -advantage_tile * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        per_cell_loss = torch.maximum(unclipped, clipped)
        loss = per_cell_loss.mean()

        with torch.no_grad():
            log_stats["loss"].append(loss.detach())
            log_stats["loss_abs_mean"].append(per_cell_loss.abs().mean().detach())
            log_stats["adv_abs_mean"].append(advantage_tile.abs().mean().detach())
            log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
            log_stats["approx_kl"].append(
                0.5 * torch.mean((log_prob_new - log_prob_old_tile) ** 2).detach()
            )
            log_stats["clipfrac"].append(
                torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
            )
            log_stats["log_prob_new_idx_0"].append(log_prob_new[0, 0].detach())
            log_stats["log_prob_old_idx_0"].append(log_prob_old_tile[0, 0].detach())
            log_stats["log_prob_mean_abs_diff"].append(
                torch.mean(torch.abs(log_prob_new - log_prob_old_tile)).detach()
            )
            # Rollout-vs-train DiT-output diff: only when rollout dumped per-step
            # post-CFG model_outputs (--diffusion-debug-mode). Compares trainer's
            # recomputed noise_pred_flat against sgld's recorded model_output for
            # the same (sample, tstep) cell.
            rollout_mo_window = grids.get("rollout_model_outputs")
            if rollout_mo_window is not None:
                rollout_mo_tile = rollout_mo_window[sample_indices][:, tstep_indices]
                rollout_mo_flat = rollout_mo_tile.reshape(
                    tile_sample_count * tile_tstep_count, *rollout_mo_tile.shape[2:]
                )
                diff = (noise_pred_flat.float() - rollout_mo_flat.float()).abs()
                ref_max = rollout_mo_flat.float().abs().max() + 1e-30
                log_stats["model_output_max_abs_diff"].append(diff.max().detach())
                log_stats["model_output_mean_abs_diff"].append(diff.mean().detach())
                log_stats["model_output_rel_max"].append((diff.max() / ref_max).detach())

        return loss


def _chunked_indices(total: int, chunk_size: int, device: torch.device) -> list[torch.Tensor]:
    """Split range(total) into 1-D LongTensor chunks of size <= chunk_size."""
    if total <= 0:
        return []
    chunk_size = max(1, chunk_size)
    return [
        torch.arange(start, min(start + chunk_size, total), device=device, dtype=torch.long)
        for start in range(0, total, chunk_size)
    ]


def _tile_collated_cond(
    cond: dict,
    *,
    sample_indices: torch.Tensor,
    tile_tstep_count: int,
    local_batch_size: int,
    use_cfg: bool,
) -> tuple[dict, dict | None]:
    """Pick `sample_indices` rows from a window-collated cond and tile each
    row `tile_tstep_count` times along the batch dim. Output dicts have
    batch = sample_indices.numel() * tile_tstep_count.

    For CFG the input packs [pos_M | neg_M] along batch=2*local_batch_size;
    pos and neg halves are extracted separately, the latter via offset
    `+ local_batch_size`. Returns (pos, None) when use_cfg is False."""
    def _tile_value(value, rows: torch.Tensor):
        if isinstance(value, torch.Tensor):
            return value.index_select(0, rows).repeat_interleave(tile_tstep_count, dim=0)
        if isinstance(value, list):
            picked = [value[int(r)] for r in rows.tolist()]
            return [item for item in picked for _ in range(tile_tstep_count)]
        return value

    pos = {k: _tile_value(v, sample_indices) for k, v in cond.items()}
    if not use_cfg:
        return pos, None
    neg_indices = sample_indices + local_batch_size
    neg = {k: _tile_value(v, neg_indices) for k, v in cond.items()}
    return pos, neg


def _pack_cond_for_joint_cfg(pos: dict, neg: dict) -> dict:
    """Pack pos and neg per-tile cond dicts into a single [pos | neg] dict
    along the batch dim, for joint CFG forward."""
    out: dict = {}
    for key, value in pos.items():
        if isinstance(value, torch.Tensor):
            out[key] = torch.cat([value, neg[key]], dim=0)
        elif isinstance(value, list):
            out[key] = value + neg[key]
        else:
            out[key] = value
    return out


def _cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    """Cast floating-point tensors to the model's compute dtype; leave bool
    masks / int / list / scalar values untouched. The bool
    encoder_hidden_states_mask must NOT be cast — diffusers reads it as a
    bool/int mask.
    """
    out: dict = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]

def apply_lora(model: torch.nn.Module, args: Namespace, train_pipeline_config) -> None:
    """Apply PEFT LoRA to the model.

    Args:
        model: The model to apply LoRA to.
        args: Arguments containing LoRA settings.
        train_pipeline_config: The train pipeline config.
    """
    from peft import LoraConfig, get_peft_model

    targets = getattr(args, "lora_target_modules", None) or train_pipeline_config.lora_target_modules
    init_lora_weight = getattr(args, "diffusion_init_lora_weight", "gaussian")
    # "kaiming-uniform" is the default for PEFT's LoraConfig (passed as `True`).
    # Other init methods: "gaussian", "olora", "pissa", "pissa_niter_N", "loftq", ...
    if init_lora_weight == "kaiming-uniform":
        init_lora_weight = True
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        init_lora_weights=init_lora_weight,
    ))
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model

def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    rollout_dtype_name = getattr(args, "diffusion_forward_dtype", None) if args is not None else None
    param_dtype = _resolve_dtype(rollout_dtype_name)
    reduce_dtype = torch.float32
    logger.info(f"FSDP: wrapping {len(modules)} modules of type {layer_cls_to_wrap}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
