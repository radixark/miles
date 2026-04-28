import logging
import os
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.sde_log_prob import sde_step_with_logprob
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils import tracking_utils

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

        self._master_dtype = _resolve_dtype(args.fsdp_master_dtype)
        self._rollout_dtype = _resolve_dtype(args.diffusion_rollout_dtype)
        pipeline = DiffusionPipeline.from_pretrained(
            args.diffusion_model,
            torch_dtype=self._master_dtype,
            trust_remote_code=True,
        )
        model = pipeline.transformer
        self.scheduler = pipeline.scheduler
        del pipeline
        clear_memory()

        self.train_pipeline_config = get_train_pipeline_config(args.diffusion_model)

        if getattr(args, "use_lora", False):
            from peft import LoraConfig, get_peft_model

            targets = getattr(args, "lora_target_modules", None) or self.train_pipeline_config.lora_target_modules
            init = getattr(args, "diffusion_init_lora_weight", "gaussian")
            # "kaiming-uniform" is the common name for PEFT's default (passed as `True`).
            # Everything else is a string PEFT already recognises ("gaussian", "olora",
            # "pissa", "pissa_niter_N", "loftq", ...).
            if init == "kaiming-uniform":
                init = True
            model = get_peft_model(model, LoraConfig(
                r=getattr(args, "lora_rank", 64),
                lora_alpha=getattr(args, "lora_alpha", 64),
                target_modules=targets,
                init_lora_weights=init,
            ))
            if dist.get_rank() == 0:
                model.print_trainable_parameters()

        model.train()

        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Move to GPU first, then FSDP shard — FSDP2 shards at init time
        # and converts params to DTensor. Must be on GPU for NCCL collectives.
        model.to(torch.cuda.current_device())
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
        # Allow bypass for alignment debugging: both training and rollout load
        # from the same HF checkpoint, so in theory no sync is needed until
        # training actually updates weights.
        disable_sync = bool(getattr(self.args, "debug_disable_weight_sync", False))
        if self.args.debug_train_only or disable_sync:
            self.weight_updater = None
            if disable_sync and dist.get_rank() == 0:
                logger.info("[debug] weight sync disabled via --debug-disable-weight-sync")
        elif getattr(self.args, "use_lora", False):
            self.weight_updater = DiffusionUpdateWeightFromTensorLoRA(self.args, self.model)
        else:
            self.weight_updater = DiffusionUpdateWeightFromTensor(self.args, self.model)

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        return

    @timer
    def sleep(self) -> None:
        if self.args.offload_train:
            self.model.cpu()
            move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after sleep DiT")

    @timer
    def wake_up(self) -> None:
        if self.args.offload_train:
            self.model.cuda()
            move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

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

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log."""
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
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            # wandb.define_metric("train/*", step_metric="train/step") pulls the
            # x-axis value from this key; ``train/step`` subsumes what we used
            # to also log as a separate ``global_step`` metric.
            reduced["train/step"] = float(step)
            tracking_utils.log(self.args, reduced, step_key="train/step")
            # Stdout mirror so we can spot misalignment / divergence without wandb.
            print(
                f"[train step {int(step)}] rollout={rollout_id} "
                + " ".join(f"{k}={v:.4f}" for k, v in sorted(reduced.items()) if k not in ("epoch", "rollout/step", "train/step")),
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
        self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            # Fetch this DP rank's data directly — already split by
            # _split_train_data_by_dp in the RolloutManager.
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.offload_train:
            self.sleep()

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Diffusion GRPO training loop, aligned with flow GRPO.

        Flow GRPO reference: sglang/3rdparty/flow_grpo/scripts/train_sd3.py:869-944
        Per timestep j:
          1. noise_pred = DiT(latents[j], timesteps[j], encoder_hidden_states)
          2. _, log_prob_new, _, _ = sde_step_with_logprob(scheduler, noise_pred, ...)
          3. ratio = exp(log_prob_new - log_prob_old[j])
          4. loss = max(-adv[j] * ratio, -adv[j] * clamp(ratio))
          5. loss.backward()
        """
        device = torch.cuda.current_device()

        denoising_envs = rollout_data["denoising_env"]
        dit_trajectories = rollout_data["dit_trajectory"]
        rewards = torch.tensor(rollout_data["rewards"], device=device, dtype=torch.float32)
        rollout_log_probs_list = rollout_data["rollout_log_probs"]
        rollout_debug_list = rollout_data.get("rollout_debug_tensors") or [None] * len(denoising_envs)
        # Per-sample sde-window step indices (from step_strategy_hub.sde_window).
        # When set, the trajectory / log_probs come back full-length and we
        # slice to this subset — mirroring flow_grpo, which only computes
        # log_prob / loss on in-window steps.
        sde_step_indices_list = rollout_data.get("sde_step_indices") or [None] * len(denoising_envs)

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

        # Broadcast scalar reward to per-timestep advantage.
        # rewards shape: (batch_size,) -> (batch_size, num_timesteps)
        advantages = rewards.unsqueeze(1).expand(-1, num_timesteps).clone()
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Use rollout's exact timesteps AND derive matching sigmas.
        # Qwen-Image flow-match uses `use_dynamic_shifting=True` (mu depends on
        # image resolution), so the rollout's timesteps are post-shift. We can't
        # just call `set_timesteps(N)` with `use_dynamic_shifting=False`, because
        # that would produce unshifted sigmas while `scheduler.sigmas[step_index]`
        # would then disagree with the rollout's std_dev_t / prev_sample_mean.
        # Flow matching invariant (no invert_sigmas): sigma_i = t_i / num_train_timesteps.
        timesteps_ref = dit_trajectories[0].timesteps.to(device).float()
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        sigmas_ref = timesteps_ref / float(num_train_timesteps)
        # Append terminal sigma=0 to match FlowMatchEulerDiscreteScheduler.set_timesteps().
        sigmas_ref = torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

        self.scheduler.timesteps = timesteps_ref
        self.scheduler.sigmas = sigmas_ref
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

        train_num_timesteps = max(1, num_timesteps)

        trajectories_per_step = max(1, int(getattr(self.args, "diffusion_gradient_accumulation_steps", 1)))
        timestep_batch = int(getattr(self.args, "diffusion_timestep_batch", 1))
        num_steps_per_rollout = (batch_size + trajectories_per_step - 1) // trajectories_per_step

        for step_id in range(num_steps_per_rollout):
            self.optimizer.zero_grad(set_to_none=True)
            log_stats = defaultdict(list)

            traj_start = step_id * trajectories_per_step
            traj_end = min(batch_size, traj_start + trajectories_per_step)

            # Inner loop: accumulate gradients over multiple trajectories.
            for i in range(traj_start, traj_end):
                tpc = self.train_pipeline_config
                latents, next_latents, timesteps_i = tpc.prepare_trajectory(dit_trajectories[i], device)
                env = denoising_envs[i]
                pos_cond = tpc.prepare_cond_kwargs(env.pos_cond_kwargs, device)
                neg_cond = tpc.prepare_cond_kwargs(env.neg_cond_kwargs, device) if use_cfg else None
                log_prob_old_i = rollout_log_probs_list[i].to(device, dtype=torch.float32)
                advantage_i = advantages[i]
                reward_i = rewards[i]

                # Restrict to the flow_grpo-style SDE window (if any). Trajectory and
                # log_probs come back full-length so scheduler.timesteps/sigmas stay
                # correct for any j via `index_for_timestep`; we just index in.
                sde_idx = sde_step_indices_list[i]
                if sde_idx is not None:
                    idx = torch.as_tensor(sde_idx, device=device, dtype=torch.long)
                    latents = latents[idx]
                    next_latents = next_latents[idx]
                    timesteps_i = timesteps_i[idx]
                    log_prob_old_i = log_prob_old_i[idx]
                    advantage_i = advantage_i[: idx.numel()]
                    sample_train_steps = int(idx.numel())
                else:
                    sample_train_steps = train_num_timesteps

                # Batch multiple timesteps for GPU utilization.
                for t_start in range(0, sample_train_steps, timestep_batch):
                    t_end = min(sample_train_steps, t_start + timestep_batch)
                    tb = t_end - t_start
                    lat_chunk = latents[t_start:t_end]
                    ts_chunk = timesteps_i[t_start:t_end]

                    # sgl-d's Qwen DiT divides timestep by num_train_timesteps
                    # inside forward; diffusers' Qwen DiT does NOT — so we must
                    # pre-scale here to land at the same time-embedding input.
                    # Ref: sglang/.../models/dits/qwen_image.py (`timestep = timestep / 1000`).
                    ts_chunk_for_model = ts_chunk / float(num_train_timesteps)

                    pos_batch = tpc.expand_cond_for_timestep_batch(pos_cond, tb)
                    if t_start == 0 and i == traj_start:
                        alloc = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        print(f"[DEBUG] before first forward: allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)
                        # Dump DiT input summary so we can see if the forward
                        # gets the same latent / timestep / cond as rollout.
                        print(
                            f"[input align traj=0 chunk_t={t_start}:{t_end}] "
                            f"lat shape={tuple(lat_chunk.shape)} norm={lat_chunk.float().norm().item():.3f} "
                            f"ts={ts_chunk.tolist()} "
                            f"enc_hid shape={tuple(pos_batch.get('encoder_hidden_states', torch.empty(0)).shape)} "
                            f"enc_hid norm={pos_batch['encoder_hidden_states'].float().norm().item():.3f} "
                            f"txt_seq_lens={pos_batch.get('txt_seq_lens')} "
                            f"img_shapes={pos_batch.get('img_shapes')}",
                            flush=True,
                        )

                    # Match rollout's compute dtype exactly. Rollout runs under
                    # torch.autocast("cuda", <dtype>) so all inputs enter the DiT
                    # as that dtype. Without explicit cast here, FSDP MixedPrecision
                    # only casts params but leaves fp32 inputs → first matmul runs
                    # at higher precision than rollout → systematic noise_pred drift.
                    # When rollout_dtype=fp32, this is a no-op (inputs already fp32).
                    _dt = self._rollout_dtype
                    _cast = lambda d: {k: v.to(_dt) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                    noise_pred_pos = self.model(
                        hidden_states=lat_chunk.to(_dt),
                        timestep=ts_chunk_for_model.to(_dt),
                        return_dict=False,
                        **_cast(pos_batch),
                    )[0]
                    if t_start == 0 and i == traj_start:
                        print(
                            f"[noise_pred_pos traj=0 chunk_t={t_start}:{t_end}] "
                            f"shape={tuple(noise_pred_pos.shape)} "
                            f"norm={noise_pred_pos.float().norm().item():.3f} "
                            f"mean={noise_pred_pos.float().mean().item():.4f} "
                            f"std={noise_pred_pos.float().std().item():.4f}",
                            flush=True,
                        )

                    if t_start == 0 and i == traj_start:
                        alloc = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        print(f"[DEBUG] after first forward: allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)

                    if use_cfg and neg_cond is not None:
                        neg_batch = tpc.expand_cond_for_timestep_batch(neg_cond, tb)
                        noise_pred_neg = self.model(
                            hidden_states=lat_chunk.to(_dt),
                            timestep=ts_chunk_for_model.to(_dt),
                            return_dict=False,
                            **_cast(neg_batch),
                        )[0]
                        noise_pred = tpc.cfg_combine(
                            noise_pred_pos,
                            noise_pred_neg,
                            guidance_scale,
                            true_cfg_scale=true_cfg_scale,
                        )
                    else:
                        noise_pred = noise_pred_pos

                    # DEBUG: compare training's noise_pred with rollout's stored
                    # model_output for the first trajectory / first chunk. If the
                    # two DiT implementations match numerically, this diff should
                    # be ~1e-3 (or smaller). Bigger diffs pinpoint a forward-input
                    # mismatch (latent, timestep, cond_kwargs) rather than SDE math.
                    if i == traj_start and step_id == 0 and rollout_debug_list[i] is not None:
                        rdt = rollout_debug_list[i]
                        if rdt is not None and rdt.rollout_model_outputs is not None:
                            ro_mo = rdt.rollout_model_outputs.to(device).float()
                            if t_start == 0:
                                print(
                                    f"[rollout_model_outputs] full shape={tuple(ro_mo.shape)} "
                                    f"norm_overall={ro_mo.norm().item():.3f}",
                                    flush=True,
                                )
                            # Expected shape: (T, C, H, W) or (T, N, C) per trajectory.
                            # Slice to the current chunk's timesteps.
                            ro_chunk = ro_mo[t_start:t_end].to(noise_pred.dtype)
                            if ro_chunk.shape == noise_pred.shape:
                                diff = (noise_pred - ro_chunk).abs()
                                print(
                                    f"[noise_pred align traj=0 chunk_t={t_start}:{t_end}] "
                                    f"mean={diff.mean().item():.4e} max={diff.max().item():.4e} "
                                    f"train_norm={noise_pred.norm().item():.3f} "
                                    f"rollout_norm={ro_chunk.norm().item():.3f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[noise_pred align] shape mismatch: "
                                    f"train={tuple(noise_pred.shape)} vs rollout_chunk={tuple(ro_chunk.shape)}",
                                    flush=True,
                                )

                    _, log_prob_new, _, _ = sde_step_with_logprob(
                        self.scheduler,
                        noise_pred.float(),
                        timesteps_i[t_start:t_end],
                        latents[t_start:t_end].float(),
                        prev_sample=next_latents[t_start:t_end].float(),
                        noise_level=noise_level,
                    )

                    adv_chunk = advantage_i[t_start:t_end]
                    old_chunk = log_prob_old_i[t_start:t_end]

                    ratio = torch.exp(log_prob_new - old_chunk)
                    unclipped = -adv_chunk * ratio
                    clipped = -adv_chunk * torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    loss = torch.mean(torch.maximum(unclipped, clipped))
                    if not getattr(self.args, "debug_skip_optimizer_step", False):
                        loss.backward()

                    with torch.no_grad():
                        per_elem = torch.maximum(unclipped, clipped)
                        log_stats["loss"].append(loss.detach())
                        # Diagnostic: abs-mean shows raw loss magnitude before sign cancellation
                        log_stats["loss_abs_mean"].append(per_elem.abs().mean().detach())
                        log_stats["adv_abs_mean"].append(adv_chunk.abs().mean().detach())
                        log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
                        log_stats["approx_kl"].append(
                            0.5 * torch.mean((log_prob_new - old_chunk) ** 2).detach()
                        )
                        log_stats["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
                        )
                        log_stats["log_prob_new_idx_0"].append(log_prob_new[0].detach())
                        log_stats["log_prob_old_idx_0"].append(old_chunk[0].detach())
                        log_stats["log_prob_mean_abs_diff"].append(torch.mean(torch.abs(log_prob_new - old_chunk)).detach())

            # One optimizer step per step_id.
            if not getattr(self.args, "debug_skip_optimizer_step", False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                self.lr_scheduler.step()
            else:
                # Keep weights frozen so noise_pred / log_prob alignment checks
                # remain interpretable across iterations.
                self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            # Prefix with "train/" so wandb groups these under the Train panel
            # and picks up define_metric("train/*", step_metric="train/step") —
            # otherwise they fall into the default "Charts" section and plot
            # against wandb's auto-incrementing internal step.
            reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
            self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)


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
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16  # default


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (--diffusion-rollout-dtype, --fsdp-master-dtype, --fp16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    rollout_dtype_name = getattr(args, "diffusion_rollout_dtype", None) if args is not None else None
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
