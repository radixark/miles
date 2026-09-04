"""torchtitan's Trainer, adopted whole as the training black box.

``Trainer.__init__`` owns the entire build and ``forward_backward_step`` hides
the PP/non-PP split (pytorch/torchtitan#3856). ``TitanTrainer`` adds nothing to
construction; it translates the shared RL loop's step-runner protocol onto that
machinery. The config tree lives in ``config``, the RL loss in ``loss``, the
dataloader and checkpoint stand-ins in ``components``, and the HF weight stream
in ``weight_bridge``; the package ``__init__`` installs ``compat`` before any of
them imports torchtitan.
"""

import functools
import inspect
import json
import os
from collections.abc import Callable

import torch
import torch.distributed as dist
from torchtitan.distributed import utils as titan_dist_utils
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.training_utils.torch_native_loop import StepMetrics

_FLEX_BLOCK = 128


class TitanTrainer(Trainer):
    """torchtitan's Trainer with the RL step surface bolted on."""

    @functools.cached_property
    def _family_forward_kwargs(self) -> dict:
        """Static per-family forward kwargs, e.g. qwen3_5's ``special_tokens``."""
        kwargs = {}
        if "special_tokens" in inspect.signature(self.model_parts[0].forward).parameters:
            hf_cfg = json.load(open(os.path.join(self.config.hf_assets_path, "config.json")))
            kwargs["special_tokens"] = {
                "image_id": hf_cfg.get("image_token_id", -1),
                "video_id": hf_cfg.get("video_token_id", -2),
            }
        return kwargs

    def padded_length(self, n_tokens: int) -> int:
        """PP stages exchange one fixed shape; CP needs a multiple of cp * flex block."""
        if self.parallel_dims.pp_enabled:
            target = self.config.training.seq_len
            if n_tokens > target:
                raise ValueError(
                    f"packed microbatch of {n_tokens} tokens exceeds --titan-seq-len "
                    f"{target}, which is the fixed shape PP stages exchange"
                )
            return target
        if self.parallel_dims.cp_enabled:
            align = self.parallel_dims.cp * _FLEX_BLOCK
            return n_tokens + (align - n_tokens % align) % align
        return n_tokens

    def align_token_side_channel(self, tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Pad, cp-shard and tp-chunk a ``[tokens, ...]`` channel exactly as the input is."""
        missing = self.padded_length(tensor.shape[0]) - tensor.shape[0]
        if missing:
            pad = torch.full((missing, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=0)
        if self.parallel_dims.cp_enabled:
            (local,), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (tensor.unsqueeze(0).to(self.device),),
                None,
                self.config.parallelism.context_parallel_load_balancer,
                input_seq_dim=1,
            )
            tensor = local.squeeze(0).to(tensor.device)
        if self.parallel_dims.tp_enabled:
            mesh = self.parallel_dims.get_mesh("tp")
            tp = mesh.size()
            if tensor.shape[0] % tp:
                raise ValueError(
                    f"a {tensor.shape[0]}-token side channel does not divide across {tp} tensor-parallel ranks"
                )
            tensor = tensor.chunk(tp, dim=0)[dist.get_rank(mesh.get_group())]
        return tensor

    def _microbatch_inputs(self, batches: list) -> tuple[list[dict], list[torch.Tensor]]:
        if self.parallel_dims.pp_enabled:
            expected = self.num_pipeline_parallel_microbatches
            if len(batches) != expected:
                raise ValueError(
                    f"the PP schedule was built for {expected} microbatches per optimizer step "
                    f"but this step has {len(batches)}; global_batch_size / dp / "
                    "micro_batch_size must be constant (no dynamic batch sizing with PP)"
                )

        def _model_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            tokens, positions = batch["tokens"], batch["position_ids"]
            pad = self.padded_length(tokens.shape[1]) - tokens.shape[1]
            if pad:
                tokens = torch.nn.functional.pad(tokens, (0, pad), value=0)
                extra = torch.arange(pad, device=positions.device, dtype=positions.dtype)
                positions = torch.cat([positions, extra.unsqueeze(0)], dim=1)
            return tokens, positions

        input_dicts = []
        for batch in batches:
            tokens, positions = _model_inputs(batch)
            input_dicts.append({"input": tokens, "positions": positions, **self._family_forward_kwargs})
        labels = [torch.full_like(input_dicts[i]["input"], i, dtype=torch.long) for i in range(len(batches))]
        return input_dicts, labels

    def _pipeline_will_infer_metadata(self, *, has_backward: bool) -> bool:
        """Whether the next schedule call starts with its shape-inference forward."""
        schedule = self.pp_schedule
        for prefix in ("_stage", "_stages"):
            forward_attr = f"{prefix}_forward_initialized"
            backward_attr = f"{prefix}_backward_initialized"
            if hasattr(schedule, forward_attr) and hasattr(schedule, backward_attr):
                break
        else:
            raise RuntimeError(
                f"{type(schedule).__name__} exposes no forward/backward initialization state; "
                "the pipeline schedule's metadata-inference forward can no longer be anticipated"
            )
        if not getattr(schedule, forward_attr):
            return True
        return has_backward != getattr(schedule, backward_attr)

    def run_forward_backward(self, batches, loss_closure: Callable) -> list[dict]:
        """One optimizer step's microbatches; under PP only the last stage returns log dicts."""
        batches = list(batches)
        self.loss_fn.arm(batches, loss_closure, "train")
        input_dicts, labels = self._microbatch_inputs(batches)
        ones = torch.ones((), device=self.device)
        with routing_replay.consumption_guard(self.model_parts, len(batches)):
            if self.parallel_dims.pp_enabled:
                if self._pipeline_will_infer_metadata(has_backward=True):
                    routing_replay.bypass_schedule_initialization(self.model_parts)
                self.forward_backward_step(input_dict=input_dicts, labels=labels, global_valid_tokens=ones)
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    self.forward_backward_step(input_dict=input_dict, labels=label, global_valid_tokens=ones)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def run_forward(self, batches, compute: Callable) -> list:
        """Forward-only over the microbatches; under PP only the last stage returns."""
        batches = list(batches)
        self.loss_fn.arm(batches, compute, "eval")
        input_dicts, labels = self._microbatch_inputs(batches)
        with routing_replay.consumption_guard(self.model_parts, len(batches)):
            if self.parallel_dims.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs = [], [], []
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    arg_mbs.append((inputs,))
                    kwarg_mbs.append(extra)
                    target_mbs.append(label)
                losses = [] if self.pp_has_last_stage else None
                if self._pipeline_will_infer_metadata(has_backward=False):
                    routing_replay.bypass_schedule_initialization(self.model_parts)
                self.pp_schedule.eval(
                    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs=kwarg_mbs,
                    target_mbs=target_mbs if self.pp_has_last_stage else None,
                    losses=losses,
                    return_outputs=False,
                )
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    pred = self.model_parts[0](inputs, **extra)
                    self.loss_fn(pred, label)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def apply_optimizer_step(self) -> StepMetrics:
        grad_norm = titan_dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        self.step += 1
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        return StepMetrics(grad_norm=float(grad_norm.item()), extra_metrics=self.lr_schedulers.get_metrics())

    def enable_context_parallel_gather(self) -> None:
        """Point the loss adapter at the CP mesh; only data-independent balancers can be undone."""
        if not self.parallel_dims.cp_enabled:
            return
        balancer_type = self.config.parallelism.context_parallel_load_balancer
        if balancer_type != "headtail":
            raise ValueError(
                f"context_parallel_load_balancer={balancer_type!r} is not supported yet: only a "
                "data-independent sharding can be undone from the sequence length"
            )
        self.loss_fn.set_context_parallel(self.parallel_dims.get_mesh("cp"), balancer_type)

    def has_last_stage(self) -> bool:
        return (not self.parallel_dims.pp_enabled) or self.pp_has_last_stage

    def step_runner(self) -> "TrainerStepRunner":
        return TrainerStepRunner(self)


class TrainerStepRunner:
    """The shared loop's step-runner protocol over the trainer, kept off the
    trainer because it already has a ``forward_backward_step`` of its own."""

    def __init__(self, trainer: TitanTrainer):
        self.trainer = trainer

    def forward_only_step(self, batches, compute: Callable) -> list:
        return self.trainer.run_forward(batches, compute)

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        return self.trainer.run_forward_backward(batches, loss_closure)

    def zero_grad(self) -> None:
        self.trainer.optimizers.zero_grad(set_to_none=True)

    def apply_step(self) -> StepMetrics:
        return self.trainer.apply_optimizer_step()
