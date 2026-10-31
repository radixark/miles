from collections.abc import Callable
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import PipelineScheduleSingle
from torchtitan.config import Configurable
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import ParallelismContext
from torchtitan.models.base import BaseModel
from torchtitan.training_engine import TrainingEngine

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.training_utils.torch_native.step_runner import StepMetrics

_FLEX_BLOCK = 128
_CP_LENGTH_BUCKET = 1024


class TitanTrainer(Configurable):
    """torchtitan's TrainingEngine builds, parallelizes, checkpoints and steps the model; miles feeds it
    variable-length packed microbatches through its own forward/backward loop."""

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingEngine.Config):
        model: BaseModel.Config
        hf_assets_path: str
        output_dir: str

        def __post_init__(self) -> None:
            TrainingEngine.Config.__post_init__(self)
            validate_model_training_config(
                self.model,
                parallelism=self.parallelism,
                training=self.training,
                debug=self.debug,
                activation_remat=self.activation_remat,
                max_num_documents=None,
            )

    def __init__(self, config: Config, *, parallelism: ParallelismContext | None = None):
        self.config = config
        torch.autograd.set_multithreading_enabled(False)
        self.engine = TrainingEngine(
            config, model_config=config.model, max_num_documents=None, output_dir=config.output_dir
        )
        if parallelism is not None:
            self.engine.parallelism_context = parallelism
        self.engine.initialize(hf_assets_path=config.hf_assets_path)
        self.parallelism = self.engine.parallelism_context
        self.model_parts = self.engine.model_parts
        self.optimizers = self.engine.optim.optimizers
        self.device = self.engine.device
        self.loss_fn = self.engine.loss_fn

    @property
    def step(self) -> int:
        return self.engine.num_completed_steps

    def padded_length(self, n_tokens: int) -> int:
        if self.parallelism.pp_enabled:
            target = self.config.training.max_context_length
            if n_tokens > target:
                raise ValueError(
                    f"packed microbatch of {n_tokens} tokens exceeds --seq-length "
                    f"{target}, which is the fixed shape PP stages exchange"
                )
            return target
        if self.parallelism.cp_enabled:
            align = max(self.parallelism.cp * _FLEX_BLOCK, _CP_LENGTH_BUCKET)
            return n_tokens + (align - n_tokens % align) % align
        return n_tokens

    def align_token_side_channel(self, tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        missing = self.padded_length(tensor.shape[0]) - tensor.shape[0]
        if missing:
            pad = torch.full((missing, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=0)
        for axis in ("cp", "tp"):
            mesh = self.parallelism.get_optional_mesh(axis)
            if mesh is None:
                continue
            if tensor.shape[0] % mesh.size():
                raise ValueError(
                    f"a {tensor.shape[0]}-token side channel does not divide across {mesh.size()} {axis} ranks"
                )
            tensor = tensor.chunk(mesh.size(), dim=0)[dist.get_rank(mesh.get_group())]
        return tensor

    def _microbatch_inputs(self, batches: list) -> list[tuple]:
        if self.parallelism.pp_enabled:
            expected = self.config.parallelism.num_pp_microbatches
            if len(batches) != expected:
                raise ValueError(
                    f"the PP schedule was built for {expected} microbatches per optimizer step "
                    f"but this step has {len(batches)}; global_batch_size / dp / "
                    "micro_batch_size must be constant (no dynamic batch sizing with PP)"
                )

        prepared = []
        for index, batch in enumerate(batches):
            tokens, positions = batch["tokens"].flatten(), batch["position_ids"].flatten()
            pad = self.padded_length(tokens.shape[0]) - tokens.shape[0]
            if pad:
                tokens = torch.nn.functional.pad(tokens, (0, pad), value=0)
                positions = torch.cat([positions, torch.arange(pad, device=positions.device, dtype=positions.dtype)])
            input_dict = {
                "input": tokens.to(self.device),
                "labels": torch.full_like(tokens, index, dtype=torch.long, device=self.device),
                "positions": positions.to(self.device),
            }
            with self.parallelism.activate_spmd():
                prepared.append(
                    self.model_parts[0].preprocess_inputs(
                        input_dict,
                        parallelism_context=self.parallelism,
                        parallelism=self.config.parallelism,
                        max_num_documents=None,
                        max_context_length=self.config.training.max_context_length,
                    )
                )
        return prepared

    def _bypass_schedule_probe(self, *, has_backward: bool) -> None:
        if self._pipeline_will_infer_metadata(has_backward=has_backward):
            routing_replay.bypass_schedule_initialization(self.model_parts)

    def _pipeline_will_infer_metadata(self, *, has_backward: bool) -> bool:
        schedule = self.engine.pp_schedule
        if isinstance(schedule, PipelineScheduleSingle):
            forward_initialized = schedule._stage_forward_initialized
            backward_initialized = schedule._stage_backward_initialized
        else:
            forward_initialized = schedule._stages_forward_initialized
            backward_initialized = schedule._stages_backward_initialized
        return not forward_initialized or has_backward != backward_initialized

    def _pipeline_microbatches(self, prepared: list[tuple]) -> dict:
        return dict(
            arg_mbs=[(inputs,) for inputs, *_ in prepared] if self.engine.pp_has_first_stage else None,
            kwarg_mbs=[model_kwargs for _, _, model_kwargs, _ in prepared],
            target_mbs=[labels for _, labels, _, _ in prepared] if self.engine.pp_has_last_stage else None,
            losses=[] if self.engine.pp_has_last_stage else None,
            return_outputs=False,
        )

    def run_forward_backward(self, batches, loss_closure: Callable) -> list[dict]:
        batches = list(batches)
        self.loss_fn.arm(batches, loss_closure, is_training=True)
        prepared = self._microbatch_inputs(batches)
        with routing_replay.consumption_guard(self.model_parts, len(batches)), self.parallelism.activate_spmd():
            if self.parallelism.pp_enabled:
                self._bypass_schedule_probe(has_backward=True)
                microbatches = self._pipeline_microbatches(prepared)
                self.engine.pp_schedule.step(**microbatches)
                for loss in microbatches["losses"] or []:
                    self.engine.loss += loss.detach()
            else:
                for inputs, labels, model_kwargs, _ in prepared:
                    pred = self.model_parts[0](inputs, **model_kwargs)
                    loss, _ = self.loss_fn(pred, labels)
                    del pred
                    with spmd.no_typecheck():
                        loss.backward()
                    self.engine.loss += loss.detach()
        return self.loss_fn.collect() if self.has_last_stage() else []

    def run_forward(self, batches, compute: Callable) -> list:
        batches = list(batches)
        self.loss_fn.arm(batches, compute, is_training=False)
        prepared = self._microbatch_inputs(batches)
        with routing_replay.consumption_guard(self.model_parts, len(batches)), self.parallelism.activate_spmd():
            if self.parallelism.pp_enabled:
                self._bypass_schedule_probe(has_backward=False)
                self.engine.pp_schedule.eval(**self._pipeline_microbatches(prepared))
            else:
                for inputs, labels, model_kwargs, _ in prepared:
                    self.loss_fn(self.model_parts[0](inputs, **model_kwargs), labels)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def zero_grad(self) -> None:
        self.optimizers.zero_grad(set_to_none=True)
        self.engine.loss = torch.zeros((), device=self.device)

    def apply_optimizer_step(self) -> StepMetrics:
        grad_norm = self.engine.optim_step()
        return StepMetrics(
            grad_norm=float(grad_norm.item()), extra_metrics=self.engine.optim.lr_schedulers.get_metrics()
        )

    def configure_loss_reduction(self) -> None:
        dims = self.parallelism
        self.loss_fn.set_gradient_scale(1.0 / (dims.dp_replicate * dims.dp_shard * dims.cp))
        if not dims.cp_enabled:
            return
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit,
            2 * (self.config.training.max_context_length // _CP_LENGTH_BUCKET + 1),
        )
        self.loss_fn.set_context_parallel(self.parallelism.get_mesh("cp"))

    def has_last_stage(self) -> bool:
        return (not self.parallelism.pp_enabled) or self.engine.pp_has_last_stage

    def step_runner(self) -> "TrainerStepRunner":
        return TrainerStepRunner(self)


class TrainerStepRunner:
    def __init__(self, trainer: TitanTrainer):
        self.trainer = trainer

    def forward_only_step(self, batches, compute: Callable) -> list:
        return self.trainer.run_forward(batches, compute)

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        return self.trainer.run_forward_backward(batches, loss_closure)

    def zero_grad(self) -> None:
        self.trainer.zero_grad()

    def apply_step(self) -> StepMetrics:
        return self.trainer.apply_optimizer_step()
