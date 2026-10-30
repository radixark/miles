import inspect
import logging


logger = logging.getLogger(__name__)


def install() -> None:
    _patch_fsdp2_grad_accumulation_attr_error()
    _patch_pipeline_schedule_microbatch_api()
    _shim_dist_set_timeout()
    _compile_flex_attention_for_dynamic_shapes()


def _compile_flex_attention_for_dynamic_shapes() -> None:
    """torchtitan compiles flex_attention for the first shape it sees and recompiles dynamically on the
    second; the two kernels round differently, so the first micro-batch of the log-prob pass and the
    same micro-batch in training disagree in the last bits (ppo_kl != 0 on step 0). One dynamic kernel
    serves both."""
    import torch
    from torch.nn.attention.flex_attention import flex_attention
    from torchtitan.models.common.attention import FlexAttention

    if getattr(
        FlexAttention._compiled_flex_attn, "_dynamic_shapes", False
    ):  # config-access-exempt: idempotence marker set below
        return
    compiled = torch.compile(flex_attention, dynamic=True, options=FlexAttention.inductor_configs)
    compiled._dynamic_shapes = True
    FlexAttention._compiled_flex_attn = staticmethod(compiled)
    logger.info("Compiled flex_attention for dynamic shapes, so every micro-batch runs one kernel")


def _shim_dist_set_timeout() -> None:
    import torch.distributed as dist

    if hasattr(
        dist, "set_timeout"
    ):  # config-access-exempt: PyTorch versions differ in the public distributed timeout API
        return
    from torch.distributed.distributed_c10d import _set_pg_timeout

    dist.set_timeout = _set_pg_timeout
    logger.info("Aliased torch.distributed.set_timeout to _set_pg_timeout")


def _patch_pipeline_schedule_microbatch_api() -> None:
    import torch
    from torchtitan.distributed.pipeline_parallel import PipelineScheduleMulti, PipelineScheduleSingle

    if "arg_mbs" in inspect.signature(PipelineScheduleSingle.step).parameters:
        return

    def _make_step(original_step):
        def step(
            self,
            *args,
            arg_mbs=None,
            kwarg_mbs=None,
            target_mbs=None,
            target=None,
            losses=None,
            return_outputs=True,
            loss_kwargs=None,
            **kwargs
        ):
            if arg_mbs is None and kwarg_mbs is None and target_mbs is None:
                return original_step(
                    self,
                    *args,
                    target=target,
                    losses=losses,
                    return_outputs=return_outputs,
                    loss_kwargs=loss_kwargs,
                    **kwargs,
                )
            if (
                self._has_backward
                and getattr(
                    self, "_backward_requires_autograd", True
                )  # config-access-exempt: PyTorch pipeline versions may omit the backward autograd flag
                and not torch.is_grad_enabled()
            ):
                raise RuntimeError(
                    "step() requires gradients to be enabled for backward computation; "
                    "call eval() under torch.no_grad() instead."
                )
            stages = getattr(self, "_stages", None) or [
                self._stage
            ]  # config-access-exempt: PyTorch pipeline schedules expose either multiple stages or one stage
            for stage in stages:
                stage.has_backward = self._has_backward
                stage.clear_runtime_states()
            return self._step_microbatches(
                arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs=loss_kwargs
            )

        return step

    PipelineScheduleSingle.step = _make_step(PipelineScheduleSingle.step)
    PipelineScheduleMulti.step = _make_step(PipelineScheduleMulti.step)
    logger.info("Patched pipeline schedules with nightly's microbatch-list step API")


def _patch_fsdp2_grad_accumulation_attr_error() -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    if "getattr" in inspect.getsource(FSDPParam.to_accumulated_grad_if_needed):
        return

    def to_accumulated_grad_if_needed(self) -> None:
        unsharded_param = getattr(
            self, "_unsharded_param", None
        )  # config-access-exempt: PyTorch FSDP may not materialize an unsharded parameter yet
        if (
            self.reduce_dtype is None
            or unsharded_param is None
            or unsharded_param.grad is None
            or unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = unsharded_param.grad
        unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    FSDPParam.to_accumulated_grad_if_needed = to_accumulated_grad_if_needed
    logger.info("Patched FSDPParam.to_accumulated_grad_if_needed with the upstream getattr guard")
