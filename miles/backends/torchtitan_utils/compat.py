import functools
import inspect
import logging

logger = logging.getLogger(__name__)


def install() -> None:
    _patch_fsdp2_grad_accumulation_attr_error()
    _patch_pipeline_schedule_microbatch_api()
    _patch_pipeline_eval_output_cache()
    _shim_dist_set_timeout()


def _patch_pipeline_eval_output_cache() -> None:
    import torch
    from torch.distributed.pipelining.stage import _PipelineStageBase

    if torch.__version__.split("+")[0] != "2.14.1":
        return
    original_forward = _PipelineStageBase.forward_one_chunk
    if getattr(original_forward, "_miles_releases_eval_outputs", False):
        return
    original_send = _PipelineStageBase.get_fwd_send_ops

    # Torch 2.14.1 retains every eval logit tensor for a backward that never runs.
    # Remove with the upstream lifecycle fix: https://github.com/pytorch/pytorch/pull/200643.
    @functools.wraps(original_forward)
    def forward_one_chunk(self, fwd_chunk_id, *args, **kwargs):
        output = original_forward(self, fwd_chunk_id, *args, **kwargs)
        if self.is_last and not self.has_backward:
            self.fwd_cache.pop(fwd_chunk_id)
        return output

    @functools.wraps(original_send)
    def get_fwd_send_ops(self, fwd_chunk_id):
        # The last stage has no downstream send, but upstream still indexes its cache.
        if self.is_last and not self.has_backward:
            return []
        return original_send(self, fwd_chunk_id)

    forward_one_chunk._miles_releases_eval_outputs = True
    _PipelineStageBase.forward_one_chunk = forward_one_chunk
    _PipelineStageBase.get_fwd_send_ops = get_fwd_send_ops
    logger.info("Patched torch 2.14.1 pipeline eval to release final-stage output caches")


def _shim_dist_set_timeout() -> None:
    import torch.distributed as dist

    if hasattr(dist, "set_timeout"):
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
                and getattr(self, "_backward_requires_autograd", True)
                and not torch.is_grad_enabled()
            ):
                raise RuntimeError(
                    "step() requires gradients to be enabled for backward computation; "
                    "call eval() under torch.no_grad() instead."
                )
            stages = getattr(self, "_stages", None) or [self._stage]
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
        unsharded_param = getattr(self, "_unsharded_param", None)
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
