"""Bridges between released torch and the torch-nightly APIs torchtitan tracks.

Each shim backports one upstream torch change and is a no-op once the running
torch carries it. ``install()`` runs before any torchtitan object is built.
"""

import inspect
import logging

logger = logging.getLogger(__name__)


def install() -> None:
    _patch_fsdp2_grad_accumulation_attr_error()
    _patch_pipeline_schedule_microbatch_api()
    _shim_dist_set_timeout()


def _shim_dist_set_timeout() -> None:
    """Alias nightly's public ``torch.distributed.set_timeout`` to 2.13's ``_set_pg_timeout``."""
    import torch.distributed as dist

    if hasattr(dist, "set_timeout"):
        return
    from torch.distributed.distributed_c10d import _set_pg_timeout

    dist.set_timeout = _set_pg_timeout
    logger.info("Aliased torch.distributed.set_timeout to _set_pg_timeout")


def _patch_pipeline_schedule_microbatch_api() -> None:
    """Give the pipeline schedules nightly's ``step(arg_mbs=, kwarg_mbs=, target_mbs=, ...)``
    over 2.13's ``_step_microbatches``, replicating step()'s per-call bookkeeping."""
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
    """Backport upstream's getattr guard in ``FSDPParam.to_accumulated_grad_if_needed``,
    which a pipeline schedule's back-to-back backwards hit on 2.13."""
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
