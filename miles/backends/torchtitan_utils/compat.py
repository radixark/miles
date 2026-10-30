"""torchtitan 0.4.0 is released against torch 2.15; these shims run it on the torch 2.14 sglang links.

`backport_torch_apis` must run before torchtitan is first imported, so the package `__init__` runs it;
`install` patches FSDP once the trainer starts.
"""

import inspect
import logging

import torch.distributed.fsdp as torch_fsdp
import torch.distributed.pipelining as torch_pipelining

logger = logging.getLogger(__name__)

_DIST_MOE_PIPELINING_APIS = ("analyze_pipeline_activation_liveness", "PipelineStageInfo")


def backport_torch_apis() -> None:
    _stub_dist_moe_pipelining_apis()
    _accept_unset_param_dtype_override()


def install() -> None:
    _patch_fsdp2_grad_accumulation_attr_error()
    _compile_flex_attention_for_dynamic_shapes()


def _compile_flex_attention_for_dynamic_shapes() -> None:
    """torchtitan compiles flex_attention for the first shape it sees and recompiles dynamically on the
    second; the two kernels round differently, so the first micro-batch of the log-prob pass and the
    same micro-batch in training disagree in the last bits (ppo_kl != 0 on step 0). One dynamic kernel
    serves both. fullgraph and recompile_limit mirror torchtitan's own compile."""
    import torch
    from torch.nn.attention.flex_attention import flex_attention
    from torchtitan.models.common.attention import FlexInnerAttention

    if getattr(
        FlexInnerAttention._compiled_flex_attn, "_dynamic_shapes", False
    ):  # config-access-exempt: idempotence marker set below
        return
    compiled = torch.compile(
        flex_attention, dynamic=True, fullgraph=True, options=FlexInnerAttention.inductor_configs, recompile_limit=16
    )
    compiled._dynamic_shapes = True
    FlexInnerAttention._compiled_flex_attn = staticmethod(compiled)
    logger.info("Compiled flex_attention for dynamic shapes, so every micro-batch runs one kernel")


def _stub_dist_moe_pipelining_apis() -> None:
    def unsupported(*args, **kwargs):
        raise NotImplementedError("Dist-MoE pipeline liveness needs torch>=2.15")

    for name in _DIST_MOE_PIPELINING_APIS:
        if not hasattr(torch_pipelining, name):  # config-access-exempt: torch 2.15 adds these pipelining APIs
            setattr(torch_pipelining, name, unsupported)


def _accept_unset_param_dtype_override() -> None:
    policy_cls = torch_fsdp.MixedPrecisionPolicy
    if "param_dtype_override_fn" in inspect.signature(policy_cls).parameters:
        return

    def mixed_precision_policy(*args, param_dtype_override_fn=None, **kwargs):
        if param_dtype_override_fn is not None:
            raise NotImplementedError("MixedPrecisionPolicy(param_dtype_override_fn=...) needs torch>=2.15")
        return policy_cls(*args, **kwargs)

    torch_fsdp.MixedPrecisionPolicy = mixed_precision_policy
    logger.info("Let MixedPrecisionPolicy accept the unset param_dtype_override_fn torchtitan passes")


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
