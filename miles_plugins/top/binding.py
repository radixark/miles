"""Public binding API for TOP family specs: role binding, serving pin, batch invariance, stamps."""

from __future__ import annotations

import functools
import logging

import torch
from megatron.core.transformer.spec_utils import ModuleSpec
from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1
from sglang.srt.true_on_policy.qk_norm import supports_fused_qk_norm

from miles.true_on_policy.device import record_trainer_device
from miles_plugins.top.install import installed as _installed
from miles_plugins.top.ops import delegated_matched_tp1_linear, delegated_qk_norm, delegated_rms_norm

try:
    from sglang.srt.debug_utils.dumper import dumper as _dumper
except Exception:  # the dumper is optional; never fail a run over instrumentation
    _dumper = None

logger = logging.getLogger(__name__)


def canon_dump(name: str, value) -> None:
    """Dump under a name shared with sglang's side."""
    if _dumper is None:
        return
    try:
        _dumper.dump(name, value)
    except Exception:
        pass


class CaptureTap(torch.nn.Module):
    """A named capture point (the dumper hooks module outputs): the value as a view, a node of its own, so the
    gradient captured here is the share that flows through this use, not the tensor's total over every use."""

    def forward(self, x):
        return x.view_as(x)


NORM_ROLES = (
    "input_layernorm",
    "pre_mlp_layernorm",
    "q_layernorm",
    "k_layernorm",
)
ROW_LINEAR_ROLES = ("linear_proj", "linear_fc2")
ATTENTION_ROLES = ("core_attention",)


class TopRMSNorm(torch.nn.Module):
    """RMSNorm delegating to SGLang with explicit arithmetic parameters."""

    def __init__(self, config, hidden_size, eps=1e-6, role="", **kwargs):
        super().__init__()
        from sglang.srt.layers.layernorm import RMSNorm

        self.eps, self.role = eps, role
        self.hidden_size = hidden_size
        self.requires_residual_pair = role != "input_layernorm"
        self._packed_residual = role in ("input_layernorm", "pre_mlp_layernorm", "final_layernorm")
        self.returns_residual = role in ("input_layernorm", "pre_mlp_layernorm")
        if (
            self._packed_residual and getattr(config, "pipeline_hidden_size", None) != 2 * config.hidden_size
        ):  # config-access-exempt: TransformerConfig versions and model families differ in optional pipeline_hidden_size
            raise RuntimeError(f"[top] {role} requires packed residual transport (pipeline_hidden_size=2H)")
        sgl = RMSNorm(hidden_size, eps=eps)
        # not a submodule: registering it too would put one tensor under two state_dict keys
        object.__setattr__(self, "_sgl", sgl)
        self.weight = sgl.weight

    def forward(self, x, residual=None, post_residual_addition=None):
        if self._packed_residual:
            from miles_plugins.top.residual_norm import delegated_residual_norm

            if residual is not None or post_residual_addition is not None:
                raise ValueError("[top] packed norm cannot also receive a separate residual")
            if x.shape[-1] == 2 * self.hidden_size:
                shape = (*x.shape[:-1], self.hidden_size)
                branch, residual = x.split(self.hidden_size, dim=-1)
                out, residual = delegated_residual_norm(
                    branch.reshape(-1, self.hidden_size),
                    residual.reshape(-1, self.hidden_size),
                    self._sgl,
                )
                out, residual = out.view(shape), residual.view(shape)
                if self.role == "input_layernorm":
                    canon_dump("top_attn_norm_out", out)
                return out if self.role == "final_layernorm" else (out, residual)
            if x.shape[-1] != self.hidden_size or self.requires_residual_pair:
                raise ValueError(f"[top] {self.role} expected a packed residual pair")
        if residual is not None or post_residual_addition is not None:
            raise NotImplementedError("[top] standalone norm roles do not take a residual")
        if (
            self.role in ("q_layernorm", "k_layernorm")
            and x.is_cuda
            and supports_fused_qk_norm(self.hidden_size, x.dtype)
            and self.weight.dtype == x.dtype
            and not self._sgl.cast_x_before_out_mul
            and self._sgl.override_orig_dtype is None
            and self._sgl.variance_size_override is None
        ):
            return delegated_qk_norm(x, self.weight, self.eps)
        # sglang's forward_cuda flattens rank>2 without restoring it
        shape = x.shape
        flat = x.reshape(-1, shape[-1]) if x.dim() != 2 else x
        out = delegated_rms_norm(flat, self._sgl, self.eps).view(shape)
        if self.role == "input_layernorm":
            canon_dump("top_attn_norm_out", out)
        return (out, x) if self.returns_residual else out


class TopRowParallelLinear:
    """Row-parallel linear whose GEMM is serving's full-K BI GEMM."""

    _cls = None

    def __new__(cls, *args, **kwargs):
        return cls._build()(*args, **kwargs)

    @classmethod
    def _build(cls):
        if cls._cls is not None:
            return cls._cls
        from megatron.core.tensor_parallel.layers import RowParallelLinear

        class _Impl(RowParallelLinear):
            def __init__(self, *args, role="", **kwargs):
                self._top_role = role
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                out = super().forward(*args, **kwargs)
                # post-reduce, matching sglang's dump point (not the per-rank partial)
                if self._top_role == "linear_proj":
                    canon_dump("top_attn_out", out[0] if isinstance(out, tuple) else out)
                return out

            def _forward_impl(self, input, weight, *args, **kwargs):
                bias = kwargs.pop("bias", None)
                if kwargs.get("sequence_parallel"):
                    raise NotImplementedError("[top] delegated row linear + sequence parallel")
                shape = input.shape
                x2d = input.reshape(-1, shape[-1])
                if self.config.tensor_model_parallel_size != 1:
                    raise ValueError("[top] the full-K row linear requires trainer TP1")
                out = delegated_matched_tp1_linear(x2d, weight, bias)
                return out.view(*shape[:-1], out.shape[-1])

        cls._cls = _Impl
        _Impl.__name__ = "TopRowParallelLinear"
        return _Impl


def bind_roles(node, roles, cls, stamp, as_spec_param=False):
    """Bind declared roles by field name."""
    from megatron.core.transformer.identity_op import IdentityOp

    if isinstance(node, ModuleSpec):
        if node.submodules is not None:
            bind_roles(node.submodules, roles, cls, stamp, as_spec_param)
        return
    if not hasattr(
        node, "__dataclass_fields__"
    ):  # config-access-exempt: layer spec trees mix dataclasses, classes and callables, which differ in __dataclass_fields__
        return
    for f in node.__dataclass_fields__:
        v = getattr(node, f)  # config-access-exempt: attribute selected at runtime from a table of names
        if f in roles:
            if v is IdentityOp or v is None:
                continue
            stamp.append(
                (f, getattr(v, "__name__", str(v)), cls.__name__)
            )  # config-access-exempt: layer spec trees mix dataclasses, classes and callables, which differ in __name__
            setattr(node, f, ModuleSpec(module=cls, params={"role": f}) if as_spec_param else cls)
        elif isinstance(v, ModuleSpec) or hasattr(
            v, "__dataclass_fields__"
        ):  # config-access-exempt: layer spec trees mix dataclasses, classes and callables, which differ in __dataclass_fields__
            bind_roles(v, roles, cls, stamp, as_spec_param)
        elif isinstance(v, functools.partial):
            for sub in v.keywords.values():
                if hasattr(
                    sub, "__dataclass_fields__"
                ):  # config-access-exempt: layer spec trees mix dataclasses, classes and callables, which differ in __dataclass_fields__
                    bind_roles(sub, roles, cls, stamp, as_spec_param)


def bind_residual_layer(spec: ModuleSpec, stamp: list[tuple[str, str, str]]) -> None:
    """Bind the one paired add+norm path without replacing family-specific submodules."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from miles_plugins.top.residual_norm import ResidualTransformerLayer, deferred_residual_add

    if spec.module not in (TransformerLayer, ResidualTransformerLayer):
        raise NotImplementedError(f"[top] residual binding requires stock TransformerLayer, found {spec.module!r}")
    bind_roles(spec, NORM_ROLES, TopRMSNorm, stamp, as_spec_param=True)
    for role in ("input_layernorm", "pre_mlp_layernorm"):
        norm = getattr(
            spec.submodules, role, None
        )  # config-access-exempt: attribute selected at runtime from a table of names
        if not isinstance(norm, ModuleSpec) or norm.module is not TopRMSNorm:
            raise NotImplementedError(f"[top] paired residual layer requires an exposed {role}")
    spec.module = ResidualTransformerLayer
    spec.submodules.self_attn_bda = deferred_residual_add
    spec.submodules.mlp_bda = deferred_residual_add
    stamp.append(("residual_add", "BF16 add", "packed branch/residual -> stock fused norm"))


def bind_dense_activation(config, stamp: list[tuple[str, str, str]]) -> None:
    """Dense SwiGLU stays Megatron's eager SiLU, then multiply: the BF16 rounding serving's SiluAndMul keeps."""
    if config.activation_func is not torch.nn.functional.silu or not config.gated_linear_unit:
        raise NotImplementedError("[top] the dense MLP activation must be SwiGLU")
    if config.bias_activation_fusion:
        raise NotImplementedError("[top] dense SwiGLU must run eagerly; pass --no-bias-swiglu-fusion")
    stamp.append(("activation", "SwiGLU", "eager SiLU * up (serving SiluAndMul rounding)"))


def _existing_server_args():
    try:
        from sglang.srt.runtime_context import get_server_args

        return get_server_args()
    except Exception:
        return None


def pin_sglang(args, server_kwargs) -> str:
    """Pin the trainer's sglang ops to the rollout's program (dispatch is process-global).

    ``server_kwargs`` are the family's serving-program arguments (attention backend, KV dtype, MoE backends)."""
    from sglang.srt.runtime_context import publish, publish_role
    from sglang.srt.server_args import ServerArgs

    contract = args.true_on_policy_contract or TRUE_ON_POLICY_V1
    record_trainer_device()
    existing = _existing_server_args()
    if existing is not None:
        if existing.true_on_policy_contract != contract:
            raise RuntimeError(
                f"[top] sglang already pinned to {existing.true_on_policy_contract!r}, program wants {contract!r}"
            )
        if (
            publish_role() != "top_trainer"
            or not existing.enable_deterministic_inference
            or existing.model_path != args.hf_checkpoint
        ):
            raise RuntimeError(
                "[top] trainer requires its own published deterministic runtime; "
                "refusing to reuse scheduler or legacy process state"
            )
        return contract
    # no tp_size: the program must not vary with topology
    publish(
        ServerArgs(
            model_path=args.hf_checkpoint,
            enable_deterministic_inference=True,
            true_on_policy_contract=contract,
            **server_kwargs,
        ),
        role="top_trainer",
    )
    return contract


def enable_batch_invariant() -> bool:
    """Bind shared GEMMs and log-softmax; the core BI flag also selects TE attention."""
    from miles_plugins.top.install import (
        install_backward_without_batch_invariance,
        install_batch_invariant_recompute,
        install_serving_bi_overloads,
    )

    # Before the import below, so the enable it binds also registers serving's extra overloads.
    install_serving_bi_overloads()
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        configure_batch_invariant_log_softmax,
        configure_batch_invariant_matmul,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )
    from sglang.srt.batch_invariant_ops import matmul_persistent
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import log_softmax

    configure_batch_invariant_matmul(matmul_persistent)
    configure_batch_invariant_log_softmax(log_softmax)
    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()
    install_backward_without_batch_invariance()
    install_batch_invariant_recompute()
    return is_batch_invariant_mode_enabled()


def report_bindings(args, contract, bik, binds) -> None:
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    logger.info(
        f"[top] sglang_pinned={contract} "
        f"transformer_impl={args.backend.transformer_impl} batch_invariant_applied={bik} "
        f"installed={_installed()}"
    )
    for role, was, now in binds:
        logger.info(f"[top] bind {role}: {was} -> {now}")


def hf_model_type(args) -> str | None:
    """Read the actual model family from the checkpoint configuration."""
    ckpt = args.hf_checkpoint
    if not ckpt:
        return None
    from miles.utils.hf_utils.config import load_hf_config

    return getattr(
        load_hf_config(ckpt), "model_type", None
    )  # config-access-exempt: third-party HF configurations may omit model_type
