"""Build a Megatron spec, then bind declared roles onto the program."""

from __future__ import annotations

import functools

import torch
from megatron.core.transformer.spec_utils import ModuleSpec

from sglang.srt.true_on_policy import should_use_tp_invariant_row_linear

from miles_plugins.top.ops import delegated_rms_norm, delegated_tp_inv_linear

try:
    from sglang.srt.debug_utils.dumper import dumper as _dumper
except Exception:  # the dumper is optional; never fail a run over instrumentation
    _dumper = None
from miles.true_on_policy.schema import TRUE_ON_POLICY_V1_SCHEMA
from miles_plugins.top.install import installed as _installed
from miles_plugins.top.program import QWEN3_DENSE_V1, NormParams

_ACTIVE_PROGRAM = None


def active_program():
    """The program this run resolved to, for code outside the spec that must ask what is declared.

    Model plugins consult this rather than sniffing flags: whether an op is delegated is a property
    of the declared program, not something to infer from --transformer-impl or a contract name.
    Returns None before the spec is built (a non-true-on-policy run never sets it).
    """
    return _ACTIVE_PROGRAM


def _canon_dump(name: str, value) -> None:
    """Dump under a name shared with sglang's side."""
    if _dumper is None:
        return
    try:
        _dumper.dump(name, value)
    except Exception:
        pass

_NORM_ROLES = (
    "input_layernorm",
    "pre_mlp_layernorm",
    "q_layernorm",
    "k_layernorm",
)
_ROW_LINEAR_ROLES = ("linear_proj", "linear_fc2")
_ATTENTION_ROLES = ("core_attention",)


class TopRMSNorm(torch.nn.Module):
    """RMSNorm delegating to sglang's dispatch; role differences come from the program."""

    def __init__(self, config, hidden_size, eps=1e-6, role="", **kwargs):
        super().__init__()
        from sglang.srt.layers.layernorm import RMSNorm

        p = _ACTIVE_PROGRAM.norm.get(role, NormParams()) if _ACTIVE_PROGRAM else NormParams()
        self.eps, self.role = eps, role
        self.hidden_size = hidden_size
        self.requires_residual_pair = role != "input_layernorm"
        self._packed_residual = role in ("input_layernorm", "pre_mlp_layernorm", "final_layernorm")
        if self._packed_residual and getattr(config, "pipeline_hidden_size", None) != 2 * config.hidden_size:
            raise RuntimeError(f"[top] {role} requires packed residual transport (pipeline_hidden_size=2H)")
        self._want_fp32_weight = p.weight_dtype is torch.float32
        sgl = RMSNorm(
            hidden_size,
            eps=eps,
            true_on_policy_weight_dtype=p.weight_dtype,
            true_on_policy_override_orig_dtype=p.override_orig_dtype,
            true_on_policy_fp32_residual=p.fp32_residual,
        )
        # not a submodule: registering it too would put one tensor under two state_dict keys
        object.__setattr__(self, "_sgl", sgl)
        self.weight = sgl.weight

    def _apply(self, fn):
        # --bf16 casts every parameter; re-float if the program declares fp32
        super()._apply(fn)
        if self._want_fp32_weight:
            self.weight.data = self.weight.data.float()
            if self.weight.grad is not None:
                self.weight.grad.data = self.weight.grad.data.float()
        return self

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
                    residual.reshape(-1, self.hidden_size), self._sgl,
                )
                out, residual = out.view(shape), residual.view(shape)
                if self.role == "input_layernorm":
                    _canon_dump("top_attn_norm_out", out)
                return out if self.role == "final_layernorm" else (out, residual)
            if x.shape[-1] != self.hidden_size or self.requires_residual_pair:
                raise ValueError(f"[top] {self.role} expected a packed residual pair")
        if residual is not None or post_residual_addition is not None:
            raise NotImplementedError("[top] standalone norm roles do not take a residual")
        # sglang's forward_cuda flattens rank>2 without restoring it
        shape = x.shape
        flat = x.reshape(-1, shape[-1]) if x.dim() != 2 else x
        out = delegated_rms_norm(flat, self._sgl, self.eps).view(shape)
        if self.role == "input_layernorm":
            _canon_dump("top_attn_norm_out", out)
        return out


class TopRowParallelLinear:
    """Row-parallel linear whose GEMM is sglang's TP-invariant matmul."""

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
                    _canon_dump("top_attn_out", out[0] if isinstance(out, tuple) else out)
                return out

            def _forward_impl(self, input, weight, *args, **kwargs):
                bias = kwargs.pop("bias", None)
                if kwargs.get("sequence_parallel"):
                    raise NotImplementedError("[top] delegated row linear + sequence parallel")
                shape = input.shape
                x2d = input.reshape(-1, shape[-1])
                if not should_use_tp_invariant_row_linear(x2d.shape[-1]):
                    return super()._forward_impl(input, weight, *args, bias=bias, **kwargs)
                out = delegated_tp_inv_linear(x2d, weight, bias)
                return out.view(*shape[:-1], out.shape[-1])

        cls._cls = _Impl
        _Impl.__name__ = "TopRowParallelLinear"
        return _Impl


def _bind_roles(node, roles, cls, stamp, as_spec_param=False):
    """Bind declared roles by field name."""
    from megatron.core.transformer.identity_op import IdentityOp

    if isinstance(node, ModuleSpec):
        if node.submodules is not None:
            _bind_roles(node.submodules, roles, cls, stamp, as_spec_param)
        return
    if not hasattr(node, "__dataclass_fields__"):
        return
    for f in node.__dataclass_fields__:
        v = getattr(node, f)
        if f in roles:
            if v is IdentityOp or v is None:
                continue
            stamp.append((f, getattr(v, "__name__", str(v)), cls.__name__))
            setattr(node, f, ModuleSpec(module=cls, params={"role": f}) if as_spec_param else cls)
        elif isinstance(v, ModuleSpec) or hasattr(v, "__dataclass_fields__"):
            _bind_roles(v, roles, cls, stamp, as_spec_param)
        elif isinstance(v, functools.partial):
            for sub in v.keywords.values():
                if hasattr(sub, "__dataclass_fields__"):
                    _bind_roles(sub, roles, cls, stamp, as_spec_param)


def _bind_residual_layer(spec: ModuleSpec, stamp: list[tuple[str, str, str]]) -> None:
    """Bind the one paired add+norm path without replacing family-specific submodules."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from miles_plugins.top.residual_norm import ResidualTransformerLayer, deferred_residual_add

    if spec.module not in (TransformerLayer, ResidualTransformerLayer):
        raise NotImplementedError(f"[top] residual binding requires stock TransformerLayer, found {spec.module!r}")
    _bind_roles(spec, _NORM_ROLES, TopRMSNorm, stamp, as_spec_param=True)
    for role in ("input_layernorm", "pre_mlp_layernorm"):
        norm = getattr(spec.submodules, role, None)
        if not isinstance(norm, ModuleSpec) or norm.module is not TopRMSNorm:
            raise NotImplementedError(f"[top] paired residual layer requires an exposed {role}")
    spec.module = ResidualTransformerLayer
    spec.submodules.self_attn_bda = deferred_residual_add
    spec.submodules.mlp_bda = deferred_residual_add
    stamp.append(("residual_add", "BF16 add", "packed branch/residual -> stock fused norm"))


def _existing_server_args():
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()
    except Exception:
        return None


def _pin_sglang(args) -> str:
    """Pin the trainer's sglang ops to the rollout's program (dispatch is process-global)."""
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    contract = getattr(args, "true_on_policy_contract", None) or TRUE_ON_POLICY_V1_SCHEMA.name
    existing = _existing_server_args()
    if existing is not None:
        found = getattr(existing, "true_on_policy_contract", None)
        if found != contract:
            raise RuntimeError(
                f"[top] sglang already pinned to {found!r}, program wants {contract!r}"
            )
        return contract
    # no tp_size: the program must not vary with topology
    set_global_server_args_for_scheduler(
        ServerArgs(model_path=args.hf_checkpoint, enable_deterministic_inference=True,
                   true_on_policy_contract=contract)
    )
    return contract


def _enable_batch_invariant(program) -> bool:
    """Enable megatron's BIK; miles never calls initialize_megatron, so the flag alone is inert."""
    if program.ops.get("matmul") != "batch_invariant":
        return False
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()
    return is_batch_invariant_mode_enabled()


def _resolved_policy() -> str:
    """What the kernels dispatch to; a contract name only says what was requested."""
    try:
        from sglang.srt.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

        sa = _existing_server_args()
        p = resolve_true_on_policy_runtime_policy(sa)
        return (f"tp_size={getattr(sa, 'tp_size', None)} "
                f"row_linear_inv={p.tp_invariant_row_linear} "
                f"tree_all_reduce={p.deterministic_tree_all_reduce}")
    except Exception as e:  # a stamp must never break a run
        return f"unresolved({type(e).__name__})"


def _stamp(args, config, contract, bik, binds) -> None:
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    print(f"[top] program={_ACTIVE_PROGRAM.name} sglang_pinned={contract} "
          f"transformer_impl={getattr(args, 'transformer_impl', None)} "
          f"tp={config.tensor_model_parallel_size} batch_invariant_applied={bik} "
          f"installed={_installed()} {_resolved_policy()}", flush=True)
    for role, was, now in binds:
        print(f"[top] bind {role}: {was} -> {now}", flush=True)


def _hf_model_type(args) -> str | None:
    """Read the actual model family from the checkpoint configuration."""
    ckpt = getattr(args, "hf_checkpoint", None)
    if not ckpt:
        return None
    from miles.utils.hf_config import load_hf_config

    return getattr(load_hf_config(ckpt), "model_type", None)


def _validate_parallel_scope(args, config) -> None:
    """Refuse an unsupported topology once, before the model is built.

    The delegated row linear has no sequence-parallel implementation, and the per-call refusal in
    `_forward_impl` fires mid-forward -- after construction, after the checkpoint load, after the
    rollout has started. `config.sequence_parallel` is known here, so the same refusal costs
    seconds instead of minutes and names the flag rather than a kernel argument.

    The per-call check stays: this one reads the config, that one reads what megatron actually
    passed, and they can disagree.
    """
    if getattr(config, "sequence_parallel", False):
        raise NotImplementedError(
            "[top] sequence parallelism is not a declared program: the delegated row linear has "
            "no sequence-parallel implementation. Drop --sequence-parallel."
        )


def get_top_spec(args, config, vp_stage):
    """--spec entry point."""
    global _ACTIVE_PROGRAM
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )

    _validate_parallel_scope(args, config)
    contract = _pin_sglang(args)
    from miles_plugins.top.residual_norm import validate_residual_config

    use_te = getattr(args, "transformer_impl", "local") == "transformer_engine"
    validate_residual_config(config, use_te=use_te)
    if getattr(args, "num_experts", None) or _hf_model_type(args) != "qwen3":
        raise NotImplementedError("[top] this program supports Qwen3 dense only")
    _ACTIVE_PROGRAM = QWEN3_DENSE_V1

    from miles_plugins.top.rope import enforce_fused_rope

    enforce_fused_rope(args, config)

    spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
        normalization=args.normalization,
        use_kitchen=config.use_kitchen,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )

    stamp: list[tuple[str, str, str]] = [("rope", "configurable dispatch", "TE fused only (enforced)")]
    _bind_residual_layer(spec, stamp)
    _bind_roles(spec, _ROW_LINEAR_ROLES, TopRowParallelLinear._build(), stamp, as_spec_param=True)
    if _ACTIVE_PROGRAM.ops.get("attention") == "delegate":
        from miles_plugins.top.attention import TopAttention

        _bind_roles(spec, _ATTENTION_ROLES, TopAttention, stamp)
    bik = _enable_batch_invariant(_ACTIVE_PROGRAM)

    from miles_plugins.top.install import install_tree_tp_reduce

    install_tree_tp_reduce()

    if not stamp:
        raise RuntimeError("[top] no declared role found in the spec")

    block = TransformerBlockSubmodules(
        layer_specs=[spec] * get_num_layers_to_build(config, vp_stage=vp_stage),
        layer_norm=ModuleSpec(module=TopRMSNorm, params={"role": "final_layernorm"}),
    )
    stamp.append(("final_layernorm", "LayerNormImpl", TopRMSNorm.__name__))

    _stamp(args, config, contract, bik, stamp)
    return block
