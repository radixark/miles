"""Megatron's mHC hyper connection on the Cake four-stream kernels of the miles-ext-kernels submodule.

``--mhc-backend cake`` (with ``--use-miles-ext-kernels``) builds :class:`CakeHyperConnection` in place of Megatron's
``HyperConnectionModule`` in every layer of the GLM-5.3 (``get_glm5_next_spec``) and DeepSeek-V4 (``get_dsv4_spec``)
specs, and :class:`CakeMeanOutputContraction` in place of GLM-5.3's stream-mean head (:func:`apply_mhc_backend`). The
subclass keeps Megatron's parameters, their initialisation and their names, so checkpoints load unchanged. Its forward
runs the mapping GEMM, the coefficients and the stream collapse as one Cake prepare call, and its
``fused_h_res_h_post_bda`` the residual mixing as one finish call (``miles_ext_kernels.mhc``); the backward of the pair
returns one single-cast ``dX``. ``--cake-mhc-variant`` picks the parameter-gradient summation: ``det`` (fixed order) or
``fast`` (FP32 atomics in ``dfn``).

Numerics against Megatron's module: the coefficients stay FP32, where Megatron casts them to the activation dtype
before it applies them. GLM-5.3's patched RMS normalisation, ``rsqrt(mean(x^2) + eps)`` with the model's
``layernorm_epsilon``, is the Cake function. DeepSeek-V4 keeps Megatron's ``1 / (rms + 1e-6)``, which the kernels
compute as ``rsqrt(mean(x^2) + 1e-6)`` (another epsilon placement), and their mapping GEMM is FP32-accurate where
Megatron's FP32 matmul may run in TF32.

Not supported, rejected at startup (:func:`validate_mhc_backend`, :func:`apply_mhc_backend`) or by the call: the mHC
recompute arena (``mhc`` in ``--recompute-modules``) and CUDA graphs, Megatron's fused mHC kernels, an FP32 residual
stream, hidden dropout, a branch-output bias, MoE expert-parallel communication overlap, and stream counts or hidden
sizes the kernels are not built for.

CUDA-only (NVIDIA SM100a / SM103a, Megatron backend): startup validation imports this module only under
``--mhc-backend cake``, which it ties to ``--train-backend megatron``; the two specs import it to call
:func:`apply_mhc_backend`, which leaves every other backend's spec untouched.
"""

from __future__ import annotations

import argparse
import dataclasses

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec

NUM_STREAMS = 4
# The --spec pairs whose layers build Megatron's HyperConnectionModule and pass the call contract checked here.
SUPPORTED_SPECS = (
    ("miles_plugins.models.glm5_next.glm5_next", "get_glm5_next_spec"),
    ("miles_plugins.models.deepseek_v4.deepseek_v4", "get_dsv4_spec"),
)
# The kernels read the mapping matrix and the logit offsets in place, with 256-bit loads.
PARAMETER_ALIGNMENT_BYTES = 32


def mhc_kernels():
    """The ``miles_ext_kernels.mhc`` kernel package."""
    try:
        from miles_ext_kernels import mhc
    except ImportError as exc:
        raise ImportError(
            "--mhc-backend cake runs miles_ext_kernels.mhc from the miles-ext-kernels submodule: pass "
            "--use-miles-ext-kernels and check the submodule out with "
            "'git submodule update --init 3rdparty/miles-ext-kernels'."
        ) from exc
    return mhc


class CakeHyperConnection(HyperConnectionModule):
    """:class:`HyperConnectionModule` whose ``forward`` and ``fused_h_res_h_post_bda`` run the Cake mHC kernels.

    ``forward`` returns ``(U, handle, None, hidden_states)`` in the slots of ``(aggregated, h_res, h_post, residual)``:
    Megatron's layer hands the prepare ``handle`` and the residual back to ``fused_h_res_h_post_bda``, which mixes the
    branch output into the four streams. Construction compiles the kernels of the layer's hidden size ahead of the
    first step (``miles_ext_kernels.mhc.prebuild``).
    """

    def __init__(self, config, layer_number: int):
        super().__init__(config, layer_number)
        if self.n != NUM_STREAMS:
            raise ValueError(f"--mhc-backend cake runs {NUM_STREAMS} residual streams, not {self.n}")
        if self.compute_h_eps != self.sinkhorn_eps:
            raise ValueError(
                f"--mhc-backend cake takes one coefficient epsilon; Megatron's are {self.compute_h_eps} (pre) and "
                f"{self.sinkhorn_eps} (Sinkhorn)"
            )
        mhc_kernels().prebuild(
            torch.accelerator.current_device_index(), self.hidden_size, sinkhorn_iters=self.sinkhorn_iterations
        )

    def forward(self, hidden_states, mhc_recompute_manager=None, output_slot=None):
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError(
                "--mhc-backend cake does not run the mHC recompute arena ('mhc' in --recompute-modules) or its "
                "CUDA-graph output slots"
            )
        fn, base = self.mapping_proj.weight, self.bias
        for name, parameter in (("mapping_proj.weight", fn), ("bias", base)):
            offset = parameter.data_ptr() % PARAMETER_ALIGNMENT_BYTES
            if offset:
                raise ValueError(
                    f"--mhc-backend cake reads the hyper-connection parameter {name} in place and needs it "
                    f"{PARAMETER_ALIGNMENT_BYTES}-byte aligned, but it starts {offset} bytes past an aligned address "
                    "(a view into a parameter buffer; Megatron's DDP buffers pad every parameter start only under "
                    "--use-distributed-optimizer)"
                )
        s, b, _ = hidden_states.shape
        scale = torch.cat([self.alpha_pre, self.alpha_post, self.alpha_res])
        U, handle = mhc_kernels().mhc_prepare(
            hidden_states.view(s * b, self.n, self.hidden_size),
            fn,
            scale,
            base,
            rms_eps=self.norm_eps,
            hc_eps=self.sinkhorn_eps,
            sinkhorn_iters=self.sinkhorn_iterations,
        )
        return U.view(s, b, self.hidden_size), handle, None, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob,
        training,
        fused,
        manager=None,
    ):
        """``h_res`` is the prepare handle of :meth:`forward` and ``h_post`` its ``None``."""
        x, bias = layer_output_with_bias
        if manager is not None:
            raise NotImplementedError("--mhc-backend cake does not run the mHC recompute arena")
        if bias is not None:
            raise NotImplementedError("--mhc-backend cake mixes a branch output without a separate bias")
        if training and dropout_prob > 0:
            raise NotImplementedError("--mhc-backend cake runs without hidden dropout (--hidden-dropout 0)")
        s, b, _ = original_residual.shape
        Y = mhc_kernels().mhc_finish(
            original_residual.view(s * b, self.n, self.hidden_size), x.view(s * b, self.hidden_size), h_res
        )
        return Y.view(s, b, self.n * self.hidden_size)


class CakeMeanOutputContraction(MegatronModule):
    """The plain mean of the block output's streams (GLM-5.3's head) on the Cake head kernels: bit for bit
    ``HyperConnectionModule.output_contract`` and its gradient. Construction compiles its kernels."""

    def __init__(self, config):
        super().__init__(config)
        self.num_streams = config.num_residual_streams
        mhc_kernels().prebuild(
            torch.accelerator.current_device_index(),
            config.hidden_size,
            sinkhorn_iters=config.mhc_sinkhorn_iterations,
            head=True,
        )

    def forward(self, hidden_states):
        s, b, width = hidden_states.shape
        hidden = width // self.num_streams
        Y = mhc_kernels().mhc_head(hidden_states.view(s * b, self.num_streams, hidden))
        return Y.view(s, b, hidden)


def _cake_connection(spec):
    """The Cake counterpart of one layer's hyper-connection submodule spec."""
    module = spec.module if isinstance(spec, ModuleSpec) else spec
    if module is CakeHyperConnection or module is IdentityOp:
        return spec
    if module is not HyperConnectionModule:
        raise NotImplementedError(f"--mhc-backend cake replaces Megatron's HyperConnectionModule, not {module}")
    if isinstance(spec, ModuleSpec):
        return dataclasses.replace(spec, module=CakeHyperConnection)
    return CakeHyperConnection


def apply_mhc_backend(block_spec, args, config, *, mean_head: bool = False):
    """Under ``--mhc-backend cake``: :class:`CakeHyperConnection` for both hyper connections of every layer of
    ``block_spec`` (a ``TransformerBlockSubmodules``) and, with ``mean_head`` (a model whose head is the plain stream
    mean), :class:`CakeMeanOutputContraction` for its ``hc_head_contraction``. Returns ``block_spec``; other backends
    leave it untouched.

    Checks the model's connection shape and the device, and hands the kernel package its parsed arguments
    (``args.miles_ext_kernels_args``).
    """
    if args.mhc_backend != "cake":
        return block_spec
    if config.num_residual_streams != NUM_STREAMS:
        raise ValueError(f"--mhc-backend cake runs {NUM_STREAMS} residual streams, not {config.num_residual_streams}")
    if config.use_fused_mhc:
        raise ValueError("--mhc-backend cake replaces the mHC computation; it does not combine with --use-fused-mhc")
    mhc = mhc_kernels()
    device = torch.accelerator.current_device_index()
    if not mhc.is_supported(device, config.hidden_size):
        raise ValueError(
            f"--mhc-backend cake runs the hidden sizes {mhc.HIDDEN_SIZES} on NVIDIA SM100a / SM103a (Blackwell); this "
            f"model has hidden size {config.hidden_size} on a device of compute capability "
            f"{torch.cuda.get_device_capability(device)}"
        )
    mhc.configure(argparse.Namespace(**(args.miles_ext_kernels_args or {})))
    for layer_spec in block_spec.layer_specs:
        submodules = layer_spec.submodules
        submodules.self_attention_hyper_connection = _cake_connection(submodules.self_attention_hyper_connection)
        submodules.mlp_hyper_connection = _cake_connection(submodules.mlp_hyper_connection)
    if mean_head:
        block_spec.hc_head_contraction = ModuleSpec(module=CakeMeanOutputContraction)
    return block_spec


def validate_mhc_backend(args) -> None:
    """Reject at startup what :class:`CakeHyperConnection` does not run, for ``--mhc-backend cake``; ``args`` is the
    parsed namespace miles validates (Megatron's arguments included)."""
    if args.mhc_backend != "cake":
        return
    if args.train_backend != "megatron":
        raise ValueError(
            "--mhc-backend cake replaces Megatron's hyper-connection module: it needs --train-backend megatron"
        )
    spec = tuple(args.spec or ())
    if spec not in SUPPORTED_SPECS:
        supported = ", ".join(" ".join(pair) for pair in SUPPORTED_SPECS)
        raise ValueError(f"--mhc-backend cake supports --spec {supported}; got --spec {' '.join(spec) or '(none)'}")
    if args.recompute_modules and "mhc" in args.recompute_modules:
        raise ValueError(
            "--mhc-backend cake does not run the mHC recompute arena: drop 'mhc' from --recompute-modules"
        )
    if args.cuda_graph_impl != "none":
        raise ValueError(f"--mhc-backend cake runs without CUDA graphs, not --cuda-graph-impl {args.cuda_graph_impl}")
    if args.use_fused_mhc:
        raise ValueError("--mhc-backend cake replaces the mHC computation; it does not combine with --use-fused-mhc")
    if args.fp32_residual_connection:
        raise ValueError("--mhc-backend cake runs BF16 residual streams, not --fp32-residual-connection")
    if args.hidden_dropout:
        raise ValueError(
            f"--mhc-backend cake runs without hidden dropout: pass --hidden-dropout 0, not {args.hidden_dropout}"
        )
    if args.overlap_moe_expert_parallel_comm:
        raise ValueError(
            "--mhc-backend cake hands the residual mixing a prepare handle, which the MoE expert-parallel "
            "communication overlap cannot carry between its stages: drop --overlap-moe-expert-parallel-comm"
        )
