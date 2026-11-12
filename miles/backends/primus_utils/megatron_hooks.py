"""Primus hooks into Miles' Megatron backend, installed from `primus_megatron_init`.

They wrap module-level functions of `miles.backends.megatron_utils` that their own module
calls by global name, so the wrappers take effect without editing that package:

- `checkpoint._load_checkpoint_hf`: Primus-built families (GLM-5.3) load the HF checkpoint
  with Primus' online loader instead of mbridge / Megatron-Bridge.
- `megatron_to_hf._convert_to_hf_core`: Primus-built families convert Megatron names to HF
  with the converter `PrimusConfigBridge` selects through `--model-name`.
- `named_weights._named_params_and_buffers_{vanilla,global}`: Primus-Turbo grouped GEMM
  keeps every expert in one packed `weights` Parameter and exposes TE-style `weight{i}`
  views of it, registered lazily on first forward. Only the views carry names the HF
  converters understand; the packed tensor would be a second copy of the same data.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence

import torch

_installed = False


def install_megatron_hooks() -> None:
    global _installed
    if _installed:
        return
    from miles.backends.megatron_utils import checkpoint, megatron_to_hf, named_weights

    checkpoint._load_checkpoint_hf = _wrap_load_checkpoint_hf(checkpoint._load_checkpoint_hf)
    megatron_to_hf._convert_to_hf_core = _wrap_convert_to_hf_core(megatron_to_hf._convert_to_hf_core)
    named_weights._named_params_and_buffers_vanilla = _wrap_named_params(
        named_weights._named_params_and_buffers_vanilla, lambda model: model
    )
    named_weights._named_params_and_buffers_global = _wrap_named_params(
        named_weights._named_params_and_buffers_global, lambda args, model: model
    )
    _installed = True


def ensure_grouped_weight_views(model: Sequence[torch.nn.Module]) -> None:
    for chunk in model:
        for module in chunk.modules():
            ensure_weight_views = getattr(module, "_ensure_weight_views", None)  # config-access-exempt: only Primus-Turbo grouped linears define it
            if callable(ensure_weight_views):
                ensure_weight_views()


def is_packed_expert_weights(name: str) -> bool:
    return name.endswith(".weights") and ".experts." in name


def _wrap_load_checkpoint_hf(original):
    @functools.wraps(original)
    def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
        from miles.backends.primus_utils.models import load_primus_hf_checkpoint

        if not load_primus_hf_checkpoint(ddp_model, args, load_path):
            return original(ddp_model, optimizer, args, load_path)
        # The tail of the wrapped function: refresh the optimizer's copy, iteration 0.
        if (args.backend.fp16 or args.backend.bf16) and optimizer is not None:
            assert not args.backend.load_main_params_from_ckpt
            optimizer.reload_model_params()
        return 0, 0

    return _load_checkpoint_hf


def _wrap_convert_to_hf_core(original):
    @functools.wraps(original)
    def _convert_to_hf_core(args, model_name, name, param):
        if model_name.lower() != "primus_glm5_next":
            return original(args, model_name, name, param)
        from miles.backends.primus_utils.glm5_next_to_hf import convert_primus_glm5_next_to_hf

        return convert_primus_glm5_next_to_hf(args, name, param)

    return _convert_to_hf_core


def _wrap_named_params(original, get_model):
    @functools.wraps(original)
    def _named_params_and_buffers(*args, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        ensure_grouped_weight_views(get_model(*args, **kwargs))
        for name, tensor in original(*args, **kwargs):
            if not is_packed_expert_weights(name):
                yield name, tensor

    return _named_params_and_buffers
