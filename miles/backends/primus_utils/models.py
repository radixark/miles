"""Model families whose Megatron implementation lives in Primus rather than in Megatron.

Families such as GLM-5.3 (`model_type: glm5_next`) are not `GPTModel` + layer spec: Primus
ships the model class, its config dataclass and an online HF loader. Miles builds them
through Primus' own provider and loads the released HF checkpoint with Primus' loader, so
no Megatron-Bridge mapping or offline torch_dist conversion is involved.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)

PRIMUS_MODEL_PROVIDER_PATH = "miles.backends.primus_utils.models.primus_model_provider"

# model_type -> (HF loader "module:function", Miles megatron_to_hf converter name)
_PRIMUS_MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "glm5_next": (
        "primus.backends.megatron.core.models.glm5_next.glm5_next_hf_loader:load_glm5_next_hf_checkpoint",
        "primus_glm5_next",
    ),
}


def primus_model_type(args: argparse.Namespace | Any) -> str | None:
    """The Primus `model_type` when it names a Primus-built family, else None.

    `args` is either the Megatron namespace or a trainer config whose `backend` holds it.
    """
    args = getattr(args, "backend", args)  # config-access-exempt: trainer configs nest the Megatron namespace
    if not getattr(args, "primus_config", None):  # config-access-exempt: absent without the Primus flags
        return None
    model_type = (
        getattr(args, "primus_params", None) or {}
    ).get(  # config-access-exempt: set only by PrimusConfigBridge.attach
        "model_type"
    )
    return model_type if model_type in _PRIMUS_MODEL_FAMILIES else None


def primus_converter_name(model_type: str) -> str:
    return _PRIMUS_MODEL_FAMILIES[model_type][1]


def primus_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    """Build the model with Primus' provider for the configured `model_type`.

    Primus' builders read `megatron.training.get_args()`, which `megatron_init` points at a
    view holding both Megatron's arguments and the Primus-only fields (hc_mult, index_topk, ...).
    The family is read from the Primus parameters: Megatron's `get_model` overwrites
    `args.model_type` with its own `ModelType` enum before calling the provider.
    """
    from megatron.training import get_args
    from primus.core.utils.import_utils import get_model_provider

    model_type = primus_model_type(get_args())
    assert model_type is not None, "primus_model_provider needs a Primus-built model_type in --primus-config"
    provider = get_model_provider(model_type)
    return provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


def load_primus_hf_checkpoint(ddp_model, args: argparse.Namespace | Any, load_path: str) -> bool:
    """Load an HF checkpoint with Primus' online loader. False if the model is not Primus-built."""
    model_type = primus_model_type(args)
    if model_type is None:
        return False

    import importlib

    module_name, func_name = _PRIMUS_MODEL_FAMILIES[model_type][0].split(":")
    loader = getattr(importlib.import_module(module_name), func_name)  # config-access-exempt: loader named by the family table
    # Primus-Turbo grouped GEMM registers its per-expert views lazily; the loader writes through them.
    for chunk in ddp_model:
        for module in chunk.modules():
            ensure_weight_views = getattr(module, "_ensure_weight_views", None)  # config-access-exempt: only Primus-Turbo grouped linears define it
            if callable(ensure_weight_views):
                ensure_weight_views()
    num_tensors = loader(ddp_model, load_path)
    logger.info("Loaded %d tensors from HF checkpoint %s with Primus' %s loader", num_tensors, load_path, model_type)
    return True
