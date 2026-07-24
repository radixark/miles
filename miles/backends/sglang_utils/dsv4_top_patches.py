"""Narrow SGLang compatibility patches required by the DSV4 TOP contract.

These patches run in the spawned SGLang server process. Keep them small,
version-tolerant, and guarded by an explicit Miles environment variable.
"""

from __future__ import annotations

from functools import wraps
import os
from pathlib import Path


def _patch_hash_topk_fp32_input() -> None:
    """Feed the NVIDIA DSV4 hash-topk kernel its required FP32 logits.

    SGLang's ``HashTopK.forward`` must retain the original router-logit tensor
    in ``StandardTopKOutput``. Patching the imported JIT function, instead of
    the module forward, preserves that behavior while fixing the kernel input.
    """
    import sglang.jit_kernel.dsv4 as dsv4_kernels

    original = dsv4_kernels.hash_topk
    if getattr(original, "_miles_dsv4_top_fp32_input", False):
        return

    @wraps(original)
    def hash_topk_fp32_input(*args, **kwargs):
        if "router_logits" in kwargs:
            kwargs = dict(kwargs)
            kwargs["router_logits"] = kwargs["router_logits"].float()
        else:
            args = list(args)
            args[0] = args[0].float()
        return original(*args, **kwargs)

    hash_topk_fp32_input._miles_dsv4_top_fp32_input = True
    dsv4_kernels.hash_topk = hash_topk_fp32_input


def _activate_reference_source_patches() -> None:
    """Load the version-pinned DSV4 TOP SGLang arithmetic contract.

    SGLang normally enters its source-patcher path only when tensor dumping is
    enabled. TOP needs the patcher without enabling tensor collection, so make
    a configured source patch sufficient to enter that path. The patch engine
    itself is fail-closed: any source mismatch raises during model startup.
    """
    config_path = Path(__file__).with_name("dsv4_top_sglang_reference.yaml")
    if not config_path.is_file():
        raise RuntimeError(f"Missing DSV4 TOP SGLang contract: {config_path}")

    configured_path = os.environ.get("DUMPER_SOURCE_PATCHER_CONFIG")
    if configured_path and Path(configured_path).resolve() != config_path.resolve():
        if (
            os.environ.get(
                "MILES_DSV4_TOP_ALLOW_SOURCE_PATCH_OVERRIDE",
                "0",
            )
            != "1"
        ):
            raise RuntimeError(
                "DSV4 TOP cannot replace its inference contract with "
                f"DUMPER_SOURCE_PATCHER_CONFIG={configured_path}; set "
                "MILES_DSV4_TOP_ALLOW_SOURCE_PATCH_OVERRIDE=1 only for "
                "controlled compatibility ablations"
            )
    else:
        os.environ["DUMPER_SOURCE_PATCHER_CONFIG"] = str(config_path)

    from sglang.srt.debug_utils.dumper import _Dumper

    may_enable = _Dumper.may_enable
    if getattr(may_enable.fget, "_miles_dsv4_top_source_patches", False):
        return

    original_getter = may_enable.fget

    def may_enable_source_patches(self) -> bool:
        return bool(original_getter(self) or self._config.source_patcher_config is not None)

    may_enable_source_patches._miles_dsv4_top_source_patches = True
    _Dumper.may_enable = property(may_enable_source_patches)


def apply_dsv4_top_sglang_patches() -> None:
    """Apply the inference-side compatibility portion of DSV4 TOP."""
    source_contract = os.environ.get("MILES_DSV4_TOP_SOURCE_CONTRACT", "1")
    if source_contract not in ("0", "1"):
        raise RuntimeError("MILES_DSV4_TOP_SOURCE_CONTRACT must be 0 or 1, " f"got {source_contract!r}")
    if source_contract == "1":
        _activate_reference_source_patches()
    _patch_hash_topk_fp32_input()
