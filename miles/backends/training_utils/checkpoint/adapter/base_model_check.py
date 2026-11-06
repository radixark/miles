"""Whether a source adapter was trained on this run's base model, judged from both sides' HF files.

The check compares every norm weight and the first layer's attention weights that both sides hold unquantized:
every family has both, they are small, and a full fine-tune changes the norms while a merged LoRA changes the
projections. Weights are read from the HF files, never from a loaded (possibly quantized) model.

Not detected: a LoRA merged only into MLP or expert weights, or only into layers past the first; against an FP8 or
MXFP4 copy of the base, only the norms are left to compare. Layer 0's MLP is left out because every rank reads the
fingerprint, and a dense model's MLP (or a packed-experts tensor) is gigabytes.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from safetensors import safe_open


# HF names put layers at the root (DeepSeek-V4.1: layers.0.attn.wq_a) or under a prefix (Qwen: model.layers.0.self_attn)
_FIRST_LAYER_ATTENTION_WEIGHT = re.compile(r"(^|\.)layers\.0\.(.+\.)?[^.]*attn[^.]*\.(.+\.)?weight$")


def is_base_check_tensor(name: str) -> bool:
    return name.endswith("norm.weight") or _FIRST_LAYER_ATTENTION_WEIGHT.search(name) is not None


# safetensors dtype names; FP8 and the integer containers of MXFP4 / int quantization are left out
_UNQUANTIZED_DTYPES = {"F16", "BF16", "F32", "F64"}


def read_base_check_tensors(hf_dir: Path) -> dict[str, torch.Tensor]:
    """The fingerprint tensors an HF dir holds unquantized: a quantized copy of the base (FP8, MXFP4) compares by
    the tensors quantization leaves alone."""
    index = hf_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else []
    files = files or sorted(p.name for p in hf_dir.glob("*.safetensors"))
    assert files, f"{hf_dir} has no safetensors weights"
    tensors = {}
    for file in files:
        with safe_open(str(hf_dir / file), "pt") as f:
            for name in f.keys():
                if is_base_check_tensor(name) and f.get_slice(name).get_dtype() in _UNQUANTIZED_DTYPES:
                    tensors[name] = f.get_tensor(name)
    return tensors


def assert_same_base_tensors(
    source: str, source_tensors: dict[str, torch.Tensor], run_tensors: dict[str, torch.Tensor], run_base: str
) -> None:
    """Compared on the names both sides hold (a source may lack layers, e.g. MTP, or hold some quantized)."""
    common = sorted(source_tensors.keys() & run_tensors.keys())
    assert any(
        name.endswith("norm.weight") for name in common
    ), f"{source} and --hf-checkpoint {run_base} share no norm weights to compare their base models by"
    # fp32 is exact for every base dtype, so a trainer's fp32 copy of a bf16 base compares equal
    differing = [name for name in common if not torch.equal(source_tensors[name].float(), run_tensors[name].float())]
    assert (
        not differing
    ), f"{source} was trained on another base model than --hf-checkpoint {run_base}: e.g. {differing[:3]} differ"


def resolve_hf_dir(name_or_path: str) -> Path | None:
    """A local directory with weights, or a hub id whose weights are in the local cache; None when neither
    (nothing is downloaded)."""
    if Path(name_or_path).is_dir():
        hf_dir = Path(name_or_path)
    else:
        try:
            hf_dir = Path(snapshot_download(name_or_path, local_files_only=True))
        except (LocalEntryNotFoundError, HFValidationError):
            return None
    # a cache that holds only a config or tokenizer cannot fingerprint the base
    return hf_dir if any(hf_dir.glob("*.safetensors")) else None


def assert_same_base_model(source: str, candidates: Iterable[str], run_hf_checkpoint: str, *, remedy: str) -> None:
    """``candidates``: what the source records as its base, best first; the first resolvable one is compared.
    ``remedy``: how to give the source's base local HF files."""
    candidates = [c for c in candidates if c]
    hf_dir = next((d for c in candidates if (d := resolve_hf_dir(c)) is not None), None)
    assert hf_dir is not None, (
        f"{source}'s base model ({candidates or 'not recorded'}) has no local HF files to compare with "
        f"--hf-checkpoint {run_hf_checkpoint}: {remedy}"
    )
    run_dir = resolve_hf_dir(run_hf_checkpoint)
    assert run_dir is not None, f"--hf-checkpoint {run_hf_checkpoint} has no local HF weights to compare with"
    if hf_dir.resolve() == run_dir.resolve():
        return
    assert_same_base_tensors(
        source, read_base_check_tensors(hf_dir), read_base_check_tensors(run_dir), run_hf_checkpoint
    )
