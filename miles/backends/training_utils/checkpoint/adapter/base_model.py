"""Whether a source adapter was trained on this run's base model, judged from both sides' HF files.

The fingerprint hashes every ``*norm.weight`` tensor: every family has them, they are small, and any full fine-tune
of the base changes them. Weights are read from the HF files, never from a loaded (possibly quantized) model.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from safetensors import safe_open


def is_fingerprint_tensor(name: str) -> bool:
    return name.endswith("norm.weight")


def fingerprint_named_tensors(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    names = []
    for name, tensor in sorted(named, key=lambda item: item[0]):
        names.append(name)
        digest.update(name.encode())
        # fp32 is exact for every base dtype, so a trainer's fp32 copy of a bf16 base hashes the same
        digest.update(tensor.float().contiguous().numpy().tobytes())
    assert names, "no *norm.weight tensors to fingerprint"
    return digest.hexdigest()


def hf_base_fingerprint(hf_dir: Path) -> str:
    index = hf_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else []
    files = files or sorted(p.name for p in hf_dir.glob("*.safetensors"))
    assert files, f"{hf_dir} has no safetensors weights"
    named = []
    for file in files:
        with safe_open(str(hf_dir / file), "pt") as f:
            named += [(name, f.get_tensor(name)) for name in f.keys() if is_fingerprint_tensor(name)]
    return fingerprint_named_tensors(named)


def resolve_hf_dir(name_or_path: str) -> Path | None:
    """A local directory, or a hub id already in the local cache; None when neither (nothing is downloaded)."""
    if Path(name_or_path).is_dir():
        return Path(name_or_path)
    try:
        return Path(snapshot_download(name_or_path, local_files_only=True))
    except (LocalEntryNotFoundError, HFValidationError):
        return None


def assert_same_base_model(source: str, candidates: Iterable[str], run_hf_dir: str) -> None:
    """``candidates``: what the source records as its base, best first; the first resolvable one is compared."""
    candidates = [c for c in candidates if c]
    hf_dir = next((d for c in candidates if (d := resolve_hf_dir(c)) is not None), None)
    assert hf_dir is not None, (
        f"{source}'s base model ({candidates or 'not recorded'}) has no local HF files to compare with "
        f"--hf-checkpoint {run_hf_dir}"
    )
    assert hf_base_fingerprint(hf_dir) == hf_base_fingerprint(
        Path(run_hf_dir)
    ), f"{source} was trained on {hf_dir}, whose weights differ from --hf-checkpoint {run_hf_dir}"
