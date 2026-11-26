"""Portable local files for tensor exchange between different Torch runtimes."""

from pathlib import Path
from tempfile import TemporaryDirectory

from safetensors.torch import save_file


def serialize_tensor_bucket(named_tensors):
    """Return the wire payload and its lifetime owner; release it only after acknowledgement."""
    directory = TemporaryDirectory(prefix="miles-lmdeploy-weights-")
    path = Path(directory.name) / "bucket.safetensors"
    save_file(
        {name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in named_tensors},
        str(path),
    )
    return [str(path)], directory
