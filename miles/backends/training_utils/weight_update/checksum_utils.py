import hashlib

import torch


def hash_tensor_sha256(tensor: torch.Tensor) -> str:
    """Real (cryptographic) hash: a mismatch here has to mean a bug."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()
