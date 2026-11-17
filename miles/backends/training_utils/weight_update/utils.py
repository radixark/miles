from miles.backends.training_utils.weight_update.checksum_utils import hash_tensor_sha256


def record_lora_checksums(bucket, checksums) -> None:
    """Accumulate the sha256 checksums the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        checksums[lora_name][hf_key] = hash_tensor_sha256(tensor)
