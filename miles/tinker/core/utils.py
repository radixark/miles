import hashlib
import json
import os
import re
from pathlib import Path

from miles.tinker.core.types import GatewayConfig, ModelRecord, OwnershipError, UserInputError


def parse_tinker_path(path: str) -> tuple[str, str, str]:
    if not path.startswith("tinker://"):
        raise UserInputError(f"not a tinker path: {path!r}")
    parts = path.removeprefix("tinker://").split("/")
    if len(parts) != 3 or parts[1] not in ("weights", "sampler_weights"):
        raise UserInputError(f"malformed tinker path: {path!r}")
    for segment in parts:
        validate_checkpoint_segment(segment)
    return parts[0], parts[1], parts[2]


def validate_checkpoint_segment(segment: str) -> None:
    """Reject client path segments that could escape the checkpoint root."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", segment) is None:
        raise UserInputError(f"invalid checkpoint path segment {segment!r}")


def resolve_checkpoint_dir(checkpoint_root: str, model_id: str, kind: str, name: str) -> str:
    root = os.path.realpath(checkpoint_root)
    path = os.path.realpath(f"{root}/{model_id}/{kind}/{name}")
    assert path.startswith(root + os.sep), f"checkpoint path {path!r} escapes {root!r}"
    return path


def build_checkpoint_metadata(record: ModelRecord, config: GatewayConfig) -> dict:
    return {
        # the digest proves ownership without persisting the bearer credential itself
        "tenant_digest": _tenant_digest(record.tenant),
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }


def read_checkpoint_metadata(checkpoint_dir: str, tenant: str, shown_path: str) -> dict:
    meta_file = Path(checkpoint_dir) / "META.json"
    if not meta_file.exists():
        raise UserInputError(f"unknown checkpoint {shown_path!r}")
    try:
        meta = json.loads(meta_file.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UserInputError(f"cannot read checkpoint {shown_path!r}: {error}") from error
    if meta["tenant_digest"] != _tenant_digest(tenant):
        raise OwnershipError(f"checkpoint {shown_path!r} does not belong to this tenant")
    return meta


def validate_checkpoint_compatibility(meta: dict, record: ModelRecord, config: GatewayConfig, shown_path: str) -> None:
    """Reject settings that would change the saved tensors' meaning."""
    expected = {
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise UserInputError(
                f"checkpoint {shown_path!r} was saved with {key}={meta[key]!r}; this model expects {key}={value!r}"
            )


def resolve_sampler_checkpoint(checkpoint_root: str, tenant: str, model_path: str, base_model: str) -> tuple[str, str]:
    """Return the adapter name and directory so engines can reload evicted snapshots."""
    model_id, kind, name = parse_tinker_path(model_path)
    if kind != "sampler_weights":
        raise UserInputError(f"cannot sample from {model_path!r}: not a sampler_weights path")
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_root, model_id, "sampler_weights", name)
    meta = read_checkpoint_metadata(checkpoint_dir, tenant, model_path)
    if meta["base_model"] != base_model:
        raise UserInputError(
            f"checkpoint {model_path!r} uses base_model={meta['base_model']!r}; this server serves {base_model!r}"
        )
    return f"{model_id}@{name}", checkpoint_dir


def _tenant_digest(tenant: str) -> str:
    return hashlib.sha256(tenant.encode()).hexdigest()
