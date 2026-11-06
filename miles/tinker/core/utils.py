import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from miles.tinker.core.input_validation import validate_checkpoint_metadata, validate_checkpoint_segment
from miles.tinker.core.types import GatewayConfig, ModelRecord, OwnershipError, UserInputError


CHECKPOINT_KINDS = ("weights", "sampler_weights", "external_weights")


def parse_tinker_path(path: str) -> tuple[str, str, str]:
    if not path.startswith("tinker://"):
        raise UserInputError(f"not a tinker path: {path!r}")
    parts = path.removeprefix("tinker://").split("/")
    if len(parts) != 3 or parts[1] not in CHECKPOINT_KINDS:
        raise UserInputError(f"malformed tinker path: {path!r}")
    for segment in parts:
        validate_checkpoint_segment(segment)
    return parts[0], parts[1], parts[2]


def resolve_checkpoint_dir(checkpoint_root: str, model_id: str, kind: str, name: str) -> str:
    root = os.path.realpath(checkpoint_root)
    path = f"{root}/{model_id}/{kind}/{name}"
    resolved = os.path.realpath(path)
    assert resolved.startswith(root + os.sep), f"checkpoint path {resolved!r} escapes {root!r}"
    return path


def build_parametrization_fields(record: ModelRecord, config: GatewayConfig) -> dict:
    """The parametrization fields the tenant record and the gateway own; targets come from the model."""
    return {
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "lora_type": config.lora_type,
        "experts_shared_outer_loras": config.experts_shared_outer_loras,
        "base_model_fingerprint": config.base_model_fingerprint,
    }


# a checkpoint saved before commits: its owner and settings, written last
PRE_COMMIT_META_FILENAME = "META.json"


def read_checkpoint_metadata(checkpoint_dir: str, tenant: str, shown_path: str) -> dict:
    meta_file = Path(checkpoint_dir) / PRE_COMMIT_META_FILENAME
    if not meta_file.exists():
        raise UserInputError(f"unknown checkpoint {shown_path!r}")
    try:
        meta = json.loads(meta_file.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserInputError(f"cannot read checkpoint {shown_path!r}: {error}") from error
    validate_checkpoint_metadata(meta, shown_path)
    if meta["tenant_digest"] != tenant_digest(tenant):
        raise OwnershipError(f"checkpoint {shown_path!r} does not belong to this tenant")
    return meta


def resolve_sampler_export(
    checkpoint_dir: str,
    tenant: str,
    model_id: str,
    shown_path: str,
    resolve_adapter: Callable[[str, str, str, str], dict | None],
) -> dict | None:
    """The committed export behind a sampler name, or None for a dir written before commits; a link whose target is
    gone is an unknown checkpoint."""
    if not os.path.islink(checkpoint_dir):
        return None
    committed = resolve_adapter(os.path.dirname(checkpoint_dir), os.path.basename(checkpoint_dir), tenant, model_id)
    if committed is None:
        raise UserInputError(f"unknown checkpoint {shown_path!r}")
    return committed


def resolve_sampler_checkpoint(
    config: GatewayConfig, tenant: str, model_path: str, resolve_adapter: Callable[[str, str, str, str], dict | None]
) -> tuple[str, str]:
    """Return the adapter name and directory so engines can reload evicted snapshots."""
    model_id, kind, name = parse_tinker_path(model_path)
    if kind != "sampler_weights":
        raise UserInputError(
            f"sampling from training checkpoints is not supported: {model_path!r}; use save_weights_for_sampler()"
        )
    checkpoint_dir = resolve_checkpoint_dir(config.checkpoint_root, model_id, "sampler_weights", name)
    committed = resolve_sampler_export(checkpoint_dir, tenant, model_id, model_path, resolve_adapter)
    if committed is None:
        # written before commits existed: a plain directory with META.json, served as it is
        meta = {"experts_shared_outer_loras": False, **read_checkpoint_metadata(checkpoint_dir, tenant, model_path)}
        lora_name = f"{model_id}@{name}"
    else:
        meta, checkpoint_dir, lora_name = committed["parametrization"], committed["path"], committed["lora_name"]
    # the engines serve one base model and one expert layout
    served = {"base_model": config.base_model, "experts_shared_outer_loras": config.experts_shared_outer_loras}
    for key, value in served.items():
        if meta[key] != value:
            raise UserInputError(f"checkpoint {model_path!r} uses {key}={meta[key]!r}; this server serves {value!r}")
    return lora_name, checkpoint_dir


def tenant_digest(tenant: str) -> str:
    return hashlib.sha256(tenant.encode()).hexdigest()
