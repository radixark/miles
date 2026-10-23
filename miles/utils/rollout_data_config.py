"""Training dataset configuration for prompt-level mixing."""

import re
from dataclasses import dataclass, fields, replace

from omegaconf import OmegaConf

from miles.utils.eval_config import DATASET_SAMPLE_SPECS, ensure_dataset_list
from miles.utils.file_arg_utils import resolve_file_arg

DEFAULT_DATA_SOURCE_PATH = "miles.rollout.data_source.RolloutDataSource"
MIXED_DATA_SOURCE_PATH = "miles.rollout.mixed_data_source.MixedRolloutDataSource"


@dataclass(frozen=True)
class RolloutDatasetConfig:
    name: str
    path: str
    samples_per_step: int
    input_key: str
    label_key: str | None
    tool_key: str | None
    metadata_key: str
    max_prompt_len: int | None
    sha256: str | None = None


def build_rollout_dataset_configs(args) -> tuple[RolloutDatasetConfig, ...]:
    """Resolve training-only fields without inheriting evaluation-specific defaults."""
    raw = OmegaConf.to_container(OmegaConf.create(resolve_file_arg(args.rollout_data_config)), resolve=True)
    if not isinstance(raw, dict) or set(raw) - {"datasets", "defaults"}:
        raise ValueError("--rollout-data-config must be a mapping containing datasets and optional defaults")
    defaults = raw.get("defaults", {})
    if defaults is None:
        defaults = {}
    loading_fields = {*DATASET_SAMPLE_SPECS, "max_prompt_len"}
    if not isinstance(defaults, dict) or set(defaults) - loading_fields:
        raise ValueError(f"rollout data defaults may only contain {sorted(loading_fields)}")

    # Training reads only launch defaults, never the eval_* overrides in these shared specs.
    launch_defaults = {name: getattr(args, spec["arg_attrs"][-1]) for name, spec in DATASET_SAMPLE_SPECS.items()}
    launch_defaults["max_prompt_len"] = args.rollout_max_prompt_len
    configs = []
    names = set()
    allowed_fields = {item.name for item in fields(RolloutDatasetConfig)}
    for entry in ensure_dataset_list(raw.get("datasets")):
        unknown = set(entry) - allowed_fields
        if unknown:
            raise ValueError(f"Unsupported rollout dataset fields: {sorted(unknown)}")
        missing = {"name", "path", "samples_per_step"} - set(entry)
        if missing:
            raise ValueError(f"Rollout dataset is missing required fields: {sorted(missing)}")
        config = RolloutDatasetConfig(**{**launch_defaults, **defaults, **entry})
        config = _validate_dataset(config, global_max_prompt_len=args.rollout_max_prompt_len)
        if config.name in names:
            raise ValueError(f"Duplicate rollout dataset name: {config.name!r}")
        names.add(config.name)
        configs.append(config)
    if not configs:
        raise ValueError("--rollout-data-config must define at least one dataset")
    return tuple(configs)


def _validate_dataset(config: RolloutDatasetConfig, *, global_max_prompt_len: int | None) -> RolloutDatasetConfig:
    for name in ("name", "path", "input_key", "metadata_key", "label_key", "tool_key"):
        value = getattr(config, name)
        if name in ("label_key", "tool_key") and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Rollout dataset {name} must be a non-empty string")
    if type(config.samples_per_step) is not int or config.samples_per_step <= 0:
        raise ValueError(f"Dataset {config.name!r}: samples_per_step must be a positive integer weight")
    limit = config.max_prompt_len if config.max_prompt_len is not None else global_max_prompt_len
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError(f"Dataset {config.name!r}: max_prompt_len must be a positive integer")
        if global_max_prompt_len is not None and limit > global_max_prompt_len:
            raise ValueError(f"Dataset {config.name!r}: max_prompt_len exceeds --rollout-max-prompt-len")
    digest = config.sha256
    if digest is not None:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"Dataset {config.name!r}: sha256 must contain exactly 64 hexadecimal characters")
        digest = digest.lower()
    return replace(config, max_prompt_len=limit, sha256=digest)
