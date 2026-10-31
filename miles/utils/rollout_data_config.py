"""Training datasets mixed in every rollout step, read from ``--rollout-data-yaml-config``.

```yaml
datasets:
  math:
    path: /data/math.jsonl
    samples_per_step: 3
  agentic:
    path: /data/agentic.jsonl
    sha256: 51b70c3d
    samples_per_step: 5
    max_prompt_len: 8192
    max_response_len: 32768
    generate_function: miles.rollout.generate_hub.agentic_tool_call.generate
    agent_function: my_pkg.swe_agent
    max_seq_len: 65536
    agent:
      max_turns: 50
```

``samples_per_step`` values are weights: each request for N prompts is split across the datasets in
proportion to them. Per-dataset length limits may only lower the run-wide ``--rollout-max-prompt-len`` /
``--rollout-max-response-len``, which training memory and packing are sized from, and ``max_seq_len`` may only
lower the agentic ``--max-seq-len``.
"""

import re
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from miles.utils.file_arg_utils import resolve_file_arg

# sample metadata the mixer writes for agentic_tool_call, which reads it before the run-wide flags
AGENT_FUNCTION_METADATA_KEY = "agent_function_path"
MAX_SEQ_LEN_METADATA_KEY = "dataset_max_seq_len"
RESERVED_SAMPLE_METADATA_KEYS = frozenset({AGENT_FUNCTION_METADATA_KEY, MAX_SEQ_LEN_METADATA_KEY})

_SHA256_PREFIX_PATTERN = re.compile(r"[0-9a-f]{8}")


class RolloutDatasetConfig(BaseModel):
    """One training dataset and its share of every rollout step."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    samples_per_step: int = Field(ge=1)
    # first 8 hex characters of the file's SHA-256
    sha256: str | None = None
    max_prompt_len: int | None = Field(default=None, ge=1)
    max_response_len: int | None = Field(default=None, ge=1)
    generate_function: str | None = Field(default=None, min_length=1)
    # agent function for agentic_tool_call, instead of --custom-agent-function-path
    agent_function: str | None = Field(default=None, min_length=1)
    max_seq_len: int | None = Field(default=None, ge=1)
    # free-form, for this dataset's generate function
    agent: dict[str, Any] | None = None

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PREFIX_PATTERN.fullmatch(value):
            raise ValueError("must be the first 8 lowercase hex characters of the dataset file's SHA-256")
        return value


def load_rollout_data_config(value: str) -> list[RolloutDatasetConfig]:
    """Parse a ``--rollout-data-yaml-config`` file path or inline ``base64:`` payload."""
    document = OmegaConf.to_container(OmegaConf.create(resolve_file_arg(value)), resolve=True)
    return parse_rollout_datasets(document)


def parse_rollout_datasets(document: Any) -> list[RolloutDatasetConfig]:
    if not isinstance(document, dict) or set(document) != {"datasets"}:
        raise ValueError("--rollout-data-yaml-config must be a mapping with exactly one key, `datasets`")
    raw_datasets = document["datasets"]
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("--rollout-data-yaml-config `datasets` must be a non-empty mapping from name to dataset")

    datasets = []
    for name, fields in raw_datasets.items():
        if not isinstance(fields, dict):
            raise ValueError(f"--rollout-data-yaml-config dataset {name!r} must be a mapping")
        if "name" in fields:
            raise ValueError(f"--rollout-data-yaml-config dataset {name!r}: the name is the mapping key, drop `name`")
        try:
            datasets.append(RolloutDatasetConfig(name=str(name), **fields))
        except ValidationError as e:
            raise ValueError(f"--rollout-data-yaml-config dataset {name!r} is invalid:\n{e}") from e
    return datasets


def check_dataset_length_limits(
    datasets: list[RolloutDatasetConfig],
    *,
    rollout_max_prompt_len: int | None,
    rollout_max_response_len: int | None,
    max_seq_len: int | None = None,
) -> None:
    """Reject a per-dataset limit above the run-wide one; an unset run-wide limit bounds nothing."""
    for dataset in datasets:
        for field_name, value, upper_bound, flag in (
            ("max_prompt_len", dataset.max_prompt_len, rollout_max_prompt_len, "--rollout-max-prompt-len"),
            ("max_response_len", dataset.max_response_len, rollout_max_response_len, "--rollout-max-response-len"),
            ("max_seq_len", dataset.max_seq_len, max_seq_len, "--max-seq-len"),
        ):
            if value is not None and upper_bound is not None and value > upper_bound:
                raise ValueError(
                    f"--rollout-data-yaml-config dataset {dataset.name!r}: {field_name}={value} exceeds "
                    f"{flag} {upper_bound}; a dataset limit may only lower the run-wide one"
                )
