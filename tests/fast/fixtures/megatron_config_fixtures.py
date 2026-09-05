from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.common import encode_pseudo_file


def write_megatron_config(tmp_path: Path, *model_ids: str, filename: str = "megatron.yaml") -> str:
    return write_megatron_config_trainers(
        tmp_path, [{"model_id": model_id} for model_id in model_ids], filename=filename
    )


def write_megatron_config_trainers(
    tmp_path: Path, trainers: list[dict[str, Any]], *, filename: str = "megatron.yaml"
) -> str:
    path = tmp_path / filename
    path.write_text(yaml.dump({"trainers": trainers}))
    return str(path)


def encode_megatron_config(*model_ids: str) -> str:
    return encode_pseudo_file(yaml.dump({"trainers": [{"model_id": model_id} for model_id in model_ids]}))
