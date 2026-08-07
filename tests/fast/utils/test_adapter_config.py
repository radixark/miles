"""YAML parsing for adapter run configs."""

import dataclasses
from pathlib import Path

import yaml

from miles.utils.adapter_config import AdapterRunConfig, parse_adapter_run_yaml

# One representative YAML value per AdapterRunConfig field. A field added to
# the dataclass but missed by the parser (or this table) fails the round-trip.
_YAML_VALUES = {
    "data": "/data/prompts.jsonl",
    "input_mode": "multi-lora",
    "rank": 8,
    "alpha": 16,
    "rollout_batch_size": 4,
    "n_samples_per_prompt": 2,
    "save": "/ckpts/demo",
    "input_key": "prompt",
    "label_key": "label",
    "metadata_key": "meta",
    "rm_type": "math",
    "custom_rm_path": "my_pkg.rewards:score",
    "rollout_function_path": "my_pkg.rollouts:generate",
    "num_step": 12,
    "num_epoch": 3,
    "metadata": {"tag": "round-trip"},
}


def _parse(tmp_path: Path, payload: dict) -> AdapterRunConfig:
    yaml_path = tmp_path / "adapter.yaml"
    yaml_path.write_text(yaml.safe_dump(payload))
    return parse_adapter_run_yaml(yaml_path)


def test_yaml_round_trip_covers_every_field(tmp_path):
    config = _parse(tmp_path, dict(_YAML_VALUES))
    for field in dataclasses.fields(AdapterRunConfig):
        assert field.name in _YAML_VALUES, f"add a YAML round-trip value for new field '{field.name}'"
        expected = Path(_YAML_VALUES[field.name]) if field.name == "save" else _YAML_VALUES[field.name]
        assert getattr(config, field.name) == expected, field.name


def test_optional_fields_default_when_absent(tmp_path):
    config = _parse(tmp_path, {"data": "/data/prompts.jsonl"})
    assert config.rollout_function_path is None
    assert config.rank is None and config.alpha is None
    assert config.save is None
    assert config.input_key == "text"
    assert config.metadata == {}
