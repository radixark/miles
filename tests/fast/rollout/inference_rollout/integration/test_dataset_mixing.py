import copy
import json
from collections import Counter

import pytest
import yaml
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import load_and_call_train

from miles.rollout.mixed_data_source import MixedRolloutDataSource
from miles.utils.arguments import _resolve_data_source_path


@pytest.mark.parametrize(
    "rollout_env",
    [
        RolloutEnvConfig(
            extra_argv=[
                "--rollout-function-path",
                "miles.rollout.sglang_rollout.generate_rollout",
                "--custom-generate-function-path",
                "miles.rollout.sglang_rollout.generate",
                "--rollout-batch-size",
                "4",
                "--n-samples-per-prompt",
                "2",
            ]
        ),
        RolloutEnvConfig(
            extra_argv=[
                "--rollout-function-path",
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
                "--custom-generate-function-path",
                "miles.rollout.generate_hub.single_turn.generate",
                "--rollout-batch-size",
                "4",
                "--n-samples-per-prompt",
                "2",
            ]
        ),
    ],
    ids=["legacy-rollout", "modular-rollout"],
    indirect=True,
)
def test_mixed_prompts_reach_both_rollout_apis_with_tags_and_homogeneous_groups(rollout_env, tmp_path):
    datasets = {}
    for name in ("math", "science"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"input": "What is 1+7?", "label": "8", "metadata": {"rm_type": "math"}}) + "\n")
        datasets[name] = {"path": str(path), "samples_per_step": 1}
    config_path = tmp_path / "mix.yaml"
    config_path.write_text(yaml.safe_dump({"datasets": datasets}))
    args = copy.copy(rollout_env.args)
    args.prompt_data = None
    args.rollout_data_config = str(config_path)
    _resolve_data_source_path(args)
    source = MixedRolloutDataSource(args)

    output = load_and_call_train(args, source)

    assert len(output.samples) == 4
    assert Counter(group[0].metadata["dataset_name"] for group in output.samples) == {"math": 2, "science": 2}
    assert len({sample.index for group in output.samples for sample in group}) == 8
    for group in output.samples:
        assert len(group) == 2
        assert len({sample.group_index for sample in group}) == 1
        assert len({sample.metadata["dataset_name"] for sample in group}) == 1
        assert all(sample.reward == 1 for sample in group)
