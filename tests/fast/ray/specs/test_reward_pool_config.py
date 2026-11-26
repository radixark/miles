"""A text run's reward pools, declared with --reward-pool-config."""

import pytest
import yaml
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.reward import RewardPoolSpec
from miles.utils.args.configs.scaling import ScalingConfig

_HPSV3 = dict(
    name="hpsv3",
    scorer_path="miles.rollout.rm_hub.image.hpsv3.HPSv3Scorer",
    scorer_kwargs=dict(checkpoint_path="/models/HPSv3/HPSv3.safetensors", base_model_path="/models/Qwen2-VL-7B"),
    num_workers=1,
    uses_gpu=True,
    batch_size=16,
)
_GPUS = ("--rollout-num-gpus", "2", "--num-gpus-per-node", "4")


def _parse(tmp_path, pools: list[dict], *argv: str):
    path = tmp_path / "pools.yaml"
    path.write_text(yaml.safe_dump(pools))
    return parse_fsdp_test_config("--reward-pool-config", str(path), *argv)


def _reward_specs(args) -> list[RewardPoolSpec]:
    return [spec for spec in compute_specs(args) if isinstance(spec, RewardPoolSpec)]


def test_a_text_run_declares_its_pool_on_a_gpu_after_the_rollout_gpus(tmp_path):
    args = _parse(tmp_path, [_HPSV3], *_GPUS)

    (pool,) = args.reward_pools
    (spec,) = _reward_specs(args)
    scheduling = spec.scheduling(ScalingConfig.slice_from(args))

    assert pool.name == "hpsv3" and pool.scorer_kwargs == _HPSV3["scorer_kwargs"] and pool.num_gpus_per_node == 4
    assert pool.max_wait_ms == 0
    assert (scheduling.num_cells, scheduling.num_gpu_slots_per_worker) == (1, 1)
    assert (scheduling.pg_name, scheduling.pg_slot_offset) == ("rollout", 2)
    assert _get_placement_group_layout(args)[0] == _get_placement_group_layout(parse_fsdp_test_config(*_GPUS))[0] + 1


def test_pools_follow_each_other_and_the_eval_gpus_and_a_cpu_pool_takes_none(tmp_path):
    pools = [_HPSV3 | dict(num_workers=2), _HPSV3 | dict(name="ocr", uses_gpu=False), _HPSV3 | dict(name="pick")]
    (tmp_path / "eval.jsonl").write_text("{}\n")
    eval_fleet = ("--eval-num-gpus", "1", "--eval-num-gpus-per-engine", "1", "--eval-interval", "5")
    eval_fleet += (
        "--eval-hf-dir",
        str(tmp_path / "eval-hf"),
        "--eval-prompt-data",
        "dummy",
        str(tmp_path / "eval.jsonl"),
    )
    args = _parse(tmp_path, pools, *_GPUS, *eval_fleet)

    offsets = {spec.pool.name: spec.scheduling(ScalingConfig.slice_from(args)) for spec in _reward_specs(args)}

    assert (offsets["hpsv3"].pg_slot_offset, offsets["hpsv3"].num_cells) == (3, 2)
    assert offsets["ocr"].num_gpu_slots_per_worker == 0 and offsets["ocr"].pg_name is None
    assert offsets["pick"].pg_slot_offset == 5
    assert args.reward_scaling.num_gpus == 3


def test_a_run_without_the_flag_declares_no_pool():
    args = parse_fsdp_test_config()

    assert args.reward_pools == [] and args.reward_scaling.num_gpus == 0
    assert _reward_specs(args) == []


def test_a_pool_may_not_set_what_miles_derives_from_the_run(tmp_path):
    with pytest.raises(AssertionError, match="num_gpus_per_node"):
        _parse(tmp_path, [_HPSV3 | dict(num_gpus_per_node=1)])


def test_two_pools_of_one_name_are_rejected(tmp_path):
    with pytest.raises(AssertionError, match="unique"):
        _parse(tmp_path, [_HPSV3, _HPSV3])


def test_a_pool_name_that_cannot_be_a_keyword_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a Python identifier"):
        _parse(tmp_path, [_HPSV3 | dict(name="hps-v3")])


@pytest.mark.parametrize(
    "argv, needs",
    [
        (("--cluster-backend", "kubernetes"), "one Ray deployment"),
        (("--worker-comm-backend", "rpc"), "one Ray deployment"),
        (("--colocate",), "GPUs of their own"),
        (("--debug-rollout-only",), "a rollout placement group"),
    ],
)
def test_a_declared_pool_runs_only_where_the_rollout_can_host_it(tmp_path, argv, needs):
    with pytest.raises(ValueError, match=needs):
        _parse(tmp_path, [_HPSV3], *_GPUS, *argv)


@pytest.mark.parametrize("flag", ["--rollout-function-path", "--eval-function-path"])
def test_a_plain_rollout_function_cannot_take_pools(tmp_path, flag):
    """The legacy adapter calls a plain function as fn(args, rollout_id, data_source), with no pools to hand on."""
    with pytest.raises(ValueError, match="class-based rollout functions"):
        _parse(tmp_path, [_HPSV3], *_GPUS, flag, "miles.rollout.sglang_rollout.generate_rollout")


def test_the_v1_rollout_cannot_take_pools(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "1")

    with pytest.raises(ValueError, match="the v2 rollout"):
        _parse(tmp_path, [_HPSV3])
