import json
import shlex

import pytest
import ray

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

LAUNCHER = REPO_ROOT / "scripts/run_qwen3_6_35b_a3b_mopd.py"


def _submitted(monkeypatch, *, external_ray: bool = False, **cli_args) -> tuple[list[str], dict[str, str], list]:
    """Run `execute` under the recorder; return the train argv, the runtime env, and the node checks it ran."""
    freeze_environment(monkeypatch)
    if external_ray:
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(LAUNCHER)
    node_checks = []
    monkeypatch.setattr(launcher, "_check_trainer_node_sorts_first", node_checks.append)

    launcher.execute(launcher.ScriptArgs(**cli_args))

    (submit,) = [cmd for cmd in recording.commands if "ray job submit" in cmd]
    runtime_env = json.loads(shlex.split(submit.split("--runtime-env-json=", 1)[1])[0])["env_vars"]
    return shlex.split(submit.split(" -- ", 1)[1]), runtime_env, node_checks


@pytest.mark.parametrize("layout, gpus", [("b200-sync", 8), ("gb300-sync", 4)])
def test_sync_layouts_share_the_gpus_and_take_pi_old_from_the_training_forward(monkeypatch, layout, gpus):
    argv, _, node_checks = _submitted(monkeypatch, layout=layout)

    assert argv[1].endswith("/train.py")
    assert "--colocate" in argv and "--rollout-num-gpus" not in argv
    assert "--skip-actor-forward-only" in argv and "--use-rollout-logprobs" not in argv
    assert argv[argv.index("--expert-model-parallel-size") + 1] == str(gpus)
    assert argv[argv.index("--num-gpus-per-node") + 1] == str(gpus)
    assert not node_checks


def test_sync_layouts_can_keep_the_forward_only_pass(monkeypatch):
    argv, _, _ = _submitted(monkeypatch, layout="b200-sync", skip_actor_forward_only=False)

    assert "--skip-actor-forward-only" not in argv and "--use-rollout-logprobs" not in argv


@pytest.mark.parametrize("layout, actor_gpus, rollout_gpus", [("b200-async", 8, 6), ("gb300-async", 4, 2)])
def test_async_layouts_overlap_one_step_and_take_pi_old_from_the_rollout(
    monkeypatch, layout, actor_gpus, rollout_gpus
):
    argv, _, node_checks = _submitted(monkeypatch, external_ray=True, layout=layout)

    assert argv[1].endswith("/train_async.py") and "--fully-async" not in argv
    assert "--colocate" not in argv
    assert argv[argv.index("--rollout-num-gpus") + 1] == str(rollout_gpus)
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == str(actor_gpus)
    assert "--use-rollout-logprobs" in argv and "--skip-actor-forward-only" not in argv
    assert [layout.actor_gpus for layout in node_checks] == [actor_gpus]


def test_async_layouts_need_an_external_ray_cluster(monkeypatch):
    with pytest.raises(AssertionError, match="MILES_SCRIPT_EXTERNAL_RAY=1"):
        _submitted(monkeypatch, layout="b200-async")


def test_every_layout_routes_each_task_to_its_teacher_and_keeps_the_recipe_defaults(monkeypatch):
    argv, _, _ = _submitted(monkeypatch, teacher_host="10.0.0.2")

    teachers = argv[argv.index("--opd-teacher-urls") + 1 : argv.index("--opd-teacher-key")]
    assert teachers == [
        "caesar_cipher=http://10.0.0.2:30000/generate",
        "simple_geometry=http://10.0.0.2:30001/generate",
    ]
    assert "--calculate-per-token-loss" in argv
    assert argv[argv.index("--log-probs-backend") + 1] == "fused"
    assert (
        argv[argv.index("--custom-rm-path") + 1]
        == "examples.on_policy_distillation.mopd_reasoning_gym.hooks.reward_func"
    )
    assert "--custom-megatron-init-path" not in argv
    assert "--sglang-speculative-algorithm" not in argv


def test_mtp_spec_decode_turns_on_eagle_with_the_mtp_head(monkeypatch):
    argv, runtime_env, _ = _submitted(monkeypatch, mtp_spec_decode=True)

    assert argv[argv.index("--sglang-speculative-algorithm") + 1] == "EAGLE"
    assert argv[argv.index("--sglang-speculative-num-draft-tokens") + 1] == "3"
    assert runtime_env["SGLANG_ENABLE_SPEC_V2"] == "1"


def _ray_nodes(monkeypatch, nodes: list[tuple[str, int]]) -> None:
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [{"Alive": True, "NodeManagerAddress": ip, "Resources": {"GPU": gpus}} for ip, gpus in nodes]
        + [{"Alive": True, "NodeManagerAddress": "10.0.0.1", "Resources": {"CPU": 8}}],
    )


def test_the_node_check_sorts_ips_numerically_as_miles_places_the_trainer(monkeypatch):
    launcher = import_launch_script(LAUNCHER)
    # "10.0.0.10" sorts before "10.0.0.9" as a string, but not as an address
    _ray_nodes(monkeypatch, [("10.0.0.10", 6), ("10.0.0.9", 8)])

    launcher._check_trainer_node_sorts_first(launcher._LAYOUTS["b200-async"])


def test_the_node_check_rejects_a_rollout_node_that_sorts_first(monkeypatch):
    launcher = import_launch_script(LAUNCHER)
    _ray_nodes(monkeypatch, [("10.0.0.9", 6), ("10.0.0.10", 8)])

    with pytest.raises(AssertionError, match=r"expose \[6, 8\] GPUs"):
        launcher._check_trainer_node_sorts_first(launcher._LAYOUTS["b200-async"])
