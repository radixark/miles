import json
import shlex

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


_LAUNCHER = REPO_ROOT / "scripts/run_step5.py"


def _record(monkeypatch, tmp_path, entrypoint: str, overrides: dict) -> list[str]:
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    call_entrypoint(import_launch_script(_LAUNCHER), entrypoint, overrides, sandbox=tmp_path)
    return recording.commands


def _flag_values(command: str) -> dict[str, str | None]:
    argv = shlex.split(command.split(" -- ", 1)[1])
    return {
        flag: (argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None)
        for i, flag in enumerate(argv)
        if flag.startswith("--")
    }


@pytest.mark.parametrize("task", ["dapo_math", "geo3k"])
@pytest.mark.parametrize(
    "settings,reuse",
    [
        ({}, False),
        ({"reuse_indexer_on_recompute": True}, True),
    ],
)
def test_launcher_passes_the_indexer_reuse_setting(monkeypatch, tmp_path, task, settings, reuse):
    commands = _record(monkeypatch, tmp_path, "train", {"task": task, **settings})
    flags = _flag_values(commands[-1])
    assert ("--reuse-indexer-on-recompute" in flags) is reuse


def test_geo3k_prepare_downloads_the_image_dataset(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path, "prepare", {"task": "geo3k", "model_dir": str(tmp_path / "models")})

    assert any("hf download --repo-type dataset chenhegu/geo3k_imgurl" in command for command in commands)
    assert not any("dapo-math-17k" in command for command in commands)


def test_geo3k_train_feeds_images_to_the_vl_model(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path, "train", {"task": "geo3k", "data_dir": "/data"})

    flags = _flag_values(commands[-1])
    assert flags["--prompt-data"] == "/data/geo3k_imgurl/train.parquet"
    assert (flags["--input-key"], flags["--label-key"]) == ("problem", "answer")
    assert "--apply-chat-template" in flags
    assert flags["--multimodal-keys"] == '{"image": "images"}'
    assert flags["--custom-generate-function-path"] == "examples.geo3k_vlm.rollout.generate"
    assert flags["--custom-model-provider-path"] == "miles_plugins.models.step5.vl_model.step5_vl_model_provider"
    assert "--sglang-enable-multimodal" in flags
    assert flags["--qkv-format"] == "thd"
    # the cap would count one token per image, not the tokens SGLang expands it into
    assert "--rollout-max-prompt-len" not in flags


@pytest.mark.parametrize("task,prompt_cap", [("dapo_math", "2048"), ("geo3k", None)])
def test_one_node_micro_batch_grows_to_hold_a_long_response(monkeypatch, tmp_path, task, prompt_cap):
    commands = _record(monkeypatch, tmp_path, "train", {"task": task, "rollout_max_response_len": 32768})

    flags = _flag_values(commands[-1])
    assert (flags["--rollout-max-response-len"], flags["--max-tokens-per-gpu"]) == ("32768", str(2048 + 32768))
    assert flags.get("--rollout-max-prompt-len") == prompt_cap
    assert "--rollout-max-context-len" not in flags


_TWO_NODES = {"num_nodes": 2, "num_gpus_per_node": 8, "data_dir": "/data"}
_FULL_MODEL = {**_TWO_NODES, "model_name": "Step5"}
# The layout, not the model, picks the recipe, so the slice rehearses the full model's on the same nodes.
_TWO_NODE_MODEL_NAMES = ["Step5", "Step5-5layer"]


@pytest.mark.parametrize("model_name", _TWO_NODE_MODEL_NAMES)
def test_two_node_prepare_fetches_the_dataset_and_converts_nothing(monkeypatch, tmp_path, model_name):
    commands = _record(monkeypatch, tmp_path, "prepare", {**_TWO_NODES, "model_name": model_name})

    assert any("hf download --repo-type dataset zhuzilin/dapo-math-17k" in command for command in commands)
    assert not any("convert_hf_to_torch_dist" in command for command in commands)


@pytest.mark.parametrize("model_name", _TWO_NODE_MODEL_NAMES)
def test_two_node_trainer_reads_the_hf_checkpoint(monkeypatch, tmp_path, model_name):
    overrides = {**_TWO_NODES, "model_name": model_name, "model_dir": "/models"}
    commands = _record(monkeypatch, tmp_path, "train", overrides)

    flags = _flag_values(commands[-1])
    assert flags["--hf-checkpoint"] == flags["--ref-load"] == f"/models/{model_name}"
    assert "--load" not in flags


@pytest.mark.parametrize("model_name", _TWO_NODE_MODEL_NAMES)
def test_two_node_parallelism_and_offloaded_optimizer(monkeypatch, tmp_path, model_name):
    commands = _record(monkeypatch, tmp_path, "train", {**_TWO_NODES, "model_name": model_name})

    flags = _flag_values(commands[-1])
    assert (flags["--tensor-model-parallel-size"], flags["--expert-model-parallel-size"]) == ("4", "16")
    assert "--sequence-parallel" in flags
    assert (flags["--rollout-num-gpus-per-engine"], flags["--sglang-tp-size"], flags["--sglang-ep-size"]) == ("8",) * 3
    assert (flags["--actor-num-nodes"], flags["--num-gpus-per-node"]) == ("2", "8")
    assert flags["--sglang-max-total-tokens"] == str(2 * 1024**2)
    assert (flags["--exp-avg-dtype"], flags["--exp-avg-sq-dtype"], flags["--main-params-dtype"]) == (
        "bf16",
        "bf16",
        "fp32",
    )
    assert "--chunked-optimizer-state-offload" in flags
    assert flags["--optimizer-state-offload-fraction"] == "1.0"
    assert "--grad-reduce-in-bf16" in flags
    assert "--accumulate-allreduce-grads-in-fp32" not in flags
    assert "--optimizer-cpu-offload" not in flags
    assert json.loads(flags["--train-env-vars"]) == {"PYTORCH_ALLOC_CONF": "pinned_max_cached_size_mb:4"}
    assert "--use-rollout-routing-replay" in flags


def test_two_node_samples_32k_responses_at_temperature_one(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path, "train", {**_FULL_MODEL, "num_rollout": 20})

    flags = _flag_values(commands[-1])
    assert flags["--rollout-max-response-len"] == "32768"
    assert flags["--max-tokens-per-gpu"] == str(2048 + 32768)
    # a longer prompt would not fit a micro-batch beside a full-length response
    assert flags["--rollout-max-prompt-len"] == "2048"
    assert "--rollout-max-context-len" not in flags
    assert (flags["--rollout-temperature"], flags["--rollout-top-p"], flags["--rollout-top-k"]) == ("1.0", "1.0", "-1")
    assert flags["--num-rollout"] == "20"


def test_two_node_evaluates_aime26_before_the_first_step_every_10_and_after_the_last(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path, "train", {**_FULL_MODEL, "num_rollout": 20})

    command = commands[-1]
    flags = _flag_values(command)
    assert "--eval-prompt-data aime26 /data/aime-2026/aime-2026.jsonl" in command
    # with the eval before training and the one miles adds after the last step: steps 0, 10 and 20
    assert flags["--eval-interval"] == "10"
    assert "--skip-eval-before-train" not in flags
    assert flags["--n-samples-per-eval-prompt"] == "16"
    assert (flags["--eval-temperature"], flags["--eval-top-p"], flags["--eval-top-k"]) == ("1.0", "0.95", "-1")
    assert flags["--eval-max-response-len"] == "32768"
    # the eval answers at full length even where the trainer caps rollouts
    assert int(flags["--eval-max-context-len"]) > 32768


def test_two_node_can_skip_the_eval(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path, "train", {**_FULL_MODEL, "eval_aime": False})

    assert not any(flag.startswith("--eval") for flag in _flag_values(commands[-1]))


def test_slice_rehearsal_on_two_nodes_caps_prompts_by_the_micro_batch(monkeypatch, tmp_path):
    overrides = {**_TWO_NODES, "model_name": "Step5-5layer", "rollout_max_response_len": 3584, "num_rollout": 2}
    commands = _record(monkeypatch, tmp_path, "train", overrides)

    flags = _flag_values(commands[-1])
    assert (flags["--max-tokens-per-gpu"], flags["--rollout-max-response-len"]) == (str(2048 + 32768), "3584")
    assert flags["--rollout-max-prompt-len"] == str(2048 + 32768 - 3584)
    assert "--rollout-max-context-len" not in flags
    # miles still evaluates after the last step, so a 2-step rehearsal evaluates at steps 0 and 2
    assert (flags["--num-rollout"], flags["--eval-interval"]) == ("2", "10")
    # eval samples never reach the trainer, so they still answer at full length
    assert (flags["--eval-max-response-len"], flags["--eval-max-context-len"]) == ("32768", str(2048 + 32768))


def test_the_slice_on_one_node_keeps_the_tp1_torch_dist_recipe(monkeypatch, tmp_path):
    model_name = "Step5-5layer"
    commands = _record(monkeypatch, tmp_path, "train", {"model_name": model_name, "model_dir": "/models"})

    flags = _flag_values(commands[-1])
    assert flags["--ref-load"] == f"/models/{model_name}_torch_dist"
    assert (flags["--tensor-model-parallel-size"], flags["--expert-model-parallel-size"]) == ("1", "4")
    assert (flags["--rollout-max-response-len"], flags["--max-tokens-per-gpu"]) == ("1024", "16384")
    assert flags["--rollout-max-prompt-len"] == str(16384 - 1024)
    assert "--accumulate-allreduce-grads-in-fp32" in flags
    assert "--chunked-optimizer-state-offload" not in flags
    assert not any(flag.startswith("--eval") for flag in flags)


def test_two_nodes_join_the_other_node_over_ssh_from_a_hostfile(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    recording = install_command_recorder(monkeypatch)
    overrides = {**_FULL_MODEL, "ray_hostfile": "/root/hostfile"}
    call_entrypoint(import_launch_script(_LAUNCHER), "train", overrides, sandbox=tmp_path)

    ssh_commands = [command for command in recording.commands if "ssh root@" in command]
    assert len(ssh_commands) == 1
    assert "awk '{print $1}' /root/hostfile" in ssh_commands[0]
    assert "ray start --address=10.0.0.1:6379 --num-gpus 8" in ssh_commands[0]


def test_a_hostfile_without_master_addr_is_refused(monkeypatch, tmp_path):
    """The head would listen on 127.0.0.1, which no hostfile entry matches, so the join would restart ray on it."""
    freeze_environment(monkeypatch)
    monkeypatch.delenv("MASTER_ADDR")
    install_command_recorder(monkeypatch)
    overrides = {**_FULL_MODEL, "ray_hostfile": "/root/hostfile"}

    with pytest.raises(AssertionError, match="MASTER_ADDR"):
        call_entrypoint(import_launch_script(_LAUNCHER), "train", overrides, sandbox=tmp_path)


def test_full_model_refuses_a_layout_other_than_two_eight_gpu_nodes():
    module = import_launch_script(_LAUNCHER)

    with pytest.raises(AssertionError, match="2 x 8 B300"):
        module.ScriptArgs(model_name="Step5", num_nodes=4, num_gpus_per_node=4)


def test_two_node_recipe_refuses_image_prompts():
    module = import_launch_script(_LAUNCHER)

    with pytest.raises(AssertionError, match="text decoder only"):
        module.ScriptArgs(**_TWO_NODES, model_name="Step5-5layer", task="geo3k")


def test_two_node_recipe_refuses_a_response_its_micro_batch_cannot_hold():
    module = import_launch_script(_LAUNCHER)

    with pytest.raises(AssertionError, match="micro-batch"):
        module.ScriptArgs(**_FULL_MODEL, rollout_max_response_len=65536)


def test_four_h200_nodes_train_adapters_without_routed_experts(monkeypatch, tmp_path):
    overrides = {"model_name": "Step5", "num_nodes": 4, "num_gpus_per_node": 8, "eval_aime": False}
    flags = _flag_values(_record(monkeypatch, tmp_path, "train", overrides)[-1])

    assert (flags["--tensor-model-parallel-size"], flags["--expert-model-parallel-size"]) == ("4", "32")
    assert (flags["--lora-rank"], flags["--lora-alpha"], flags["--lr"]) == ("16", "32", "1e-05")
    assert flags["--target-modules"] == "all-linear"
    assert flags["--exclude-modules"] == "model.layers.*.moe.experts.*"
    assert "--lora-base-cpu-backup" in flags
    assert "--chunked-optimizer-state-offload" not in flags
    assert flags["--sglang-attention-backend"] == "fa3"
    assert flags["--rollout-max-response-len"] == "8192"
    assert not any(flag.startswith("--eval") for flag in flags)
