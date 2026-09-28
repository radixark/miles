import shlex
from types import SimpleNamespace

import pytest

from examples.multi_lora import serve_glm5_3_flash_tinker as launcher


def test_three_node_launcher_is_disaggregated_64k_and_recompute_free(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    args = launcher.ScriptArgs()
    monkeypatch.setattr(
        args,
        "create_backend",
        lambda: SimpleNamespace(execute_train=lambda **kwargs: requests.append(kwargs)),
    )

    launcher._serve(args)

    assert len(requests) == 1
    argv = shlex.split(requests[0]["train_args"])
    assert argv[argv.index("--actor-num-nodes") + 1] == "2"
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "8"
    assert argv[argv.index("--rollout-num-gpus") + 1] == "8"
    assert argv[argv.index("--pipeline-model-parallel-size") + 1] == "2"
    assert argv[argv.index("--expert-model-parallel-size") + 1] == "8"
    assert argv[argv.index("--decoder-first-pipeline-num-layers") + 1] == "22"
    assert argv[argv.index("--decoder-last-pipeline-num-layers") + 1] == "23"
    assert argv[argv.index("--sglang-dsa-prefill-backend") + 1] == "tilelang"
    assert argv[argv.index("--sglang-dsa-decode-backend") + 1] == "tilelang"
    assert argv[argv.index("--target-modules") + 1] == launcher._ROLLOUT_PORTABLE_TARGETS
    assert argv[argv.index("--seq-length") + 1] == "65504"
    assert argv[argv.index("--rollout-max-context-len") + 1] == "65504"
    assert "--colocate" not in argv
    assert "--recompute-granularity" not in argv
    assert requests[0]["job_lifetime"] == "launcher"


def test_five_node_launcher_remains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    args = launcher.ScriptArgs(actor_num_nodes=4, pp=4)
    monkeypatch.setattr(
        args,
        "create_backend",
        lambda: SimpleNamespace(execute_train=lambda **kwargs: requests.append(kwargs)),
    )

    launcher._serve(args)

    argv = shlex.split(requests[0]["train_args"])
    assert argv[argv.index("--actor-num-nodes") + 1] == "4"
    assert argv[argv.index("--pipeline-model-parallel-size") + 1] == "4"
    assert argv[argv.index("--expert-model-parallel-size") + 1] == "8"
    assert argv[argv.index("--decoder-first-pipeline-num-layers") + 1] == "11"
    assert argv[argv.index("--decoder-last-pipeline-num-layers") + 1] == "12"
    assert "--recompute-granularity" not in argv


def test_launcher_rejects_unvalidated_attention_lora() -> None:
    with pytest.raises(ValueError, match="MLP/MoE LoRA"):
        launcher.ScriptArgs(target_modules="attn,mlp")


def test_one_node_four_layer_launcher_uses_disjoint_gpu_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    args = launcher.ScriptArgs(
        model_name="GLM-5.3-Flash-4layer",
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus=4,
        tp=2,
        pp=2,
        ep=2,
        context_length=4096,
        n_adapters=2,
    )
    monkeypatch.setattr(
        args,
        "create_backend",
        lambda: SimpleNamespace(execute_train=lambda **kwargs: requests.append(kwargs)),
    )

    launcher._serve(args)

    argv = shlex.split(requests[0]["train_args"])
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "4"
    assert argv[argv.index("--rollout-num-gpus") + 1] == "4"
    assert argv[argv.index("--sglang-tp-size") + 1] == "4"
    assert argv[argv.index("--pipeline-model-parallel-size") + 1] == "2"
    assert argv[argv.index("--multi-lora-n-adapters") + 1] == "2"
    assert "--decoder-first-pipeline-num-layers" not in argv
    assert args.hf_checkpoint == "/data/model-cache/models/GLM-5.3-Flash-4layer"
    assert requests[0]["megatron_model_type"] == "glm5.3-flash-4layer"
