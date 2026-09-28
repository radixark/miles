import shlex
from types import SimpleNamespace

import pytest

from examples.multi_lora import serve_qwen_dense_tinker as launcher


@pytest.mark.parametrize(
    ("base_model", "model_type", "tp", "memory_fraction"),
    [
        ("Qwen/Qwen3.5-4B", "qwen3.5-4B", "2", "0.7"),
        ("Qwen/Qwen3.6-27B", "qwen3.6-27B", "4", "0.5"),
        ("Qwen/Qwen3.8-27B", "qwen3.8-27B", "4", "0.8"),
    ],
)
def test_three_node_64k_presets(base_model, model_type, tp, memory_fraction, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    args = launcher.ScriptArgs(base_model=base_model)
    monkeypatch.setattr(
        args,
        "create_backend",
        lambda: SimpleNamespace(execute_train=lambda **kwargs: requests.append(kwargs)),
    )

    launcher._serve(args)

    argv = shlex.split(requests[0]["train_args"])
    assert argv[argv.index("--actor-num-nodes") + 1] == "2"
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "8"
    assert argv[argv.index("--rollout-num-gpus") + 1] == "8"
    assert argv[argv.index("--tensor-model-parallel-size") + 1] == tp
    assert argv[argv.index("--pipeline-model-parallel-size") + 1] == "2"
    assert argv[argv.index("--seq-length") + 1] == "65504"
    assert argv[argv.index("--rollout-max-context-len") + 1] == "65504"
    assert argv[argv.index("--max-tokens-per-gpu") + 1] == "65504"
    assert argv[argv.index("--rollout-num-gpus-per-engine") + 1] == "1"
    assert argv[argv.index("--sglang-mem-fraction-static") + 1] == memory_fraction
    assert "--recompute-granularity" in argv
    assert "--colocate" not in argv
    assert requests[0]["megatron_model_type"] == model_type
    assert requests[0]["job_lifetime"] == "launcher"


def test_launcher_rejects_noncanonical_topology() -> None:
    with pytest.raises(ValueError, match="two 8-GPU trainer nodes"):
        launcher.ScriptArgs(actor_num_nodes=1)
