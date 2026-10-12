"""The partial K3 case retains the common workflow and a bounded sample count."""

import shlex

import pytest
from tests.e2e.torchtitan import _common
from tests.e2e.torchtitan.test_kimi_k3_4layer_cp import CASE as CP_CASE
from tests.e2e.torchtitan.test_kimi_k3_4layer_ep import CASE


def _flags(case, monkeypatch):
    monkeypatch.setattr(_common.U, "get_default_wandb_args", lambda _: "")
    argv = shlex.split(_common.build_train_args(case, wandb_file=__file__))
    return {token: argv[i + 1] for i, token in enumerate(argv[:-1]) if token.startswith("--")}


@pytest.mark.parametrize("case", [CASE, CP_CASE])
def test_partial_model_uses_sixteen_samples_and_bf16_for_both_engines(monkeypatch, case):
    flags = _flags(case, monkeypatch)
    assert flags["--context-parallel-size"] == str(case.cp_size)
    assert int(flags["--rollout-batch-size"]) * int(flags["--n-samples-per-prompt"]) == 16
    assert flags["--global-batch-size"] == "16"
    assert flags["--hf-checkpoint"].endswith("Kimi-K3-4layer-64experts-bf16")
    assert flags["--expert-model-parallel-size"] == "8"
    assert flags["--tensor-model-parallel-size"] == "1"
    assert flags["--sglang-ep-size"] == "4"
    assert flags["--rollout-num-gpus-per-engine"] == "4"
    assert flags["--num-rollout"] == "2"
    assert flags["--rm-type"] == "deterministic_random"
    assert flags["--prompt-data"].endswith("/gsm8k/train.parquet")
    assert flags["--input-key"] == "messages"
    assert flags["--rollout-max-response-len"] == "256"


def test_existing_cases_keep_their_sample_budget(monkeypatch):
    case = _common.CaseConfig(
        model_repo="Qwen/Qwen3-0.6B",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        num_gpus=2,
        seq_len=1024,
        max_response_len=512,
    )
    flags = _flags(case, monkeypatch)
    assert (flags["--rollout-batch-size"], flags["--n-samples-per-prompt"], flags["--global-batch-size"]) == (
        "8",
        "4",
        "32",
    )
