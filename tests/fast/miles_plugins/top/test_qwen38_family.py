"""Qwen3.8-Flash-Next (qwen4_exp) TOP family: the trainer layouts its ops admit and what its launch carries."""

import pytest

from miles.true_on_policy.gdn import GDN_CHUNKED, GDN_MODE_ENV, qwen_gdn_model
from miles.true_on_policy.launch import top_launch
from miles_plugins.top.registry import TrainerLayout, admit_trainer


def _layout(**kwargs):
    fields = dict(tp=1, etp=1, ep=4, cp=1, pp=1, virtual_pipeline=False, sequence_parallel=False, allgather_cp=False)
    fields.update(kwargs)
    return TrainerLayout(**fields)


def test_expert_parallel_whole_samples_are_admitted():
    assert admit_trainer("qwen4_exp", _layout(), fp8=False).name == "Qwen3.8-Flash-Next"


@pytest.mark.parametrize(
    ("layout", "refusal"),
    [
        (dict(cp=2, allgather_cp=True), "op qsa: cp=2 refused"),
        (dict(cp=2, allgather_cp=True), "op ple: cp=2 refused"),
        (dict(pp=2), "op ple: pp=2 refused: the PLE layer hashes input ids"),
        (dict(pp=2), "op hyper_connection: pp=2 refused"),
        (dict(ep=1), "op qwen2_moe_experts: ep=1 refused"),
    ],
)
def test_layouts_without_a_program_are_refused(layout, refusal):
    """GDN alone admits all-gather CP; the hyper-connection, PLE and QSA ops are what hold a sample on one rank."""
    with pytest.raises(NotImplementedError, match=refusal):
        admit_trainer("qwen4_exp", _layout(**layout), fp8=False)


def test_a_block_fp8_checkpoint_is_refused():
    with pytest.raises(NotImplementedError, match="no block-FP8 program"):
        admit_trainer("qwen4_exp", _layout(), fp8=True)


def test_the_launch_runs_the_gdn_program_with_the_ple_provider():
    """Without the model's provider no forward publishes PLE n-gram ids, and the trainer fails at its first step."""
    launch = top_launch(model_type="qwen4_exp", engine_gpus=4)
    args = list(launch.train_args)
    assert qwen_gdn_model("qwen4_exp").moe
    assert launch.env_vars[GDN_MODE_ENV] == GDN_CHUNKED
    assert launch.env_vars["SGLANG_USE_ATTN_TP_NGRAM"] == "1"
    assert args[args.index("--custom-model-provider-path") + 1].endswith("get_qwen3_8_next_model_provider")
    assert args[args.index("--model-name") + 1] == "qwen4_exp"
    assert args[args.index("--sglang-ep-size") + 1] == "4"
    assert "--sglang-disable-shared-experts-fusion" in args
