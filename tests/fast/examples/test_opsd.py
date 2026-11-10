import json
import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import ray.cloudpickle
import torch
from examples.on_policy_distillation.qwen3_1_7b_opsd import opsd, prepare_data, run_qwen3_1_7b_opsd
from tests.fast.launch_scripts.py_harness import format_recording, freeze_environment, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

from miles.backends.training_utils import log_utils, parallel
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.ray.rollout.train_data_conversion import process_rollout_data_shard, split_train_data_by_dp_raw
from miles.utils.types import Sample


@pytest.fixture
def args(monkeypatch):
    group = SimpleNamespace(size=1, rank=0)
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(cp=group, tp=group, is_pp_last_stage=True))
    return Namespace(
        train_backend="megatron",
        object_store_backend="ray",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        true_on_policy_mode=False,
        use_dynamic_global_batch_size=False,
        use_opd=False,
        compute_advantages_and_returns=False,
        qkv_format="thd",
        opsd_top_k=3,
        opsd_kl_coef=0.37,
        opsd_kl_clip=0.05,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        balance_data=False,
        ci_test=False,
        log_multi_turn=False,
        log_correct_samples=False,
    )


@pytest.mark.parametrize("layout", ["thd", "bshd"])
@pytest.mark.parametrize("clip", [None, 0.05, 10.0])
def test_loss_and_gradients_match_dense_reference(args, layout, clip):
    args.qkv_format, args.opsd_kl_clip = layout, clip
    batch = {
        "target_tokens": [[0, 1, 0, 1, 2, 0], [1, 3, 0]],
        "loss_weights": [[0.6, 0.3, 0, 0.4, 0.5, 0], [0.5, 0.3, 0]],
        "unconcat_tokens": [torch.tensor([0, 1, 2]), torch.tensor([1, 2])],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 0]), torch.tensor([1])],
        "max_seq_lens": [3, 3] if layout == "bshd" else None,
    }
    reducer = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
    shape = (1, 5, 4) if layout == "thd" else (2, 3, 4)
    logits = torch.linspace(-3, 3, math.prod(shape)).reshape(shape).requires_grad_()
    loss, metrics = opsd.loss_function(args, batch, logits, reducer)
    reference = logits.detach().clone().requires_grad_()
    teacher = torch.tensor([[0.6, 0.3, 0, 0], [0, 0.4, 0.5, 0], [0, 0.5, 0, 0.3]])
    terms = torch.special.xlogy(teacher, teacher) - teacher * reference.reshape(-1, 4)[[0, 1, 3]].log_softmax(-1)
    expected = args.opsd_kl_coef * reducer((terms if clip is None else terms.clamp(max=clip)).sum(-1))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, reference)[0])
    assert metrics["opd_teacher_coverage"].item() == pytest.approx(1.7)
    assert metrics["opd_forward_kl"].item() == pytest.approx(loss.item() / args.opsd_kl_coef)
    expected_clipped = 0 if clip is None else reducer((terms > clip).float().sum(-1) / args.opsd_top_k).item()
    assert metrics["opd_kl_clipfrac"].item() == pytest.approx(expected_clipped)
    assert all(not metric.requires_grad for metric in metrics.values())


def test_empty_response_has_zero_loss_and_gradient(args):
    batch = {
        "target_tokens": [[]],
        "loss_weights": [[]],
        "unconcat_tokens": [torch.tensor([0, 1])],
        "total_lengths": [2],
        "response_lengths": [0],
    }
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    loss, _ = opsd.loss_function(args, batch, logits, torch.sum)
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "key,value",
    [
        ("train_backend", "fsdp"),
        ("object_store_backend", "mooncake"),
        ("tensor_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("true_on_policy_mode", True),
        ("use_dynamic_global_batch_size", True),
        ("use_opd", True),
        ("compute_advantages_and_returns", True),
        ("opsd_top_k", 0),
        ("opsd_top_k", 1.5),
        ("opsd_kl_clip", 0),
        ("opsd_kl_clip", -1),
        ("opsd_kl_clip", math.nan),
        ("opsd_kl_clip", math.inf),
        ("opsd_kl_coef", math.nan),
    ],
)
def test_unsupported_configuration_rejected(args, key, value):
    setattr(args, key, value)
    with pytest.raises(ValueError):
        opsd._validate(args)


async def test_teacher_scores_privileged_prompt_and_student_response(args, monkeypatch):
    args.hf_checkpoint, args.chat_template_path, args.rm_url = "model", None, "teacher"
    sample = Sample(tokens=[7, 1, 2, 3], response_length=2, metadata={"teacher_prompt": "solution"})
    monkeypatch.setattr(opsd, "load_tokenizer", lambda *a, **kw: SimpleNamespace(encode=lambda *a, **kw: [8, 9]))

    async def post(url, payload):
        assert url == "teacher"
        assert payload["input_ids"] == [8, 9, 2, 3]
        assert payload["logprob_start_len"] == 1
        assert payload["sampling_params"]["max_new_tokens"] == 0
        return {
            "meta_info": {
                "input_token_logprobs": [None, [-0.1, 2], [-0.2, 3]],
                "input_top_logprobs": [None, [[-0.1, 2]], [[-0.2, 3]]],
            }
        }

    monkeypatch.setattr(opsd, "post", post)
    assert await opsd.reward_func(args, sample) == 0
    assert sample.train_metadata["target_tokens"] == [2, 0, 0, 3, 0, 0]
    assert sample.train_metadata["loss_weights"] == pytest.approx([math.exp(-0.1), 0, 0, math.exp(-0.2), 0, 0])


@pytest.mark.parametrize("row", [[], None, [[math.nan, 1]], [[0.1, 1]], [[-1, -1]], [[-1, 1], [-2, 1]]])
def test_invalid_teacher_support_rejected(row):
    response = {"meta_info": {"input_token_logprobs": [[-1, 1]], "input_top_logprobs": [row]}}
    with pytest.raises(ValueError):
        opsd._teacher_targets(response, [1], 3)


def test_teacher_token_alignment_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        opsd._teacher_targets({"meta_info": {"input_token_logprobs": [[-1, 2]]}}, [1], 3)


def test_targets_survive_sharding_logging_and_reordered_batching(args, monkeypatch):
    samples = [
        Sample(
            index=i,
            tokens=[i, i + 1],
            response_length=1,
            reward=0,
            train_metadata={"target_tokens": [i, 0, 0], "loss_weights": [0.8, 0, 0]},
        )
        for i in range(4)
    ]
    data = opsd.convert_samples(args, samples)
    assert "metadata" not in data and "loss_fn" not in data
    monkeypatch.setattr(log_utils, "gather_log_data", lambda *a, **kw: None)
    for shard in split_train_data_by_dp_raw(args, data, dp_size=2):
        shard = process_rollout_data_shard(args, ray.cloudpickle.loads(ray.cloudpickle.dumps(shard)))
        shard["tokens"] = [torch.tensor(tokens) for tokens in shard["tokens"]]
        shard["loss_masks"] = [torch.tensor(mask) for mask in shard["loss_masks"]]
        shard["max_seq_lens"] = [2, 2]
        log_utils.log_rollout_data(0, args, shard)
        batch = get_batch(
            DataIterator(shard, micro_batch_indices=[[1, 0]]),
            [
                "tokens",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "max_seq_lens",
                "target_tokens",
                "loss_weights",
            ],
            qkv_format="bshd",
        )
        for tokens, ids, weights in zip(batch["tokens"], batch["target_tokens"], batch["loss_weights"], strict=True):
            assert ids == samples[tokens[0].item()].train_metadata["target_tokens"]
            assert weights == [0.8, 0, 0]


def test_evaluation_handles_boxed_and_zero_padded_answers():
    assert opsd._is_correct("The answer is \\boxed{25}.", "025") == 1
    assert opsd._is_correct("The answer is \\boxed{26}.", "025") == 0
    assert opsd._is_correct("25", "025") == 0
    assert opsd._extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


@pytest.mark.parametrize("entrypoint", ["prepare", "execute"])
def test_launcher_recording(entrypoint, monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    original = run_qwen3_1_7b_opsd.U.exec_command_cpu

    def record(command, capture_output=False):
        result = original(command, capture_output)
        return "12345" if capture_output else result

    monkeypatch.setattr(run_qwen3_1_7b_opsd.U, "exec_command_cpu", record)
    args = run_qwen3_1_7b_opsd.ScriptArgs(
        model_dir=str(tmp_path / "models"),
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        run_id="opsd-test",
    )
    getattr(run_qwen3_1_7b_opsd, entrypoint)(args)
    snapshot = (
        REPO_ROOT
        / "tests/snapshots/launch_scripts/py/examples/on_policy_distillation/qwen3_1_7b_opsd/run_qwen3_1_7b_opsd.py"
        / f"{entrypoint}.txt"
    )
    text = "\n".join(line.rstrip() for line in format_recording(recording, sandbox=tmp_path).splitlines()) + "\n"
    assert_matches_snapshot(snapshot, text, entrypoint)


@pytest.mark.parametrize(
    "key,value", [("MILES_SCRIPT_EXTERNAL_RAY", "1"), ("RAY_ADDRESS", "auto"), ("CUDA_VISIBLE_DEVICES", "0,1,2,3")]
)
def test_launcher_rejects_unsafe_gpu_allocation(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="dedicated node"):
        run_qwen3_1_7b_opsd.ScriptArgs()


def test_prepare_data_keeps_reference_solution_out_of_student_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare_data, "_rows", lambda directory: iter([{"problem": "question", "solution": "secret", "answer": "025"}])
    )
    monkeypatch.setattr(
        prepare_data,
        "load_tokenizer",
        lambda model: SimpleNamespace(
            apply_chat_template=lambda messages, **kw: f"{kw['enable_thinking']}:{messages[0]['content']}"
        ),
    )
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    prepare_data.prepare("model", "train", "eval", train, evaluation)
    row = json.loads(train.read_text())
    assert "secret" not in row["prompt"] and row["prompt"].startswith("False:")
    assert "secret" in row["metadata"]["teacher_prompt"]
    assert json.loads(evaluation.read_text())["prompt"].startswith("True:")
