import fcntl
import math
import os
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from scripts.run_deepseek_v4 import (
    _DSV4_TE_PRECISION_CONFIG,
    ScriptArgs,
    _prepare_download,
    _prepare_single,
    _prepare_spmd,
    _train,
)
from tests.ci.ci_register import register_cuda_ci

from miles.utils.test_utils.comparisons.metrics import compare_metrics
from miles.utils.types import Sample

register_cuda_ci(
    est_time=1000,
    suite="stage-c-4-gpu-h200",
    labels=["precision", "long"],
)

_MODEL_NAME = "DeepSeek-V4-Flash-FP8-4layer"
_NUM_GPUS = 4
_NUM_PROMPTS = 8
_SAMPLES_PER_PROMPT = 2
_NUM_SAMPLES = _NUM_PROMPTS * _SAMPLES_PER_PROMPT
_ROLLOUT_RUN_ID = "rollout"
_ROLLOUT_ID = "0"
_CUSTOM_RM_PATH = "tests.e2e.precision.test_dsv4_bshd_thd_rollout_parity.deterministic_index_reward"

_LOG_PROB_RTOL = 0.2
_LOG_PROB_ATOL = 0.9
_LOG_PROB_MEAN_ABS = 0.1
_LOG_PROB_P99_ABS = 0.9
_LOG_PROB_MAX_ABS = 1.5


async def deterministic_index_reward(
    args: Namespace,
    samples: Sample | list[Sample],
    **_kwargs,
) -> float | list[float]:
    def score(sample: Sample) -> float:
        assert sample.index is not None
        return float(sample.index % args.n_samples_per_prompt)

    if isinstance(samples, list):
        return [score(sample) for sample in samples]
    return score(samples)


def _prepare_args() -> ScriptArgs:
    return ScriptArgs(
        run_id="prepare",
        model_name=_MODEL_NAME,
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=_NUM_GPUS,
        hardware="H200",
        skip_saving=True,
        use_fault_tolerance=False,
    )


def _common_extra_args(*, te_precision_config_path: Path, micro_batch_size: int) -> str:
    return (
        "--ci-test "
        "--check-weight-update-allow-quant-error "
        "--ci-disable-logprobs-checker "
        "--wandb-mode disabled "
        "--num-rollout 1 "
        f"--rollout-batch-size {_NUM_PROMPTS} "
        f"--n-samples-per-prompt {_SAMPLES_PER_PROMPT} "
        f"--global-batch-size {_NUM_SAMPLES} "
        f"--micro-batch-size {micro_batch_size} "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.7 "
        "--rollout-seed 42 "
        "--sglang-cuda-graph-max-bs 4 "
        "--seed 1234 "
        f"--te-precision-config-file {te_precision_config_path} "
    )


def _run_args(
    *,
    run_id: str,
    debug_root: Path,
    hf_checkpoint: str,
    extra_args: str,
    debug_train: bool,
) -> ScriptArgs:
    return ScriptArgs(
        run_id=run_id,
        model_name=_MODEL_NAME,
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=_NUM_GPUS,
        hardware="H200",
        hf_checkpoint=hf_checkpoint,
        skip_saving=True,
        use_fault_tolerance=False,
        dump_details=True,
        debug_data_root=str(debug_root),
        debug_train_run_id=_ROLLOUT_RUN_ID if debug_train else None,
        debug_train_rollout_id=_ROLLOUT_ID if debug_train else None,
        extra_args=extra_args,
    )


def _prepare(args: ScriptArgs) -> str:
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / f".{args.model_name}.ci-prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _prepare_download(args)
        _prepare_single(args)
        _prepare_spmd(args)

    return f"{args.model_local_dir}/{args.model_name}"


def _assert_rollout_artifact(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["rollout_id"] == 0
    samples = sorted(payload["samples"], key=lambda sample: sample["index"])
    assert [sample["index"] for sample in samples] == list(range(_NUM_SAMPLES))
    assert [sample["group_index"] for sample in samples] == [
        group_index for group_index in range(_NUM_PROMPTS) for _ in range(_SAMPLES_PER_PROMPT)
    ]

    for start in range(0, _NUM_SAMPLES, _SAMPLES_PER_PROMPT):
        pair = samples[start : start + _SAMPLES_PER_PROMPT]
        assert pair[0]["group_index"] == pair[1]["group_index"]
        assert [sample["reward"] for sample in pair] == [0.0, 1.0]
        assert all(0 < sample["response_length"] <= 256 for sample in pair)
        responses = [tuple(sample["tokens"][-sample["response_length"] :]) for sample in pair]
        assert responses[0] != responses[1]


def _run_rollout(*, debug_root: Path, hf_checkpoint: str, te_precision_config_path: Path) -> None:
    args = _run_args(
        run_id=_ROLLOUT_RUN_ID,
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        debug_train=False,
        extra_args=(
            _common_extra_args(
                te_precision_config_path=te_precision_config_path,
                micro_batch_size=_NUM_SAMPLES,
            )
            + "--debug-rollout-only "
            + "--sglang-server-concurrency 1 "
            + f"--custom-rm-path {_CUSTOM_RM_PATH} "
        ),
    )
    _train(args)
    rollout_path = debug_root / _ROLLOUT_RUN_ID / "dump_details" / "rollout_data" / f"{_ROLLOUT_ID}.pt"
    _assert_rollout_artifact(rollout_path)


def _run_train(
    *,
    qkv_format: str,
    debug_root: Path,
    hf_checkpoint: str,
    te_precision_config_path: Path,
) -> Path:
    grad_norm_path = debug_root / f"{qkv_format}_grad_norm.pt"
    # BSHD uses one sequence per microbatch as the reference; THD packs all samples.
    micro_batch_size = 1 if qkv_format == "bshd" else _NUM_SAMPLES
    args = _run_args(
        run_id=qkv_format,
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        debug_train=True,
        extra_args=(
            _common_extra_args(
                te_precision_config_path=te_precision_config_path,
                micro_batch_size=micro_batch_size,
            )
            + f"--qkv-format {qkv_format} "
            + f"--ci-save-grad-norm {grad_norm_path} "
        ),
    )
    _train(args)
    grad_norm = float(torch.load(grad_norm_path, map_location="cpu", weights_only=False))
    assert math.isfinite(grad_norm) and grad_norm > 0
    return debug_root / qkv_format / "dump_details"


def _load_rank_zero_train_data(directory: Path) -> dict:
    dump_files = sorted(directory.glob("*.pt"))
    expected_names = [f"{_ROLLOUT_ID}_{rank}.pt" for rank in range(_NUM_GPUS)]
    assert [path.name for path in dump_files] == expected_names

    payload = torch.load(dump_files[0], map_location="cpu", weights_only=False)
    assert payload["rollout_id"] == 0
    assert payload["rank"] == 0
    rollout_data = payload["rollout_data"]
    assert rollout_data["sample_indices"] == list(range(_NUM_SAMPLES))
    return rollout_data


def _compare_train_log_probs(bshd_dir: Path, thd_dir: Path) -> None:
    bshd = _load_rank_zero_train_data(bshd_dir)
    thd = _load_rank_zero_train_data(thd_dir)
    bshd_log_probs: list[torch.Tensor] = []
    thd_log_probs: list[torch.Tensor] = []

    for sample_index in range(_NUM_SAMPLES):
        assert bshd["response_lengths"][sample_index] == thd["response_lengths"][sample_index]
        response_length = bshd["response_lengths"][sample_index]
        for field in ("tokens", "loss_masks"):
            torch.testing.assert_close(
                bshd[field][sample_index],
                thd[field][sample_index],
                rtol=0,
                atol=0,
            )

        baseline = bshd["log_probs"][sample_index]
        target = thd["log_probs"][sample_index]
        for log_probs in (baseline, target):
            assert log_probs.ndim == 1
            assert log_probs.is_floating_point()
            assert log_probs.numel() == response_length
            assert torch.isfinite(log_probs).all()
        bshd_log_probs.append(baseline)
        thd_log_probs.append(target)

    baseline = torch.cat(bshd_log_probs).double()
    target = torch.cat(thd_log_probs).double()
    diffs = (baseline - target).abs()
    mean_abs = float(diffs.mean())
    p99_abs = float(torch.quantile(diffs, 0.99))
    max_abs = float(diffs.max())
    print(
        f"Train log-prob comparison: count={diffs.numel()}, "
        f"mean_abs={mean_abs:.8e}, p99_abs={p99_abs:.8e}, max_abs={max_abs:.8e}"
    )

    torch.testing.assert_close(baseline, target, rtol=_LOG_PROB_RTOL, atol=_LOG_PROB_ATOL)
    assert mean_abs <= _LOG_PROB_MEAN_ABS
    assert p99_abs <= _LOG_PROB_P99_ABS
    assert max_abs <= _LOG_PROB_MAX_ABS


def _compare(bshd_dump: Path, thd_dump: Path) -> None:
    _compare_train_log_probs(
        bshd_dump / "train_data",
        thd_dump / "train_data",
    )
    compare_metrics(
        baseline_dir=str(bshd_dump),
        target_dir=str(thd_dump),
        rtol=0.1,
        atol=0.03,
        key_prefixes=["train/loss", "train/ppo_kl", "train/grad_norm"],
        exclude_keys=[],
    )


def main() -> None:
    prepare_args = _prepare_args()
    hf_checkpoint = _prepare(prepare_args)
    for env_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "WANDB_API_KEY"):
        os.environ.pop(env_var, None)

    debug_root = Path(tempfile.mkdtemp(prefix="miles-dsv4-bshd-thd-", dir="/tmp"))
    te_precision_config_path = debug_root / "te_precision.yaml"
    te_precision_config_path.write_text(f"{_DSV4_TE_PRECISION_CONFIG}\n", encoding="utf-8")
    _run_rollout(
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        te_precision_config_path=te_precision_config_path,
    )
    bshd_dump = _run_train(
        qkv_format="bshd",
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        te_precision_config_path=te_precision_config_path,
    )
    thd_dump = _run_train(
        qkv_format="thd",
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        te_precision_config_path=te_precision_config_path,
    )
    _compare(bshd_dump, thd_dump)


if __name__ == "__main__":
    main()
