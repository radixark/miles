"""B200 E2E test for DSV4 rollout-prefill/train true-on-policy parity.

The test performs one short rollout, recomputes its log-probabilities with
SGLang prefill, and requires exact per-token BF16 equality with the Megatron
training forward pass. It also exercises Miles' standard TOP CI assertion.
"""

import json
import math
import os
from pathlib import Path

import torch

from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.sglang_rollout import (
    GenerateState,
    generate_rollout as _base_generate_rollout,
    get_model_url,
)
from miles.utils.async_utils import run
from scripts.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=1900,
    suite="stage-c-8-gpu-b200",
    labels=["megatron", "model-scripts"],
)

_PREFILL_RECOMPUTE_PASSES = 5


def generate_rollout_with_prefill_repeats(args, rollout_id, data_source, evaluation=False):
    """Run the normal rollout, then repeat clean-cache scoring on the same engine."""
    output = _base_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    if evaluation:
        return output

    samples = [sample for group in output.samples for sample in group]
    token_snapshots = {id(sample): list(sample.tokens) for sample in samples}
    logprob_passes = {id(sample): [list(sample.rollout_log_probs)] for sample in samples}
    state = GenerateState(args)

    for _ in range(_PREFILL_RECOMPUTE_PASSES - 1):
        run(
            recompute_samples_rollout_logprobs_via_prefill(
                args,
                samples,
                url=get_model_url(args, "default"),
                sampling_params=state.sampling_params,
            )
        )
        for sample in samples:
            if sample.tokens != token_snapshots[id(sample)]:
                raise AssertionError("Repeated prefill scoring mutated the sample token sequence")
            logprob_passes[id(sample)].append(list(sample.rollout_log_probs))

    for sample in samples:
        sample.metadata["prefill_recompute_logprob_passes"] = logprob_passes[id(sample)]
        # Keep the value consumed by training anchored to the original recompute.
        sample.rollout_log_probs = list(logprob_passes[id(sample)][0])

    return output


def _args() -> ScriptArgs:
    # The trainer must use the same fused HC-head arithmetic as the SGLang
    # prefill scorer. Keep the production default unchanged and opt in only
    # for this exact-parity contract test.
    os.environ.setdefault("MILES_DSV4_TOP_FUSED_HC_HEAD", "1")

    storage_root = os.environ.get("MILES_DSV4_TOP_STORAGE_ROOT", "/scratch/models")
    debug_root = os.environ.get(
        "MILES_DSV4_TOP_DEBUG_ROOT",
        f"{storage_root}/miles-top-dsv4-debug",
    )
    rollout_tp = int(os.environ.get("MILES_DSV4_ROLLOUT_TP", "8"))
    if rollout_tp not in (4, 8):
        raise ValueError(f"MILES_DSV4_ROLLOUT_TP must be 4 or 8, got {rollout_tp}")
    runtime_patches = os.environ.get("MILES_DSV4_TOP_RUNTIME_PATCHES", "1")
    if runtime_patches not in ("0", "1"):
        raise ValueError("MILES_DSV4_TOP_RUNTIME_PATCHES must be 0 or 1, " f"got {runtime_patches!r}")
    source_contract = os.environ.get("MILES_DSV4_TOP_SOURCE_CONTRACT", "1")
    if source_contract not in ("0", "1"):
        raise ValueError("MILES_DSV4_TOP_SOURCE_CONTRACT must be 0 or 1, " f"got {source_contract!r}")

    return ScriptArgs(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=8,
        hardware="B200",
        model_dir=storage_root,
        model_local_dir=storage_root,
        data_dir=storage_root,
        save_dir=storage_root,
        debug_data_root=debug_root,
        dump_details=True,
        skip_saving=True,
        use_fault_tolerance=False,
        enable_r3=False,
        optimizer_offload=False,
        extra_env_vars=json.dumps(
            {
                "MILES_DSV4_TOP_RUNTIME_PATCHES": runtime_patches,
                "MILES_DSV4_TOP_SOURCE_CONTRACT": source_contract,
                "MILES_DSV4_TOP_FUSED_HC_HEAD": "1",
                "SGLANG_DSA_FUSE_TOPK": "1",
                "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
                "SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC": "1",
                "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
            },
            separators=(",", ":"),
        ),
        extra_args=(
            "--num-rollout 1 "
            "--rollout-batch-size 1 "
            "--n-samples-per-prompt 1 "
            "--rollout-max-response-len 16 "
            "--rollout-temperature 1 "
            "--rollout-top-p 1 "
            "--rollout-top-k -1 "
            "--recompute-logprobs-via-prefill "
            "--true-on-policy-mode "
            "--ci-test "
            "--ci-disable-weight-update-checker "
            "--true-on-policy-logprob-dtype fp32 "
            "--batch-invariant-mode "
            "--attention-backend flash "
            "--sglang-cuda-graph-backend-decode disabled "
            "--sglang-cuda-graph-backend-prefill disabled "
            "--debug-deterministic-collective "
            f"--rollout-num-gpus-per-engine {rollout_tp} "
            f"--sglang-tp-size {rollout_tp} "
            "--sglang-dp-size 1 "
            f"--sglang-ep-size {rollout_tp} "
            "--sglang-router-policy consistent_hashing "
            "--rollout-function-path "
            "tests.e2e.megatron.model_scripts."
            "test_deepseek_v4_flash_4layer_top_parity."
            "generate_rollout_with_prefill_repeats "
            "--sglang-dsa-topk-backend flashinfer "
            "--miles-dsa-topk-backend flashinfer "
            "--disable-weights-backuper "
            "--debug-disable-optimizer "
        ),
    )


def prepare(args: ScriptArgs) -> None:
    hf_sentinel = Path(args.model_dir) / args.model_name / "config.json"
    if args.hf_checkpoint is None and hf_sentinel.exists():
        args.hf_checkpoint = str(hf_sentinel.parent)
        print(f"Using existing HF checkpoint: {args.hf_checkpoint}")

    dataset_sentinel = Path(args.data_dir) / "gsm8k" / "train.parquet"
    if hf_sentinel.exists() and dataset_sentinel.exists():
        print("Skipping checkpoint/dataset download: " f"{hf_sentinel}, {dataset_sentinel}")
    else:
        _prepare_download(args)

    bf16_sentinel = Path(args.model_dir) / args.bf16_name / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"Skipping FP8->BF16 conversion: {bf16_sentinel}")

    torch_dist_sentinel = Path(args.model_dir) / args.torch_dist_name / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"Skipping torch_dist conversion: {torch_dist_sentinel}")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"


def _bf16_bits_and_order(value: float) -> tuple[int, int]:
    scalar = torch.tensor(value, dtype=torch.bfloat16)
    bits = int(scalar.view(torch.int16).item()) & 0xFFFF
    ordered = ((~bits) & 0xFFFF) if bits & 0x8000 else bits | 0x8000
    return bits, ordered


def _require_finite_dump_values(*, name: str, values: torch.Tensor, dump_path: Path) -> None:
    if not torch.isfinite(values).all():
        raise AssertionError(f"{name} contains non-finite active values in {dump_path}")


def assert_prefill_repeatability(args: ScriptArgs) -> None:
    dump_path = Path(args.debug_data_root) / args.run_id / "dump_details" / "rollout_data" / "0.pt"
    if not dump_path.is_file():
        raise AssertionError(f"No rollout debug dump found at {dump_path}")

    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    samples = payload.get("samples") or []
    if not samples:
        raise AssertionError(f"No samples found in {dump_path}")

    checked_values = 0
    for sample_index, sample in enumerate(samples):
        passes = sample.get("metadata", {}).get("prefill_recompute_logprob_passes")
        if passes is None:
            raise AssertionError(f"Sample {sample_index} in {dump_path} has no repeated-prefill snapshots")
        if len(passes) != _PREFILL_RECOMPUTE_PASSES:
            raise AssertionError(
                f"Sample {sample_index} in {dump_path} has {len(passes)} prefill passes, "
                f"expected {_PREFILL_RECOMPUTE_PASSES}"
            )

        baseline = torch.tensor(passes[0], dtype=torch.float32)
        if not torch.isfinite(baseline).all():
            raise AssertionError(f"Prefill pass 0 contains non-finite values in {dump_path}")

        baseline_bits = baseline.view(torch.int32)
        baseline_bf16_bits = baseline.to(torch.bfloat16).view(torch.int16)
        checked_values += baseline.numel()

        for pass_index, values in enumerate(passes[1:], start=1):
            candidate = torch.tensor(values, dtype=torch.float32)
            if candidate.shape != baseline.shape:
                raise AssertionError(
                    f"Prefill pass shape mismatch for sample {sample_index}: "
                    f"pass0={tuple(baseline.shape)}, pass{pass_index}={tuple(candidate.shape)}"
                )
            if not torch.isfinite(candidate).all():
                raise AssertionError(f"Prefill pass {pass_index} contains non-finite values in {dump_path}")

            raw_bad = baseline_bits != candidate.view(torch.int32)
            if raw_bad.any():
                offset = int(torch.nonzero(raw_bad, as_tuple=False)[0].item())
                bf16_bad = baseline_bf16_bits != candidate.to(torch.bfloat16).view(torch.int16)
                abs_diff = (baseline - candidate).abs()
                raise AssertionError(
                    "DeepSeek-V4 repeated clean-prefill scoring is not bitwise deterministic: "
                    f"sample={sample_index}, pass={pass_index}, "
                    f"raw_mismatches={int(raw_bad.sum())}/{baseline.numel()}, "
                    f"bf16_mismatches={int(bf16_bad.sum())}/{baseline.numel()}, "
                    f"mean_abs_diff={abs_diff.mean().item():.8g}, "
                    f"max_abs_diff={abs_diff.max().item():.8g}, "
                    f"first_offset={offset}, pass0={baseline[offset].item():.10g}, "
                    f"pass{pass_index}={candidate[offset].item():.10g}. "
                    f"Dump={dump_path}"
                )

    print(
        "PASS: exact FP32 bitwise repeatability across "
        f"{_PREFILL_RECOMPUTE_PASSES} clean-cache same-engine DSV4 prefill passes "
        f"for {checked_values} response-token logprobs; dump={dump_path}"
    )


def assert_prefill_train_parity(args: ScriptArgs) -> None:
    dump_root = Path(args.debug_data_root) / args.run_id / "dump_details" / "policy_loss_debug"
    dump_paths = sorted(dump_root.glob("rank_*_call_*.pt"))
    if not dump_paths:
        raise AssertionError(f"No policy-loss debug dumps found under {dump_root}")

    active_count = 0
    mismatch_count = 0
    all_abs_diffs: list[torch.Tensor] = []
    first_mismatch = None

    for dump_path in dump_paths:
        payload = torch.load(dump_path, map_location="cpu", weights_only=True)
        rank = int(payload["rank"])

        for sample in payload["samples"]:
            if "rollout_log_probs" not in sample:
                raise AssertionError(
                    f"{dump_path} has no rollout_log_probs; " "--recompute-logprobs-via-prefill did not reach training"
                )

            train = sample["train_log_probs"].flatten()
            prefill = sample["rollout_log_probs"].flatten()
            mask = sample["local_loss_mask"].flatten().bool()

            if train.shape != prefill.shape or train.shape != mask.shape:
                raise AssertionError(
                    f"Shape mismatch in {dump_path}, sample={sample['index']}: "
                    f"train={tuple(train.shape)}, prefill={tuple(prefill.shape)}, "
                    f"mask={tuple(mask.shape)}"
                )

            active_train = train[mask]
            active_prefill = prefill[mask]
            _require_finite_dump_values(
                name="train_log_probs",
                values=active_train,
                dump_path=dump_path,
            )
            _require_finite_dump_values(
                name="prefill_log_probs",
                values=active_prefill,
                dump_path=dump_path,
            )

            # TOP's transport/training contract is BF16. The trainer scorer is
            # intentionally computed in FP32, so compare the BF16 values that
            # are actually consumed instead of requiring the raw FP32 result
            # to have already landed on the BF16 grid.
            train_bf16 = train.to(torch.bfloat16)
            prefill_bf16 = prefill.to(torch.bfloat16)
            abs_diff = (train_bf16[mask].float() - prefill_bf16[mask].float()).abs()
            all_abs_diffs.append(abs_diff)
            active_count += int(mask.sum().item())

            bad = mask & (train_bf16.view(torch.int16) != prefill_bf16.view(torch.int16))
            num_bad = int(bad.sum().item())
            mismatch_count += num_bad

            if num_bad and first_mismatch is None:
                offset = int(torch.nonzero(bad, as_tuple=False)[0].item())
                first_mismatch = {
                    "rank": rank,
                    "sample": int(sample["index"]),
                    "offset": offset,
                    "train": float(train_bf16[offset].item()),
                    "prefill": float(prefill_bf16[offset].item()),
                    "train_raw": float(train[offset].item()),
                    "prefill_raw": float(prefill[offset].item()),
                    "dump_path": str(dump_path),
                }

    if active_count == 0:
        raise AssertionError(f"No active response tokens found under {dump_root}")

    if first_mismatch is not None:
        diffs = torch.cat(all_abs_diffs).float()
        train_value = first_mismatch["train"]
        prefill_value = first_mismatch["prefill"]
        delta = train_value - prefill_value
        ratio = math.exp(max(-80.0, min(80.0, delta)))

        train_bits, train_order = _bf16_bits_and_order(train_value)
        prefill_bits, prefill_order = _bf16_bits_and_order(prefill_value)
        ulp = abs(train_order - prefill_order)

        raise AssertionError(
            "DeepSeek-V4 per-token prefill/train parity failed: "
            f"{mismatch_count}/{active_count} active token/rank entries mismatched; "
            f"mean_abs_diff={diffs.mean().item():.8g}, "
            f"p99_abs_diff={torch.quantile(diffs, 0.99).item():.8g}, "
            f"max_abs_diff={diffs.max().item():.8g}. "
            f"First mismatch: rank={first_mismatch['rank']}, "
            f"sample={first_mismatch['sample']}, "
            f"response_offset={first_mismatch['offset']}, "
            f"prefill_bf16={prefill_value:.10g} (bits=0x{prefill_bits:04x}, "
            f"raw={first_mismatch['prefill_raw']:.10g}), "
            f"train_bf16={train_value:.10g} (bits=0x{train_bits:04x}, "
            f"raw={first_mismatch['train_raw']:.10g}), "
            f"bf16_ulp_distance={ulp}, exp(train-prefill)={ratio:.10g}. "
            f"First dump={first_mismatch['dump_path']}"
        )

    print(
        "PASS: exact BF16 prefill/train log-prob parity for "
        f"{active_count} active token/rank entries; dumps={dump_root}"
    )


def execute(args: ScriptArgs) -> None:
    _train(args)
    assert_prefill_repeatability(args)
    if os.environ.get("MILES_DSV4_PREFILL_REPEATABILITY_ONLY") == "1":
        return
    assert_prefill_train_parity(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
