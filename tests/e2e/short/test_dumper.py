# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord, parse_record_json

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="test_miles_dumper_"))
DUMP_DIR: str = str(_RUN_DIR / "dumps")
MEGATRON_SOURCE_PATCHER_CONFIG_PATH: str = str(_RUN_DIR / "megatron_source_patcher.yaml")
SGLANG_SOURCE_PATCHER_CONFIG_PATH: str = str(_RUN_DIR / "sglang_source_patcher.yaml")

EXP_PATTERNS = ["engine_*", "fwd_only", "fwd_bwd"]

SOURCE_PATCHED_FIELDS = ["layer_input", "attn_output", "pre_mlp_residual", "mlp_output"]

EXPECTED_FIELDS: dict[str, list[str]] = {
    "engine_*": ["input_ids", "positions"] + SOURCE_PATCHED_FIELDS,
    "fwd_only": ["input_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"] + SOURCE_PATCHED_FIELDS,
    "fwd_bwd": ["input_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"] + SOURCE_PATCHED_FIELDS,
}

MEGATRON_SOURCE_PATCHER_CONFIG_YAML: str = """\
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t(sp) 1 h')"
      - match: "nvtx_range_pop(suffix=\\"self_attention\\")"
        append: "dumper.dump('attn_output', attention_output_with_bias[0], dims='t(sp) 1 h')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "residual = hidden_states"
        append: "dumper.dump('pre_mlp_residual', residual, dims='t(sp) 1 h')"
      - match: "nvtx_range_pop(suffix=\\"mlp\\")"
        append: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='t(sp) 1 h')"
"""

SGLANG_SOURCE_PATCHER_CONFIG_YAML: str = """\
patches:
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
        prepend: "dumper.dump('layer_input', hidden_states if residual is None else hidden_states + residual, dims='t h')"
      - match: |
          if hidden_states.shape[0] != 0:
              hidden_states = self.self_attn(
                  positions=positions,
                  hidden_states=hidden_states,
                  forward_batch=forward_batch,
              )
        append: "dumper.dump('attn_output', hidden_states, dims='t h(tp,partial)')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', residual, dims='t h')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h(tp,partial)')"
"""

# Two configs that together cover all parallelism dimensions:
#   Config A: TP=2, SP, PP=2, EP=2, DP=2            → covers DP
#   Config B: TP=2, SP, PP=2, CP=2, EP=2, eTP=2     → covers CP, expert_TP
CONFIGS: dict[str, str] = {
    "tp2_pp2_ep2_dp2": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    ),
    "tp2_pp2_cp2_ep2_etp2": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 2 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    ),
}


def prepare() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)
    U.exec_command(f"rm -rf {DUMP_DIR}")

    Path(MEGATRON_SOURCE_PATCHER_CONFIG_PATH).write_text(MEGATRON_SOURCE_PATCHER_CONFIG_YAML)
    Path(SGLANG_SOURCE_PATCHER_CONFIG_PATH).write_text(SGLANG_SOURCE_PATCHER_CONFIG_YAML)


def _execute(perf_args: str, dump_subdir: str) -> None:
    dump_dir = f"{DUMP_DIR}/{dump_subdir}"

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        "--num-rollout 1 --rollout-batch-size 4 --n-samples-per-prompt 2 "
        "--rollout-max-response-len 20 --rollout-temperature 0.8 "
        "--global-batch-size 8 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--optimizer-cpu-offload --use-precision-aware-optimizer "
    )

    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    sglang_args = "--rollout-num-gpus-per-engine 8 " "--sglang-mem-fraction-static 0.6 "

    dumper_filter: str = "'filter=layer_id is None or layer_id < 3'"
    dumper_args = (
        f"--dumper-enable --dumper-dir {dump_dir} "
        f"--dumper-inference {dumper_filter} "
        f"--dumper-fwd-only enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-fwd-bwd enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-source-patcher-config-train {MEGATRON_SOURCE_PATCHER_CONFIG_PATH} "
        f"--dumper-source-patcher-config-inference {SGLANG_SOURCE_PATCHER_CONFIG_PATH} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
        "--moe-token-dispatcher-type alltoall "
        "--use-miles-router --use-rollout-routing-replay "
    )

    train_args = " ".join(
        [
            ckpt_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            perf_args,
            sglang_args,
            dumper_args,
            misc_args,
            U.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def _check_dump_dir(phase_dir: Path, exp_pattern: str, expected_fields: list[str] | None = None) -> None:
    assert phase_dir.exists(), f"Missing dump dir: {phase_dir}"
    dump_subdirs = list(phase_dir.glob(exp_pattern))
    assert len(dump_subdirs) > 0, f"No {exp_pattern} subdirs in {phase_dir}"
    dump_files = list(dump_subdirs[0].glob("*.pt"))
    assert len(dump_files) > 0, f"No .pt files in {dump_subdirs[0]}"
    sample = torch.load(dump_files[0], weights_only=False)
    assert isinstance(sample, dict), f"Unexpected type: {type(sample)}"
    assert "value" in sample and "meta" in sample, f"Missing keys: {sample.keys()}"

    if expected_fields:
        for field in expected_fields:
            matches = list(phase_dir.rglob(f"*name={field}*.pt"))
            assert len(matches) > 0, f"Expected field '{field}' not found under {phase_dir}"


def verify(dump_subdir: str) -> None:
    base = Path(f"{DUMP_DIR}/{dump_subdir}")
    for pattern in EXP_PATTERNS:
        _check_dump_dir(base, pattern, expected_fields=EXPECTED_FIELDS.get(pattern))
    print(f"All dump verifications passed for {dump_subdir}!")


def _log_comparator_output(stdout: str, stderr: str) -> None:
    if stdout.strip():
        print(f"[comparator stdout]\n{stdout}")
    if stderr.strip():
        print(f"[comparator stderr]\n{stderr}")


def _verify_comparator(dump_subdir: str) -> None:
    baseline_dir: Path = Path(f"{DUMP_DIR}/{dump_subdir}/engine_0")
    target_dir: Path = Path(f"{DUMP_DIR}/{dump_subdir}/fwd_bwd")

    result: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-m",
            "sglang.srt.debug_utils.comparator",
            "--baseline-path",
            str(baseline_dir),
            "--target-path",
            str(target_dir),
            "--output-format",
            "json",
            "--grouping",
            "logical",
        ],
        capture_output=True,
        text=True,
    )

    _log_comparator_output(stdout=result.stdout, stderr=result.stderr)

    assert result.returncode == 0, f"Comparator failed (rc={result.returncode})\nstderr: {result.stderr[-2000:]}"

    records = [parse_record_json(line) for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(records) > 0

    comparisons: list[ComparisonRecord] = [r for r in records if isinstance(r, ComparisonRecord)]
    assert len(comparisons) > 0, "No comparison records produced"

    diff_passed: int = 0
    diff_failed: list[str] = []
    for comp in comparisons:
        if comp.diff is not None and comp.diff.passed:
            diff_passed += 1
        else:
            rel_diff: float = comp.diff.rel_diff if comp.diff is not None else float("nan")
            diff_failed.append(f"{comp.name} (rel_diff={rel_diff:.6f})")

    assert len(diff_failed) == 0, (
        f"Comparator found {len(diff_failed)} diff failures out of {len(comparisons)} comparisons: "
        + ", ".join(diff_failed[:10])
    )
    assert diff_passed > 0, f"No comparisons passed (total={len(comparisons)})"

    print(
        f"Comparator verification passed: engine_0 vs fwd_bwd — "
        f"total={len(comparisons)}, diff_passed={diff_passed}, diff_failed={len(diff_failed)}"
    )


def _select_configs() -> dict[str, str]:
    selected = os.environ["MILES_TEST_DUMPER_CONFIG"]
    if selected not in CONFIGS:
        raise ValueError(f"Unknown MILES_TEST_DUMPER_CONFIG={selected!r}, " f"valid values: {list(CONFIGS.keys())}")
    return {selected: CONFIGS[selected]}


if __name__ == "__main__":
    print(f"Run directory: {_RUN_DIR}")
    configs = _select_configs()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    for config_name, perf_args in configs.items():
        _execute(perf_args=perf_args, dump_subdir=config_name)
        verify(config_name)
        _verify_comparator(config_name)
