import argparse
import os

from pydantic import Field

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR


# debug
class DebugConfig(BaseConfig):
    save_debug_trajectory_data: A[
        str | None,
        Arg(
            help=(
                "Save per-sample role-tagged trajectory text (JSONL) next to the rollout "
                "dump. The file will be saved to `save_debug_trajectory_data.format(rollout_id)`, "
                "so the template must contain the `{rollout_id}` placeholder."
            )
        ),
    ] = None
    load_debug_rollout_data: A[
        str | None,
        Arg(
            help=(
                "Load the rollout data from this path for debugging. "
                "The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. "
                "When this is enabled, miles will not instantiate sglang servers."
            )
        ),
    ] = None
    debug_rollout_only: A[
        bool,
        Arg(
            help=(
                "Whether to only run the rollout generation without training. "
                "This is useful for debugging the rollout generation function."
            )
        ),
    ] = False
    debug_train_only: A[
        bool,
        Arg(
            help=(
                "Whether to run training without rollout generation. Rollout engines are "
                "skipped; a snapshot-eval fleet (--eval-num-gpus) still starts when configured."
            )
        ),
    ] = False
    save_debug_train_data: A[
        str | None,
        Arg(
            help=(
                "Save the train data to this path for debugging. "
                "The file will be saved to `save_debug_train_data.format(rollout_id)`."
            )
        ),
    ] = None
    save_debug_event_data: A[
        str | None,
        Arg(
            help=(
                "Where the audit events of this run go, including the env report. Defaults to <save>/events "
                "(or <dump-details>/events); --ci-test falls back to a run-specific temporary directory."
            )
        ),
    ] = None
    dump_details: A[
        str | None,
        Arg(help="Dump all details of training for post-hoc analysis and visualization."),
    ] = None
    dumper_enable: A[
        bool,
        Arg(
            help=(
                "Enable sglang dumper for all three phases (sglang inference, "
                "megatron forward-only, megatron forward-backward). "
                "Per-phase --dumper-inference/--dumper-fwd-only/--dumper-fwd-bwd can override."
            )
        ),
    ] = False
    dumper_dir: A[
        str,
        Arg(
            help=(
                "Base output directory for sglang dumper. Three subdirs are created: "
                "inference/, fwd_only/, fwd_bwd/."
            )
        ),
    ] = "/tmp/dumper"
    dumper_inference: A[
        list[str] | None,
        Arg(
            type_parser=None,
            nargs="*",
            help="SGLang inference phase dumper config as key=value pairs. "
            "Keys map to DumperConfig fields (e.g. enable=true filter=whatever).",
        ),
    ] = None
    dumper_fwd_only: A[
        list[str] | None,
        Arg(
            type_parser=None,
            nargs="*",
            help="Megatron forward-only phase dumper config as key=value pairs.",
        ),
    ] = None
    dumper_fwd_bwd: A[
        list[str] | None,
        Arg(
            type_parser=None,
            nargs="*",
            help="Megatron forward-backward phase dumper config as key=value pairs.",
        ),
    ] = None
    dumper_source_patcher_config_inference: A[
        str | None,
        Arg(help="Path to YAML config file for source patcher applied in SGLang inference engines."),
    ] = None
    dumper_source_patcher_config_train: A[
        str | None,
        Arg(help="Path to YAML config file for source patcher applied in Megatron training actors."),
    ] = None
    # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
    memory_snapshot_dir: A[str, Arg()] = "."
    memory_snapshot_num_steps: A[int | None, Arg()] = None
    profile_target: A[
        list[str],
        Arg(
            type_parser=str,
            choices=["train_overall", "train_actor", "train_log_probs"],
            nargs="+",
        ),
    ] = ["train_overall"]
    memory_recorder: A[str, Arg(choices=["torch", "memray"])] = "torch"
    check_weight_update_equal: A[bool, Arg()] = False
    check_weight_update_selector: A[
        str,
        Arg(
            choices=["all", "target", "draft"],
            help="Which model the post-update equality check covers: 'all' (target + "
            "draft/MTP), 'target' (target model only; skips the draft, e.g. when MTP "
            "training is off), or 'draft' (draft/MTP worker only).",
        ),
    ] = "all"
    check_weight_update_skip_list: A[
        list[str] | None,
        Arg(
            type_parser=str,
            nargs="*",
            help="Weight-name substrings to exclude from the post-update equality check; "
            "their mismatches are downgraded to non-fatal info (e.g. MTP/draft layer names "
            "that are absent on the training side).",
        ),
    ] = None
    check_weight_update_allow_quant_error: A[
        bool,
        Arg(
            help=(
                "When comparing weights after update, allow quantized tensors to differ "
                "by up to 1 ULP of the quantized dtype per side (compared in dequantized space)."
            )
        ),
    ] = False
    check_lora_weight_equal: A[
        bool,
        Arg(
            help=(
                "Verify the megatron->sglang LoRA adapter weight-sync on the colocated "
                "(from_tensors) path: on every sync the trainer ships a per-tensor sha256 "
                "manifest of the adapter it sends, and each rollout engine hashes the "
                "tensors it received and fails the load on any mismatch/missing/extra "
                "name. The LoRA analogue of --check-weight-update-equal, which only "
                "covers base weights."
            )
        ),
    ] = False
    save_local_weight_checksum: A[bool, Arg(help="Save per-rank local weight checksum per-step.")] = False
    check_weight_transfer_checksum: A[
        bool | None,
        Arg(
            action=argparse.BooleanOptionalAction,
            help="Hash every P2P weight write on the sending trainer rank and on the receiving engine rank and "
            "fail the write when they differ. Defaults on under --ci-test.",
        ),
    ] = None
    enable_event_analyzer: A[
        bool,
        Arg(
            help="Enable event analyzer to run sanity checks (e.g. cross-replica checksum consistency) before each training step."
        ),
    ] = False
    enable_sample_ownership_checker: A[
        bool | None,
        Arg(
            action=argparse.BooleanOptionalAction,
            help="Verify exactly one outcome for every consumed sample and every mature issued sample; "
            "CI enables this unless it is explicitly disabled. Every actor step appends one full "
            "consumption snapshot per replica to the event log, whose size therefore grows with steps "
            "times consumed samples, so this is meant for CI and debugging.",
        ),
    ] = None
    sample_ownership_grace_steps: A[
        int | None,
        Arg(help="Completed rollout training steps before checking an issued sample (default: 10, or 2 in CI)."),
    ] = None
    enable_witness: A[bool, Arg(help="Enable forward/backward pass witness.")] = False
    witness_buffer_size: A[
        int,
        Arg(help="Maximum number of unique witness IDs before recycling."),
    ] = 1048576
    ci_fault_hooks: A[
        str | None,
        Arg(
            help=(
                "JSON array of fault hook requests set when each process starts. Each request names the hook "
                "it waits at, the action to run there, the cell_id / rank it applies to, and the rollout_id / "
                "attempt / weight_version it fires on."
            )
        ),
    ] = None
    ci_fault_hooks_path: A[
        str | None,
        Arg(
            help=(
                "Path of a file holding the same JSON array as --ci-fault-hooks, read when a process starts. "
                "A run relaunched in place keeps the arguments its pods were rendered from, so a plan that has to "
                "change from one launch to the next is delivered through this file. Mutually exclusive with "
                "--ci-fault-hooks."
            )
        ),
    ] = None
    ci_inject_rollout_data_path: A[
        str | None,
        Arg(
            help=(
                "CI comparison tests only: path template (with {rollout_id}) of rollout "
                "data recorded via --save-debug-rollout-data. For rollouts at or after "
                "--ci-inject-rollout-data-start-rollout-id, generation still runs normally "
                "but its result is discarded and the recorded data is used for training "
                "instead. Unlike --load-debug-rollout-data, sglang engines stay alive "
                "(debug_train_only is not forced)."
            )
        ),
    ] = None
    env_report: A[
        str,
        Arg(
            help=("Path to the json record the external launcher wrote about the launch that started " "this process.")
        ),
    ] = Field(default_factory=lambda: os.environ.get(LAUNCHER_REPORT_ENV_VAR, ""))
    env_report_interval_seconds: A[
        float,
        Arg(
            help=(
                "How often every process re-records its environment, so that code loaded later "
                "(lazy imports, a swapped shared disk) is still captured. Non-positive records only at startup."
            )
        ),
    ] = 3600.0
    debug_unified_grad_fused_logprob: A[
        bool,
        Arg(
            help=(
                "Debug/test only: compute the stored log probabilities through the same grad-enabled fused "
                "cross entropy the training step uses, then detach the result, so the two invocations of the "
                "fused kernel take one execution path instead of two."
            )
        ),
    ] = False
    debug_deterministic_collective: A[
        bool,
        Arg(
            help=(
                "Debug/test only: run the training world on the det_nccl backend "
                "(miles.utils.test_utils.det_process_group), which folds order-sensitive SUM/AVG "
                "reductions in a fixed tree order so different reduction topologies become "
                "bitwise-comparable. Slow; never enable in production."
            )
        ),
    ] = False


class DebugRolloutOnlyConfig(BaseConfig):
    save_debug_rollout_data: A[
        str | None,
        Arg(
            help=(
                "Save the rollout data to this path for debugging. "
                "The file will be saved to `save_debug_rollout_data.format(rollout_id)`, "
                "so the template must contain the `{rollout_id}` placeholder."
            )
        ),
    ] = None
    load_debug_rollout_data_subsample: A[
        float | None,
        Arg(help="Subsample a portion of the debug rollout data for faster debugging."),
    ] = None
    ci_inject_rollout_data_start_rollout_id: A[
        int | None,
        Arg(
            help=(
                "First rollout_id whose training data is replaced by the " "--ci-inject-rollout-data-path recordings."
            )
        ),
    ] = None
    ci_inject_rollout_data_min_match_ratio: A[
        float,
        Arg(
            help=(
                "Minimum mean response-token match ratio between the discarded generated "
                "data and the injected recording. Below this the engine weights are considered "
                "wrong (legitimate ulp-level drift only flips occasional sampled tokens)."
            )
        ),
    ] = 0.9
