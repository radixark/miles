"""
DeepSeek V4 training script.

Supports:
  - DeepSeek-V4-Flash-FP8         Public FP8 repackage of deepseek-ai/DeepSeek-V4-Flash
                                  (sgl-project/DeepSeek-V4-Flash-FP8, 291B, 43 layers).
                                  Verified full-model profiles: 4 nodes x 8 GPUs on MI355X (gfx950),
                                  2 nodes x 8 GPUs on MI300X (gfx942).
  - DeepSeek-V4-Flash-FP8-4layer  4-layer prune of the above for single-node
                                  smoke testing. **Cannot generate meaningful output -
                                  pipeline-only sanity check.**

Hardware is read from the first visible GPU, in a subprocess so this process never holds a
device context while the job runs:
  - FP8 training uses TE's blockwise recipe where it is supported and falls back to tensorwise
    E4M3 (Float8CurrentScaling) where it is not, which is gfx942. --fp8-recipe forces one; an
    unsupported choice fails before anything is submitted. There is never a silent BF16 fallback.
  - On gfx942 the rollout computes router logits in FP32, keeps a BF16 KV cache and stores the
    indexer's K scales unrounded, so the trainer is configured to match: --moe-router-dtype fp32
    --dsv4-kv-qat off --dsv4-index-qat fp8_dynamic.
  - The colocate memory split follows the card: below 256 GiB (MI300X) the trainer is offloaded
    and SGLang gets 0.75 of the card; --sglang-mem-fraction-static overrides the fraction.

Usage patterns:

  1. One-shot full pipeline (download + convert + train):
       python scripts/run_deepseek_v4.py full-train \
           --model-name DeepSeek-V4-Flash-FP8-4layer \
           --num-nodes 1 --num-gpus-per-node 8

  2. Individual steps (download -> FP8->BF16 -> BF16->torch_dist -> train):
       python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-single   --model-name DeepSeek-V4-Flash-FP8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-spmd     --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 1 --num-gpus-per-node 8
       python scripts/run_deepseek_v4.py train            --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 4 --num-gpus-per-node 8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8

  3. Two MI300X nodes on a Ray cluster that is already joined (ray start on both nodes):
       MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://<head>:<dashboard-port> MASTER_ADDR=<head> \
       NCCL_SOCKET_IFNAME=<nic> GLOO_SOCKET_IFNAME=<nic> \
       python scripts/amd/run_deepseek_v4.py train --mode normal \
           --num-nodes 2 --num-gpus-per-node 8 \
           --model-dir /root/models --data-dir /root/datasets \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8 --skip-saving
"""

import functools
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "DeepSeek-V4-Flash-FP8": "sgl-project",
    # 4-layer prune of sgl-project/DeepSeek-V4-Flash-FP8.
    "DeepSeek-V4-Flash-FP8-4layer": "Pinaster",
}

_MEGATRON_MODEL_TYPE = {
    "DeepSeek-V4-Flash-FP8": "deepseek-v4-flash",
    "DeepSeek-V4-Flash-FP8-4layer": "deepseek-v4-flash-4layer",
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    # Context parallelism for the training actor on multi-node layouts. Unset picks the verified
    # value: 8 on 2 nodes, 1 (the historical layout) on 4. It is the only axis that divides the
    # indexer's [seqlen, seqlen_kv] score matrix: pipeline parallelism shrinks how many layers a
    # rank owns, not what each layer allocates.
    context_parallel_size: int | None = None
    run_id: str = command_utils.create_run_id()
    model_org: str = ""
    model_name: Literal[
        "DeepSeek-V4-Flash-FP8",
        "DeepSeek-V4-Flash-FP8-4layer",
    ] = "DeepSeek-V4-Flash-FP8"

    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    enable_eval: bool = True
    enable_mtp: bool = False

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    # Defaults to model_dir. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    model_local_dir: str | None = None
    save_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 8
    # use colocate by default. will switch to disaggregated mode when 0 < rollout_num_nodes < num_nodes
    rollout_num_nodes: int = 0
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)
    rollout_num_gpus: int = field(init=False)
    optimizer_offload: bool = True
    use_fault_tolerance: bool = True
    # Colocate split between the training actor and the rollout engines on one card. auto reads
    # the capacity of the first GPU: 288gb for cards of at least 256 GiB, else 192gb.
    colocate_memory_profile: Literal["auto", "192gb", "288gb"] = "auto"
    # Replaces the profile's --sglang-mem-fraction-static. The engine sizes its KV cache from
    # this fraction of the card while the offloaded trainer still holds its context and
    # communicator buffers, so the right value depends on more than the card capacity.
    sglang_mem_fraction_static: float | None = None

    # debug configs
    dump_details: bool = False
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    debug_data_root: str = "/root/shared_data"
    skip_saving: bool = False

    # precision configs
    enable_r3: bool = True
    train_deterministic: bool = True
    # Megatron-side training precision: TE FP8 GEMMs when True, BF16 when False. Rollout always
    # serves the source FP8 checkpoint either way.
    fp8_training: bool = True
    # auto: blockwise 128x128 (fp32 scales) where TE supports it, else tensorwise E4M3.
    fp8_recipe: Literal["auto", "blockwise", "tensorwise"] = "auto"
    enable_mis: bool = False

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]
        if self.model_local_dir is None:
            self.model_local_dir = self.model_dir
        assert self.rollout_num_nodes >= 0
        assert self.rollout_num_nodes < self.num_nodes
        # Checked here rather than in _train so full-train rejects a bad value before it
        # downloads and converts anything.
        if self.context_parallel_size is not None and self.context_parallel_size < 1:
            raise ValueError(f"--context-parallel-size must be at least 1, got {self.context_parallel_size}.")
        if self.sglang_mem_fraction_static is not None and not 0.0 < self.sglang_mem_fraction_static < 1.0:
            raise ValueError(
                f"--sglang-mem-fraction-static must be strictly between 0 and 1, got {self.sglang_mem_fraction_static}."
            )
        self.colocate = self.rollout_num_nodes == 0
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node
        if self.colocate:
            self.rollout_num_gpus = self.num_nodes * self.num_gpus_per_node
        else:
            self.rollout_num_gpus = self.rollout_num_nodes * self.num_gpus_per_node

    @property
    def megatron_model_type(self):
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        return f"{self.model_name}-bf16"


def _download_dataset(args: ScriptArgs):
    """Download the task-specific dataset(s)."""
    U = args.create_backend()
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _hf_checkpoint_path(args: ScriptArgs) -> str:
    """Resolve hf_checkpoint path: explicit override wins, else {model_dir}/{model_name}."""
    return args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"


def _ensure_4layer_model_type(args: ScriptArgs):
    """Undo the old deepseek_ref workaround for local 4-layer prunes."""
    if args.model_name != "DeepSeek-V4-Flash-FP8-4layer":
        return
    cfg = Path(_hf_checkpoint_path(args)) / "config.json"
    if not cfg.exists():
        return
    text = cfg.read_text()
    if '"model_type": "deepseek_ref"' in text:
        cfg.write_text(text.replace('"model_type": "deepseek_ref"', '"model_type": "deepseek_v4"'))
        print(f"[patch] {cfg}: model_type deepseek_ref -> deepseek_v4")


def _prepare_download(args: ScriptArgs):
    """Download HF checkpoint + task dataset. Idempotent: hf skips existing blobs."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Only download if the user has NOT supplied a pre-existing checkpoint dir.
    # (prepare_single / train with --hf-checkpoint bypass this.)
    if args.hf_checkpoint is None:
        dest = f"{args.model_dir}/{args.model_name}"
        U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} " f"--local-dir {dest}")
    _ensure_4layer_model_type(args)
    _download_dataset(args)


@app.command()
@command_utils.dataclass_cli
def prepare_download(args: ScriptArgs):
    """Download HF checkpoint + dataset from HuggingFace. Run on one node (shared NFS)."""
    _prepare_download(args)


def _prepare_single(args: ScriptArgs):
    U = args.create_backend()
    _download_dataset(args)

    src = _hf_checkpoint_path(args)
    U.fp8_cast_bf16(
        path_src=src,
        path_dst=f"{args.model_dir}/{args.bf16_name}/",
    )


@app.command()
@command_utils.dataclass_cli
def prepare_single(args: ScriptArgs):
    """FP8 -> BF16 cast for Megatron. Needs --hf-checkpoint (or pre-downloaded). One node."""
    _prepare_single(args)


def _prepare_spmd(args: ScriptArgs):
    U = args.create_backend()
    is_4layer = args.model_name == "DeepSeek-V4-Flash-FP8-4layer"
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    extra_args = "--dsv4-impl miles --expert-tensor-parallel-size 1 --context-parallel-size 1 "
    if actor_num_nodes == 1 and is_4layer:
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        )
    elif actor_num_nodes == 1 and args.model_name == "DeepSeek-V4-Flash-FP8":
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 8 "
        )
    else:
        raise NotImplementedError(
            f"No verified SPMD conversion config for {args.model_name} "
            f"({actor_num_nodes} actor nodes x {actor_num_gpus_per_node} GPUs/node). "
            f"Please specify your conversion parallel config in `run_deepseek_v4.py`."
        )

    num_gpus_for_convert = actor_num_gpus_per_node
    if is_4layer:
        num_gpus_for_convert = min(num_gpus_for_convert, 4)

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=True if actor_num_nodes > 1 else False,
        num_nodes=actor_num_nodes,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@command_utils.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


def _prepare_cmd(args: ScriptArgs) -> dict[str, str]:
    if args.model_local_dir == args.model_dir:
        return {}

    copies = [
        command_utils.rsync_cmd(
            f"{args.model_dir}/{args.torch_dist_name}", f"{args.model_local_dir}/{args.torch_dist_name}"
        ),
        command_utils.rsync_cmd(f"{args.model_dir}/{args.model_name}", f"{args.model_local_dir}/{args.model_name}"),
    ]
    return {"trainer": " && ".join(copies)}


_PROBE_DEVICE = r"""
import json
import torch

props = torch.cuda.get_device_properties(0)
info = {
    "arch": getattr(props, "gcnArchName", "").split(":")[0],
    "name": props.name,
    "total_gib": props.total_memory / 1024**3,
    "hip": bool(torch.version.hip),
}
try:
    import transformer_engine
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import fp8

    def support(class_name, check_name):
        check = getattr(fp8, check_name, None)
        if not hasattr(recipe, class_name) or check is None:
            return [False, f"TE {transformer_engine.__version__} lacks {class_name} / {check_name}"]
        ok, reason = check()
        return [bool(ok), str(reason)]

    info["te_version"] = transformer_engine.__version__
    info["blockwise"] = support("Float8BlockScaling", "check_fp8_block_scaling_support")
    info["tensorwise"] = support("Float8CurrentScaling", "check_fp8_support")
except ImportError as exc:
    info["te_version"] = None
    info["blockwise"] = info["tensorwise"] = [False, f"Transformer Engine is not importable: {exc}"]
print(json.dumps(info))
"""


@functools.cache
def _probe_device() -> dict:
    """Describe the first visible GPU and what TE can run on it.

    Runs in a child process: this launcher lives for the whole job, and a device context
    opened here would sit on GPU 0 next to a colocated actor and engine until the job ends.
    """
    out = subprocess.run([sys.executable, "-c", _PROBE_DEVICE], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"GPU probe failed (exit {out.returncode}):\n{out.stderr[-4000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _is_gfx942() -> bool:
    return _probe_device()["arch"] == "gfx942"


def _resolve_colocate_memory_profile(args: ScriptArgs) -> str:
    """Pick the colocate memory split from the card, not from the node count."""
    if args.colocate_memory_profile != "auto":
        return args.colocate_memory_profile
    return "288gb" if _probe_device()["total_gib"] >= 256 else "192gb"


def _resolve_sglang_mem_fraction_static(args: ScriptArgs, memory_profile: str) -> str:
    if args.sglang_mem_fraction_static is not None:
        return repr(args.sglang_mem_fraction_static)
    # 192gb: the trainer is offloaded (see the optimizer args), so SGLang gets most of the card.
    # 0.85 left the second node's engine with no room for its KV cache next to the trainer's
    # residual allocations; at 0.75 the KV pool peaked at 22% during 8192-token rollouts.
    return "0.5" if memory_profile == "288gb" else "0.75"


def _resolve_fp8_recipe(requested: str) -> str:
    device = _probe_device()
    candidates = ("blockwise", "tensorwise") if requested == "auto" else (requested,)
    for candidate in candidates:
        if device[candidate][0]:
            print(
                f"[precision] FP8 E4M3 requested={requested} selected={candidate}; "
                f"TE={device['te_version']}; device={device['name']} ({device['arch']})"
            )
            return candidate
    raise RuntimeError(
        f"FP8 recipe '{requested}' is unsupported on {device['name']} ({device['arch']}, TE {device['te_version']}). "
        f"blockwise: {device['blockwise'][1]}; tensorwise: {device['tensorwise'][1]}. "
        "Pass --no-fp8-training to train in BF16; there is no automatic BF16 fallback."
    )


def _cp_derived_layout(
    total_gpus: int, *, pipeline_size: int, first_layers: int, last_layers: int, cp_size: int
) -> str:
    """Spend the tensor-parallel budget on context parallelism, keeping the pipeline split fixed.

    TP = total_gpus / (PP * CP), so DP collapses to 1. EP stays at 8 either way: Megatron builds
    the expert group from world_size % (etp * ep * pp), a quantity CP does not enter.
    """
    if total_gpus % (pipeline_size * cp_size):
        raise NotImplementedError(
            f"--context-parallel-size {cp_size} does not divide {total_gpus} GPUs over {pipeline_size} pipeline stages."
        )
    tensor_size = total_gpus // (pipeline_size * cp_size)
    config = f"--tensor-model-parallel-size {tensor_size} "
    if tensor_size > 1:
        # Megatron rejects sequence parallelism at TP=1 rather than ignoring it.
        config += "--sequence-parallel "
    config += (
        f"--pipeline-model-parallel-size {pipeline_size} "
        f"--decoder-first-pipeline-num-layers {first_layers} "
        f"--decoder-last-pipeline-num-layers {last_layers} "
        f"--context-parallel-size {cp_size} "
    )
    if cp_size > 1:
        # DeepSeek V4 has no zigzag CP path, and arguments.py asserts on the flag rather than
        # setting it for you.
        config += "--allgather-cp "
        # Contiguous CP under a causal mask gives the last rank several times the attention work
        # of the first, and the early ranks wait inside the collective; the first step adds
        # autotuning and JIT. The 10-minute default fires inside a healthy step.
        config += "--distributed-timeout-minutes 60 "
    return config + "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    total_gpus = actor_num_nodes * actor_num_gpus_per_node

    # Single-node smoke-test configs
    if actor_num_nodes == 1:
        if args.context_parallel_size not in (None, 1):
            raise NotImplementedError("--context-parallel-size is only wired for the multi-node layouts.")
        return (
            f"--tensor-model-parallel-size {actor_num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {actor_num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )

    if actor_num_gpus_per_node == 8:
        cp_size = args.context_parallel_size or (8 if total_gpus == 16 else 1)
        if total_gpus == 32:  # 4 nodes x 8 GPUs (MI355X, full Flash): PP4/EP8, 43 layers = 11+11+11+10
            if cp_size == 1:
                # The verified layout, kept verbatim: TP4/PP4/CP1/EP8.
                return (
                    "--tensor-model-parallel-size 4 "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 4 "
                    "--decoder-first-pipeline-num-layers 11 "
                    "--decoder-last-pipeline-num-layers 10 "
                    "--context-parallel-size 1 "
                    "--expert-model-parallel-size 8 "
                    "--expert-tensor-parallel-size 1 "
                )
            return _cp_derived_layout(total_gpus, pipeline_size=4, first_layers=11, last_layers=10, cp_size=cp_size)
        if total_gpus == 16:  # 2 nodes x 8 GPUs (MI300X, full Flash): PP2/EP8, 43 layers = 22+21
            return _cp_derived_layout(total_gpus, pipeline_size=2, first_layers=22, last_layers=21, cp_size=cp_size)

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `run_deepseek_v4._get_parallel_config`."
    )


def _get_fp8_training_recipe(args: ScriptArgs) -> str | None:
    return _resolve_fp8_recipe(args.fp8_recipe) if args.fp8_training else None


def _train(args: ScriptArgs, fp8_recipe: str | None = None):
    U = args.create_backend()
    print(f"[precision] fp8_training={args.fp8_training}")
    if fp8_recipe is None:
        fp8_recipe = _get_fp8_training_recipe(args)
    memory_profile = _resolve_colocate_memory_profile(args)
    is_gfx942 = _is_gfx942()
    print(
        f"running on {args.num_nodes} nodes "
        f"({args.actor_num_nodes} actor nodes x {args.actor_num_gpus_per_node} GPUs/node, "
        f"{args.rollout_num_gpus} rollout GPUs, colocate={args.colocate}, "
        f"memory_profile={memory_profile}, arch={_probe_device()['arch']})"
    )
    _ensure_4layer_model_type(args)

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    # train reads from model_local_dir (full-train fills it in the same way).
    hf_checkpoint = args.hf_checkpoint or f"{args.model_local_dir}/{args.model_name}"
    ckpt_args = f"--hf-checkpoint {hf_checkpoint} " f"--ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} " f"--save {load_save_path} " "--save-interval 20 " "--save-retain-interval 20 "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                # DAPO prompts ask for "Answer: ...", not \boxed{}; the math grader extracts only
                # boxed answers and scores correct completions 0.
                "--rm-type dapo --reward-key score --eval-reward-key acc "
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len 8192 "
                """--apply-chat-template-kwargs '{"thinking_mode":"thinking"}' """
            )
            # AIME prompts carry no "Answer:" instruction, so responses box their answer; the
            # DAPO grader would score every one -1. dapo_boxed keeps DAPO's {score, acc} on the
            # last \boxed{}, so --eval-reward-key acc stays an accuracy.
            eval_config = (
                "eval:\n  datasets:\n    - name: aime\n"
                f"      path: {args.data_dir}/aime-2024/aime-2024.jsonl\n      rm_type: dapo_boxed\n"
            )
            eval_args += (
                f"--eval-config {command_utils.encode_pseudo_file(eval_config)} "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 4096 "
            )
        case "gsm8k":
            rollout_args += (
                "--rm-type math "
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    perf_args = _get_parallel_config(args)

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--use-precision-aware-optimizer " "--overlap-cpu-optimizer-d2h-h2d "
        )
        if args.colocate and args.actor_num_nodes == 2:
            # 16 GPUs cannot hold the optimizer at all: offload all of it, offload the trainer
            # while the engines run, and rebuild the BF16 params from the CPU masters instead of
            # backing them up (the check hashes the first two rebuilds). The FP32 masters and
            # both moments stay FP32; streaming the gradients one at a time is what keeps a full
            # FP32 CPU gradient copy from pushing the ~1.8 TB per node past host memory.
            optimizer_args += (
                "--optimizer-offload-fraction 1.0 --offload-train "
                "--optimizer-cpu-streaming-gradients --no-pin-cpu-grads --no-pin-cpu-params "
                "--rematerialize-param-from-master-weight --check-rematerialize-param-from-master-weight "
            )
        elif args.actor_num_nodes == 4:
            if memory_profile == "288gb":
                # 4-node PP4 memory balance: partial optimizer offload (keep ~25% on GPU) + keep train
                # weights on GPU; pair with --sglang-mem-fraction-static 0.5.
                optimizer_args += "--optimizer-offload-fraction 0.75 " "--no-offload-train "
            else:
                optimizer_args += "--optimizer-offload-fraction 1.0 " "--offload-train "

    sglang_world_size = 4
    sglang_tp_size = 4
    sglang_dp_size = 1
    sglang_ep_size = 4
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-dp-size {sglang_dp_size} "
        f"--sglang-ep-size {sglang_ep_size} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
    )
    if args.task == "dapo_aime":
        # The checkpoint declares max_position_embeddings=1048576 and SGLang would otherwise size
        # every engine for that. 16384 covers the longest DAPO prompt (~1.2k tokens) plus the
        # 8192-token response; SGLang rejects rather than truncates a request that exceeds it.
        sglang_args += "--sglang-context-length 16384 "
    if is_gfx942:
        # AITER's MoE and custom all-reduce accumulate with BF16 atomics on gfx942, so identical
        # forwards differ. The Triton MoE runner still executes the FP8 expert GEMMs.
        sglang_args += "--sglang-moe-runner-backend triton --sglang-disable-custom-all-reduce "
        # A GRPO group shares its prompt, so cache-aware routing (imbalance threshold 64) can hold
        # a rollout's requests on one of the four engines; the 2-node recipe spreads them evenly.
        sglang_args += "--sglang-router-policy round_robin "
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
        # unified_kv lives in compressor_v2 only; on HIP the v1 path leaves
        # compress_kv_pool unset and the memory pool asserts on it.
        "SGLANG_OPT_USE_COMPRESSOR_V2": "true",
        "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
        "SGLANG_OPT_USE_JIT_NORM": "true",
        "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "AITER_BF16_FP8_MOE_BOUND": "0",
    }
    if is_gfx942:
        extra_env_vars |= {
            # One hardware queue per process: the colocated actor and engine otherwise
            # oversubscribe the card's queues.
            "GPU_MAX_HW_QUEUES": "1",
            "SGLANG_SET_CPU_AFFINITY": "0",
            # Max-autotune re-selects inductor kernels per process, so the trainer and the
            # engines would not run the same kernels for the same op.
            "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
            "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
        }

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--train-memory-margin-bytes {3221225472 if memory_profile == '288gb' else 1073741824} "
        f"--sglang-mem-fraction-static {_resolve_sglang_mem_fraction_static(args, memory_profile)} "
        "--sglang-watchdog-timeout 1800 "  # ROCm: slow aiter gemm tune under colocate; avoid watchdog SIGQUIT
        "--accumulate-allreduce-grads-in-fp32 "
        "--dsv4-impl miles "  # ROCm has no cudnn/flash_mla path for the megatron impl
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format thd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    if is_gfx942:
        # Match the gfx942 rollout: FP32 router logits (SGLang pins them on gfx942), a BF16 KV
        # cache (the FP8 unified KV pool needs gfx95), and indexer K scales stored unrounded.
        misc_args += "--moe-router-dtype fp32 --dsv4-kv-qat off --dsv4-index-qat fp8_dynamic "
    if args.colocate:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {args.rollout_num_gpus} "

    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "

    if args.enable_mis:
        misc_args += (
            "--use-tis "
            "--custom-config-path examples/infra_features/train_infer_mismatch_helper/mis.yaml "
            "--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "

    if args.debug_train_run_id is not None:
        if args.debug_train_rollout_id is None:
            args.debug_train_rollout_id = 1
        misc_args += (
            f"--load-debug-rollout-data "
            f"{args.debug_data_root}/{args.debug_train_run_id}/dump_details/rollout_data/{args.debug_train_rollout_id}.pt "
        )
        misc_args += "--debug-train-only "

    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
        # Skip indexer-replay for now
        # misc_args += "--use-rollout-indexer-replay "

    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    if args.fp8_training:
        misc_args += (
            "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " f"--fp8-recipe {fp8_recipe} "
        )
        if fp8_recipe == "blockwise":
            misc_args += """--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"0"}' """
        # ROCm TE MoE FP8 lacks fused wgrad accumulation; disable the fusion.
        misc_args += "--no-gradient-accumulation-fusion "

    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
        # gfx950: use RCCL all-gather for speculative decoding; aiter can deadlock.
        extra_env_vars |= {"SGLANG_USE_AITER_AG": "false"}

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
        prepare_cmd=_prepare_cmd(args),
    )


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Run training. Assumes data/model/torch_dist are already prepared on {model_local_dir}."""
    _train(args)


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # Resolve the recipe first, so an unsupported one fails before the download and conversion.
    fp8_recipe = _get_fp8_training_recipe(args)
    _prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"

    _train(args, fp8_recipe=fp8_recipe)


if __name__ == "__main__":
    app()
