"""
NVIDIA Nemotron-3-Ultra-550B-A55B GRPO RL training script.

=====================

nemotron_h is a hybrid Mamba2 + Attention + latent-MoE architecture (108 layers,
512 experts top-22, moe_latent_size=2048). It loads through NVIDIA
``megatron.bridge`` (``--megatron-to-hf-mode bridge``) plus the miles
NemotronHBridge MoE/latent shim in ``miles_plugins/megatron_bridge/nemotron_h.py``.

Tested on H200.

Please use the `radixark/miles:dev` docker image.

=====================

Args:
  --model-name: Model variant to use.
      NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16          Full 108-layer model (16 nodes)
      NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer   4-layer slice (single node)
  --num-nodes: Number of nodes. Determines the parallelism config (see _parallelism).
  --mode: "normal" or "debug_minimal" (shorter response length for quick testing)
  --check-weight-update-equal: Assert the Megatron -> SGLang weight sync restores
      every tensor exactly (poisons SGLang's weights first).

Weights load straight from the HF checkpoint: miles' load_checkpoint dispatches on
what --load points at, and an HF directory routes to _load_checkpoint_hf, i.e. the
same megatron.bridge path this model is mapped through. No offline Megatron dist
conversion step is needed.

Mamba n_groups=8 caps attention/mamba tensor-parallel at 8 (n_groups % tp == 0).
The 550B (~1.1TB bf16) does not fit one 8-GPU SGLang engine, so rollout uses
32-GPU engines with EP=32 + DP-attention (dp=4) so attention/mamba run at
attn_tp = 32/4 = 8.

Rollout routing-replay (--use-rollout-routing-replay) is NOT enabled for the
108-layer Ultra yet (the routing capturer shape needs a fix for per-layer top-22
under DP-attention); train/rollout logprob diff is ~0.01 without it.

=====================

I. Usage for single node 4-layer smoke test:
  `python scripts/run_nemotron_3_ultra_550b_a55b.py full-train \
      --model-name NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer --num-nodes 1`

=====================

II. Usage for the full model (16 nodes):

  1. Setup containers on all nodes

  2. Start Ray cluster on all nodes

  3. Download model/data. Run on **head node**.
       `python scripts/run_nemotron_3_ultra_550b_a55b.py prepare --num-nodes 16`

  4. Run training. Execute on head node; uses Ray internally.
       `python scripts/run_nemotron_3_ultra_550b_a55b.py train --num-nodes 16`
"""

import re
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

FULL_MODEL_NAME = "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "nvidia"
    model_name: str = FULL_MODEL_NAME
    megatron_model_type: str = "nemotron-3-ultra-550b-a55b"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H200", "B200", "GB300"] = "auto"
    enable_eval: bool = False
    enable_optimizer_offload: bool = True
    check_weight_update_equal: bool = False
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    max_tokens_per_gpu: int = 1024
    save_interval: int = 50
    skip_saving: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"nemotron-3-ultra-550b-a55b-{m.group(1)}layer"
        elif self.model_name != FULL_MODEL_NAME:
            raise NotImplementedError(f"{self.model_name} is not supported")


def _is_pruned(args: ScriptArgs) -> bool:
    return re.search(r"(\d+)layer", args.model_name) is not None


def _hf_checkpoint(args: ScriptArgs) -> str:
    return f"{args.model_dir}/{args.model_name}"


def _parallelism(args: ScriptArgs) -> tuple[int, int, int, int]:
    """(tp, pp, ep, etp). Mamba n_groups=8 caps attention/mamba tp at 8."""
    total_gpus = args.num_nodes * args.num_gpus_per_node
    if _is_pruned(args):
        # A few layers fit on one node; give every rank to expert parallel
        # (512 experts / EP8 = 64 per rank) and keep attention/mamba at tp 1.
        return 1, 1, total_gpus, 1
    if total_gpus == 128:
        return 8, 4, 32, 1
    raise NotImplementedError(f"No parallelism config for {total_gpus} GPUs with {args.model_name}")


def _sglang_args(args: ScriptArgs) -> str:
    total_gpus = args.num_nodes * args.num_gpus_per_node
    if _is_pruned(args):
        gpus_per_engine, dp_size, mem_fraction = args.num_gpus_per_node, 2, 0.6
    else:
        # The 550B does not fit one 8-GPU engine.
        gpus_per_engine, dp_size, mem_fraction = 32, 4, 0.7
    assert total_gpus % gpus_per_engine == 0, f"{total_gpus=} must be a multiple of {gpus_per_engine=}"
    attn_tp = gpus_per_engine // dp_size
    assert (
        gpus_per_engine % dp_size == 0 and 8 % attn_tp == 0
    ), f"attn_tp = {gpus_per_engine}/{dp_size} = {attn_tp} must divide Mamba n_groups=8"
    return (
        f"--rollout-num-gpus-per-engine {gpus_per_engine} "
        f"--sglang-ep-size {gpus_per_engine} "
        f"--sglang-dp-size {dp_size} "
        "--sglang-enable-dp-attention "
        f"--sglang-mem-fraction-static {mem_fraction} "
    )


def _prepare_download(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} --local-dir {_hf_checkpoint(args)}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _execute_train(args: ScriptArgs):
    tp, pp, ep, etp = _parallelism(args)
    total_gpus = args.num_nodes * args.num_gpus_per_node

    ckpt_args = (
        # --ref-load at an HF directory routes load_checkpoint to _load_checkpoint_hf,
        # i.e. megatron.bridge; --load defaults to it when there is nothing to resume.
        f"--hf-checkpoint {_hf_checkpoint(args)} "  # tokenizer + SGLang rollout
        f"--ref-load {_hf_checkpoint(args)} "
        "--megatron-to-hf-mode bridge "
    )
    if not args.skip_saving:
        ckpt_args += (
            f"--save {args.output_dir}/{args.run_id}/checkpoints "
            f"--save-interval {args.save_interval} "
            "--no-save-optim "  # weights-only
        )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {256 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        f"--pipeline-model-parallel-size {pp} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ep} "
        f"--expert-tensor-parallel-size {etp} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 128 "
    )
    if tp > 1:
        perf_args += "--sequence-parallel "

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
    if args.enable_optimizer_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend auto "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {total_gpus} "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{_sglang_args(args)} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            # The nemotron DP-attention path uses existing kernels; skip the
            # blanket sgl-kernel version guard.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Full pipeline: download, train."""
    _prepare_download(args)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download model/data (run on head node)."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
