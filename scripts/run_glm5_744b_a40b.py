import re
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-5"
    megatron_model_type: str = "glm5-744B-A40B"
    num_gpus_per_node: int = 8
    fp8_rollout: bool = False
    enable_eval: bool = False
    enable_mtp: bool = False
    enable_pd: bool = False
    enable_optimizer_offload: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/local_data"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        assert not self.fp8_rollout, "fp8 recipe not implemented"
        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"glm5-744B-A40B_{m.group(1)}layer"


def _is_pruned(args: ScriptArgs):
    return re.search(r"(\d+)layer", args.model_name) is not None


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Skip model download for pruned variants (assumed to already exist in model_dir)
    if not _is_pruned(args):
        U.exec_command(
            f"huggingface-cli download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
        )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_megatron_ckpt(args: ScriptArgs):
    extra_args = "--tensor-model-parallel-size 1 " "--expert-tensor-parallel-size 1 "
    num_gpus_per_node = args.num_gpus_per_node
    multinode = True
    num_nodes = None

    num_layers_match = re.search(r"(\d+)layer", args.model_name)
    if num_layers_match and int(num_layers_match.group(1)) <= 4:
        extra_args += (
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 "
        )
        num_gpus_per_node = min(4, num_gpus_per_node)
        multinode = False
    elif num_layers_match:
        extra_args += "--expert-model-parallel-size 4 "
        num_nodes = 2
    else:
        extra_args += (
            "--pipeline-model-parallel-size 4 "
            "--expert-model-parallel-size 32 "
            "--decoder-last-pipeline-num-layers 18 "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        megatron_path=args.megatron_path,
    )


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}_torch_dist",
        path_dst=f"{args.model_local_dir}/{args.model_name}_torch_dist",
    )
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
    )


def _execute_train(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{args.model_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else (2048 if args.enable_pd else 32768)} "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 1 "

    if args.num_nodes == 1:
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes >= 16: # slime's setting
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 4 "
            "--decoder-last-pipeline-num-layers 18 "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 32 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        raise NotImplementedError

    perf_args += (
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if _is_pruned(args) else 16384} "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    if args.enable_pd:
        sglang_decode_max_bs = 8
        sglang_world_size = 64
    else:
        sglang_decode_max_bs = 256
        sglang_world_size = 8

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.70 "
        "--sglang-enable-dp-attention "
        f"--sglang-ep-size {sglang_world_size} "
        f"--sglang-dp-size {sglang_world_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
    )
    if args.fp8_rollout:
        sglang_args += (
            "--sglang-moe-a2a-backend deepep "
            "--sglang-deepep-mode auto "
        )
    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
    if args.enable_pd:
        sglang_args += "--prefill-num-servers 1 "
    sglang_args += (
        # dsa
        "--sglang-page-size 64 "
        "--sglang-nsa-decode-backend flashmla_sparse "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
        # concurrency
        f"--sglang-max-running-requests {512 if args.enable_pd else sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-chunked-prefill-size {131072 if args.enable_pd else sglang_world_size * sglang_decode_max_bs} "
        "--sglang-watchdog-timeout 3600 "
    )
    sglang_extra_env_vars = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": f"{32 if args.enable_pd else sglang_decode_max_bs}",
    }

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        # use deepep for megatron
        "--moe-enable-deepep "
        "--moe-token-dispatcher-type flex "
        "--allgather-cp "
        # ------------
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "

        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
        "--disable-weights-backuper "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            **sglang_extra_env_vars,
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _prepare_download(args)
    _prepare_megatron_ckpt(args)
    _prepare_cp(args)
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
