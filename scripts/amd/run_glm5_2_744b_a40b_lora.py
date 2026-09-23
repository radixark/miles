"""GLM-5.2 744B-A40B GRPO LoRA RL for AMD (MI350X / MI355X), single- or multi-node.

ROCm counterpart of scripts/run_glm5_2_744b_a40b_lora.py, using the multi-node shape of
examples/swe-agent-harbor-docker/run_glm52_lora_tb2_daytona.py: TP stays intra-node and
EP spans the whole world, so the 744B base is sharded across every GPU in the job.

Everything at the model level is the CUDA recipe's: the same checkpoint, registry, chat
template, reward, LoRA targets, GRPO settings, R3 routing replay, and the task-aware
sequence budget. Only kernel selection differs, because that is the only thing the
vendor actually changes:

  - SGLang serves the DSA attention with the TileLang NSA kernels; flashmla has no HIP
    build. Everything else in the engine config matches.

Two CUDA-recipe flags are absent for reasons that are NOT ROCm-specific -- the in-tree
multi-node CUDA LoRA launcher drops them too:
  - dp-attention: its scheduler runs per-iteration collectives that colocate weight
    syncs desync, deadlocking the engines.
  - --sglang-moe-dense-tp-size 1: leaves the dense MLP unsharded while the LoRA buffers
    are still sized input_dim / tp_size.

``--dsa-attention-backend tilelang`` needs a Megatron-Bridge that handles megatron-core's
unfused DSA kv up-projection; without it the absorb reads a ``layer_norm_weight`` that
does not exist. ``megatron`` avoids the absorb but materializes the fp32 [b, np, S, S]
scores (~59 GiB/rank at S=4096) and cannot recompute activations, so it caps near S=4096.

Model variants (the HF checkpoint must be the native glm_moe_dsa config):
  GLM-5.2         full 744B model (zai-org/GLM-5.2), multi-node
  GLM-5.2_5layer  5-layer prune (Pinaster/GLM-5.2_5layer), mechanics smoke only -- it
                  scores zero reward on every sample, so it cannot show learning

Examples:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py prepare --model-name GLM-5.2
  # 2 nodes, Ray already up across them, MILES_SCRIPT_EXTERNAL_RAY=1:
  python scripts/amd/run_glm5_2_744b_a40b_lora.py train --model-name GLM-5.2 --num-nodes 2
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = {
    "GLM-5.2": "zai-org/GLM-5.2",
    "GLM-5.2_5layer": "Pinaster/GLM-5.2_5layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.2": "glm5.2-744B-A40B_lora",
    "GLM-5.2_5layer": "glm5.2-744B-A40B_5layer_lora",
}

# Attn + MLA + MLP/MoE, excluding two things.
#
# The DSA indexer (wq_b/wk/weights_proj): on megatron the indexer adapter only gets a tiny
# aux-loss gradient, on tilelang none at all.
#
# The expert down_proj: the CUDA launcher's default list carries it, but the published
# 64-GPU run did not (wandb group `no-down-proj-260702-0948`), and its ablation is the
# reason -- with down_proj, logprob abs_diff drifts to 0.058 by step 43 (KL 0.02); without
# it, flat 0.0044-0.0077 (KL ~2e-4) at no reward cost. The down-proj delta lands on the
# residual stream and amplifies the bf16 kernel-numerics gap, which on ROCm is the larger
# of the two gaps to begin with. Add it back only with --target-modules.
_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"

# (seq_length, rollout_max_response_len); the difference is the prompt budget. These match
# the CUDA *multi-node* preset (run_glm5_lora_multinode_full.sh), not the single-node
# launcher's 8192/4096 -- the multi-node preset is the configuration the published 64-GPU
# run used. Its sweep on the full 744B put raw_reward at 0.0 for seq 1024 (everything
# truncated), 0.125 at 2048, 0.25 at 4096 and 0.3125 at 8192, and picked 4096: 8192 buys
# +0.06 reward for double the rollout time.
# gsm8k answers are short; its seq is set explicitly rather than left at the model's 1 Mi
# native context, which would size SGLang's req_to_token pool at 2.1 GiB.
_TASK_SEQ = {"dapo-math": (4096, 3584), "gsm8k": (1024, 512)}

# The SGLang TileLang NSA decode kernel tiles at least 4 attention heads per rank, and
# the engine's attention TP is its GPU count: GLM-5.2's 64 heads allow at most 16. A
# 32-GPU engine fails decode CUDA-graph capture with "head_per_block must divide heads".
_MAX_ENGINE_GPUS = 16


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["GLM-5.2", "GLM-5.2_5layer"] = "GLM-5.2"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None
    task: Literal["dapo-math", "gsm8k"] = "dapo-math"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # tilelang is O(S * topk) and allows activation recompute; megatron is a dense
    # O(S**2) reference that caps near S=4096. The matching --qkv-format is derived.
    dsa_attention_backend: Literal["megatron", "tilelang"] = "tilelang"
    # R3 rollout routing replay (arxiv 2510.11370)
    use_r3: bool = True

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _TARGET_MODULES
    # Required for true on-policy under colocate (OFF -> KL ~1.0 vs ~1e-4).
    lora_base_cpu_backup: bool = True
    # MoE-expert LoRA layout: shared-outer when True, per-expert when False.
    experts_shared_outer_loras: bool = True

    num_rollout: int = 40
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    # 0 => the task default above
    seq_length: int = 0
    rollout_max_response_len: int = 0
    lr: float = 1e-5

    # 0 => min(world, _MAX_ENGINE_GPUS). Under colocate + LoRA the engine mirrors its
    # base-weight shard into host RAM, costing ckpt_size * gpus_per_node / engine_gpus per
    # node: for bf16 744B that is 1.4 TiB/node at engine=8 (which OOMs a 2.75 TiB node
    # once the paused trainer's disk spill has filled the page cache) and half that at
    # engine=16.
    rollout_num_gpus_per_engine: int = 0
    sglang_mem_fraction_static: float = 0.85
    # 0 => seq_length, i.e. --rollout-max-context-len, so this can never be what truncates a
    # rollout: miles clamps max_new_tokens to rollout_max_context_len - prompt_len before the
    # request reaches the engine. It is set because SGLang otherwise serves the checkpoint's
    # native 1 Mi context and sizes req_to_token at max_running_requests * context_len inside
    # the memory-saver region, which is what made the ROCm Triton cache-index writer fail on
    # a non-device pointer at the first prefill.
    sglang_context_length: int = 0
    # Only rollout_batch_size * n_samples_per_prompt requests are ever in flight.
    sglang_max_running_requests: int = 0  # 0 => that product

    # See misc_args: 10 (the miles default) is shorter than TileLang's first-use JIT
    # compile, and the watchdog then kills a slow start as if it were a hang.
    distributed_timeout_minutes: int = 60

    # The colocated engine's first memory release copies its base-weight shard to pinned
    # host memory at ~1 GiB/s: ~13 min per rank at 2 nodes. miles bounds a cell tick at
    # 120 s and re-issues a cancelled release, each retry pinning another host buffer.
    rollout_cell_tick_timeout: float = 3600.0
    rollout_cell_init_timeout: float = 5400.0

    # Running with --no-offload-train does fit at >=2 nodes (engine 0.56 + trainer
    # ~105 GiB of 288), but weight sync still asks torch_memory_saver for a host backup,
    # which throws HIP invalid-argument when the trainer was never paused.
    offload_train: bool = True
    save_interval: int = 10
    save_dir: str = "/root/shared_data"
    # Where the paused trainer spills while the engine runs. Under colocate + LoRA SGLang
    # keeps its own host mirror of the base weights (~1.4 TiB for bf16 744B), and a pinned
    # host copy of the paused actor on top of that overruns a 2.75 TiB node. Must be
    # node-local and must not be tmpfs.
    offload_train_disk_dir: str = ""  # "" => {save_dir}/train_offload
    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        seq, resp = _TASK_SEQ[self.task]
        if self.seq_length == 0:
            self.seq_length = seq
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = resp
        if not self.offload_train_disk_dir:
            self.offload_train_disk_dir = f"{self.save_dir}/train_offload"
        if self.sglang_context_length == 0:
            self.sglang_context_length = self.seq_length
        if self.sglang_max_running_requests == 0:
            self.sglang_max_running_requests = self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _set_rocm_environment() -> None:
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _resolve_num_gpus(args: ScriptArgs) -> tuple[str, int]:
    hardware = U.resolve_hardware(args)
    return hardware, args.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[hardware]


def _parallel_args(args: ScriptArgs, num_gpus: int) -> str:
    """TP intra-node so its all-reduce stays on the local fabric, EP across the whole
    world so the 744B experts fit. Megatron needs EP * ETP == TP * DP, which holds for
    any node count at PP = CP = 1.
    """
    world_size = args.num_nodes * num_gpus
    qkv_format = "thd" if args.dsa_attention_backend == "tilelang" else "bshd"
    # Only thd carries packed_seq_params, which is where the cross-layer DSA top-k rides,
    # so only thd can recompute activations: under bshd a recomputed skip layer would read
    # a stale anchor top-k.
    recompute = (
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        if qkv_format == "thd"
        else ""
    )
    return (
        f"--tensor-model-parallel-size {num_gpus} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {world_size} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 {recompute}"
    )


def _download_inputs(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command_cpu(f"hf download {_HF_REPO[args.model_name]} --local-dir {args.model_dir}/{args.model_name}")
    match args.task:
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _get_wandb_args(args: ScriptArgs) -> str:
    if not args.enable_wandb:
        return ""
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args and args.wandb_team:
        wandb_args += f"--wandb-team {shlex.quote(args.wandb_team)} "
    return wandb_args


def _execute(args: ScriptArgs) -> None:
    _set_rocm_environment()
    # execute_train only ever starts a local head, so a multi-node run needs the cluster
    # to exist already. Without this the job asks for num_nodes * num_gpus placement slots
    # against one node's GPUs and waits on pg.ready() forever with no diagnostic.
    if args.num_nodes > 1 and not os.environ.get("MILES_SCRIPT_EXTERNAL_RAY"):
        raise SystemExit(
            f"--num-nodes {args.num_nodes} needs a Ray cluster spanning all {args.num_nodes} nodes "
            "before this runs: start the head, join every worker with --num-gpus equal to the "
            "node's GPU count, wait for the full GPU count, then re-run with "
            "MILES_SCRIPT_EXTERNAL_RAY=1 and MASTER_ADDR set to the head's fabric IP."
        )
    hardware, num_gpus = _resolve_num_gpus(args)
    world_size = args.num_nodes * num_gpus
    engine_gpus = args.rollout_num_gpus_per_engine or min(world_size, _MAX_ENGINE_GPUS)
    print(
        f"[run] GLM-5.2 LoRA on {hardware}: model={args.model_name}, task={args.task}, "
        f"{args.num_nodes} node(s) x {num_gpus} GPUs (TP={num_gpus} EP={world_size}), "
        f"dsa={args.dsa_attention_backend}, seq={args.seq_length}, rollout tp={engine_gpus}"
    )

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )
    # Megatron asserts save_interval > 0 whenever --save is set, so the pair goes together.
    if args.save_interval > 0:
        ckpt_args += f"--save {args.save_dir}/{args.run_id} --save-interval {args.save_interval} "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" --no-gradient-accumulation-fusion '
    )
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    match args.task:
        case "dapo-math":  # {prompt, label} jsonl
            data_args = f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "
        case "gsm8k":  # {messages, label} parquet
            data_args = f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "

    rollout_args = (
        f"{data_args}--label-key label --apply-chat-template --rollout-shuffle --balance-data "
        "--rm-type math --rollout-temperature 1.0 "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--seq-length {args.seq_length} --rollout-max-context-len {args.seq_length} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )

    # Routing replay only: --use-rollout-indexer-replay is debug-only and its
    # ~78-128 GB/rank host buffer OOMs the colocate pod.
    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-ep-size {engine_gpus} "
        "--sglang-attention-backend nsa "
        # ROCm: flashmla_sparse / flashmla_kv have no HIP build.
        "--sglang-nsa-prefill-backend tilelang --sglang-nsa-decode-backend tilelang "
        "--sglang-page-size 64 "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--sglang-cuda-graph-max-bs {args.sglang_max_running_requests} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-chunked-prefill-size {min(8192, 2048 * engine_gpus)} --sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        # Required: without it sglang miscounts the gate_up slices -> engine-init crash.
        f"--sglang-max-lora-rank {args.lora_rank} --sglang-lora-backend triton "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss "
        # The RCCL 2.27.7 AllToAll kernel deadlocks on this model's EP dispatch payload
        # under colocate; the same exchange as ordered isend/irecv pairs completes.
        "--moe-token-dispatcher-type alltoall --moe-ep-p2p-alltoall --colocate "
        f"--rollout-cell-tick-timeout {args.rollout_cell_tick_timeout} "
        f"--rollout-cell-init-timeout {args.rollout_cell_init_timeout} "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} "
        # Trainer offload spills to node-local disk rather than pinned host RAM: with
        # --lora-base-cpu-backup the engine already mirrors the base weights on the host,
        # and both copies together overrun the node (Ray killed workers on OOM at the
        # first prefill).
        + (
            "--offload-train-target disk "
            f"--offload-train-disk-dir {args.offload_train_disk_dir} --offload-train-disk-chunk-mb 256 "
            if args.offload_train
            else "--no-offload-train "
        )
        +
        # bf16 has no grad scaler, so Megatron reports found_inf=False unconditionally and a
        # non-finite grad norm reaches the step, where clipping writes NaN into every adapter
        # tensor. This routes the step through miles' own guard, which skips it instead.
        "--no-check-for-nan-in-loss-and-grad "
        # train/entropy_loss is hardcoded 0.0 without this, and falling entropy is the
        # earliest warning of policy collapse.
        "--observe-training-entropy "
        # miles defaults this to 10 minutes, which is shorter than this recipe's startup.
        # The TileLang NSA kernels are JIT-compiled per rank on first use, and a rank that
        # is still compiling while its peers have entered the first collective on
        # TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP trips the watchdog at exactly 600 s --
        # killing every rank on what is a slow start, not a hang. Observed on both 2 and 4
        # nodes, and intermittent because a warm TileLang cache changes the compile time.
        f"--distributed-timeout-minutes {args.distributed_timeout_minutes} "
    )

    train_args = (
        f"{ckpt_args}{lora_args}{rollout_args}{optimizer_args}{grpo_args}{r3_args}"
        f"{_get_wandb_args(args)}{_parallel_args(args, num_gpus)}{sglang_args}{misc_args}{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences.
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "SGLANG_NSA_FORCE_MLA": "1",
            # No garbage_collection_threshold: it makes the caching allocator release
            # blocks mid-forward, and each release device-synchronises. expandable_segments
            # breaks torch_memory_saver under colocate. torch 2.9 ignores the old
            # PYTORCH_CUDA_ALLOC_CONF name.
            "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
            # A stalled collective leaves no Python frame (the dispatcher enqueues it
            # asynchronously); the watchdog dump is what names it.
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the model checkpoint and the task dataset. Run once per node."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Run GRPO LoRA training using prepared inputs."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download inputs and run GRPO LoRA training."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
