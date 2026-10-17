"""GLM-5.3-Flash BF16 image SFT with a trainable or frozen visual tower.

The BF16 HF checkpoint must contain model.visual weights and processor metadata.
The language checkpoint must already be converted to Megatron torch_dist. For
multi-node runs, join Ray externally and set MILES_SCRIPT_EXTERNAL_RAY=1.

Args:
  --model-name: GLM-5.3-Flash-4layer (8 GPUs) or GLM-5.3-Flash (32/64 GPUs).
  --hf-checkpoint / --ref-load: HF source and converted language checkpoint.
  --prompt-data: JSONL/Parquet containing messages and an images list.
  --num-nodes / --num-gpus-per-node: Training topology.
  --num-rollout: Number of SFT optimizer steps, one step per rollout.
  --rollout-batch-size / --global-batch-size: Number of conversations per step.
  --output-dir: Training checkpoint destination; reused for resume.
  --save-interval: Checkpoint interval in optimizer steps.
  --vision-trainable: Visual parameters to train; none, projector or full.
  --freeze-language: Freeze the LLM and train only the selected visual scope.
  --vision-recompute: Recompute the ViT blocks instead of storing their
    activations. Defaults to on for --vision-trainable full, which is the only
    scope that trains those blocks and the only one whose activations overflow a
    95 GiB card at realistic image sizes.
  --rollout-seed: Data shuffle seed; two runs must share it to be comparable.
  --optimizer-cpu-offload: Offload Adam states to host RAM.
  --rotary-base: Override the RoPE value read from local checkpoint metadata.

Example:
  python scripts/run_glm5_3_flash_vision_sft.py \
    --num-nodes 1 --num-gpus-per-node 8 \
    --hf-checkpoint /root/models/GLM-5.3-Flash-4layer-BF16 \
    --ref-load /root/models/glm5.3-flash-4layer_torch_dist \
    --prompt-data /root/datasets/glm53_image_sft.jsonl \
    --output-dir /root/runs/glm53_image_sft --num-rollout 2 --save-interval 1
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_MODEL_TYPES = {
    "GLM-5.3-Flash": "glm5.3-flash",
    "GLM-5.3-Flash-4layer": "glm5.3-flash-4layer",
}


def _checkpoint_rotary_base(hf_checkpoint: str) -> int:
    config_path = Path(hf_checkpoint) / "config.json"
    if not config_path.is_file():
        return 10000
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config.get("text_config", config).get("rope_theta", 10000)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_name: Literal["GLM-5.3-Flash", "GLM-5.3-Flash-4layer"] = "GLM-5.3-Flash-4layer"
    num_gpus_per_node: int = 8
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    hf_checkpoint: str | None = None
    ref_load: str | None = None
    prompt_data: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 100
    rollout_batch_size: int = 16
    global_batch_size: int = 16
    max_tokens_per_gpu: int = 8192
    rollout_seed: int = 42
    save_interval: int = 100
    lr: float = 1e-5
    rotary_base: int | None = None
    vision_trainable: Literal["none", "projector", "full"] = "projector"
    freeze_language: bool = False
    # None means "ask the scope"; resolved by resolved_vision_recompute, not in
    # __post_init__, so copying this args object and changing a field still resolves.
    vision_recompute: bool | None = None
    optimizer_cpu_offload: bool = True
    extra_args: str = ""

    @property
    def resolved_vision_recompute(self) -> bool:
        """Whether to recompute the ViT blocks, from the scope unless asked otherwise.

        Deliberately not resolved in __post_init__. `dataclasses.replace(args,
        vision_trainable="full")` runs __init__, then __post_init__ with the old
        vision_recompute already copied over, so a field written there would leave the
        override looking explicit and silently disable recomputation for the one scope
        that cannot fit without it.
        """
        if self.vision_recompute is not None:
            return self.vision_recompute
        # Only "full" trains the ViT blocks, and at realistic image sizes their stored
        # activations overflow a 95 GiB card, so that scope needs recomputation to fit.
        return self.vision_trainable == "full"

    def __post_init__(self):
        self.hf_checkpoint = self.hf_checkpoint or f"{self.model_dir}/{self.model_name}-BF16"
        self.ref_load = self.ref_load or f"{self.model_dir}/{self.megatron_model_type}_torch_dist"
        self.prompt_data = self.prompt_data or f"{self.data_dir}/glm53_image_sft.jsonl"
        self.rotary_base = (
            self.rotary_base if self.rotary_base is not None else _checkpoint_rotary_base(self.hf_checkpoint)
        )
        _ = self.parallel_args
        if self.num_rollout < 1 or self.save_interval < 1:
            raise ValueError("num_rollout and save_interval must be positive")
        if self.rollout_batch_size != self.global_batch_size or self.global_batch_size < 1:
            raise ValueError("One-step image SFT requires equal positive rollout and global batch sizes")
        if self.global_batch_size % self.data_parallel_size:
            raise ValueError(f"global_batch_size must be divisible by DP={self.data_parallel_size}")
        if self.freeze_language and self.vision_trainable == "none":
            raise ValueError("freeze_language requires a trainable visual scope")

    @property
    def megatron_model_type(self) -> str:
        return _MODEL_TYPES[self.model_name]

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node

    @property
    def data_parallel_size(self) -> int:
        return self.total_gpus // (32 if self.model_name == "GLM-5.3-Flash" else 4)

    @property
    def parallel_args(self) -> str:
        if self.model_name == "GLM-5.3-Flash":
            if self.total_gpus not in (32, 64):
                raise ValueError("The full GLM-5.3 image SFT recipe requires 32 or 64 GPUs")
            return (
                "--tensor-model-parallel-size 8 --sequence-parallel "
                "--pipeline-model-parallel-size 4 "
                "--decoder-first-pipeline-num-layers 11 --decoder-last-pipeline-num-layers 12 "
                "--context-parallel-size 1 "
                f"--expert-model-parallel-size {self.total_gpus // 4} --expert-tensor-parallel-size 1 "
            )
        # TP2 x PP2 fixes the model-parallel group at 4 GPUs; the rest becomes data parallel.
        if self.total_gpus < 8 or self.total_gpus % 4:
            raise ValueError("The 4-layer GLM-5.3 image SFT recipe requires at least 8 GPUs, in multiples of 4")
        return (
            "--tensor-model-parallel-size 2 --sequence-parallel "
            "--pipeline-model-parallel-size 2 --context-parallel-size 1 "
            "--expert-model-parallel-size 2 --expert-tensor-parallel-size 1 "
        )


def _train_request(args: ScriptArgs) -> dict:
    parallel_args = args.parallel_args
    checkpoint_path = shlex.quote(f"{args.output_dir}/checkpoints")
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(args.hf_checkpoint)} "
        f"--ref-load {shlex.quote(args.ref_load)} "
        f"--load {checkpoint_path} --save {checkpoint_path} --save-interval {args.save_interval} "
    )
    rollout_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {shlex.quote(args.prompt_data)} --input-key messages "
        """--multimodal-keys '{"image":"images"}' """
        # The shuffle seed is what makes two runs see the same data in the same order,
        # which any comparison between --vision-trainable scopes depends on.
        f"--loss-mask-type glm5 --rollout-shuffle --rollout-seed {args.rollout_seed} "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--global-batch-size {args.global_batch_size} "
        "--n-samples-per-prompt 1 --num-steps-per-rollout 1 "
        "--loss-type sft_loss --calculate-per-token-loss "
        "--disable-compute-advantages-and-returns --debug-train-only "
    )
    perf_args = (
        f"{parallel_args}"
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += "--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer "
    misc_args = (
        "--bf16 "
        # The local checkpoint's RoPE metadata overrides the model script's default.
        f"--rotary-base {args.rotary_base} "
        "--custom-model-provider-path miles_plugins.models.glm5_next.vision.glm5_next_vlm_model_provider "
        f"--glm5-vision-trainable {args.vision_trainable} "
        f"{'--glm5-freeze-language ' if args.freeze_language else ''}"
        f"{'--glm5-vision-recompute ' if args.resolved_vision_recompute else ''}"
        "--model-name glm5_next --qkv-format thd "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --distributed-timeout-minutes 60 "
    )
    train_args = (
        f"{ckpt_args} {rollout_args} {perf_args} {optimizer_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} {misc_args} {args.extra_args}"
    )
    extra_env_vars = {"TORCHINDUCTOR_COMPILE_THREADS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if cache_dir := os.environ.get(name):
            extra_env_vars[name] = cache_dir
    return {
        "train_args": train_args,
        "num_gpus_per_node": args.num_gpus_per_node,
        "megatron_model_type": args.megatron_model_type,
        "train_script": "train_async.py",
        "megatron_path": args.megatron_path,
        "extra_env_vars": extra_env_vars,
    }


def execute(args: ScriptArgs):
    args.create_backend().execute_train(**_train_request(args))


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
